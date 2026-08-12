from flask import Flask, request, jsonify, Response
import json
import os
from flask_cors import CORS
import threading
import time
import traceback

from vision_service import VisionService
from robot_controller import RobotController
from nfc_reader import start_nfc_reader

app = Flask(__name__)
CORS(app)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# How long a fresh connection gets to report a token already in the gripper
# before we treat its silence as "empty" and send the grab code. Only pendants
# that report nothing at all fall back on this: a robot that says GRABBED or
# EMPTY has answered the question, and the grace is skipped. It exists because
# asking a silent-but-loaded robot for a token inside that window is exactly
# what sends it off to fetch a second one. Only has to cover connect -> first
# send on the pendant, which is immediate; it is not waiting for arm movement.
CONNECT_ANNOUNCE_GRACE = 2.0

# How often to mention that no robot is on the wire. The detection loop notices
# 50x a second, which is far too often to print.
ROBOT_ABSENT_LOG_INTERVAL = 15.0

# How often to complain that a grab code went out and no GRABBED came back.
GRAB_ACK_WARN_INTERVAL = 15.0

# How long a grab code stays "possibly still being worked on" after it went out.
# An EMPTY that arrives inside this window cancels the outstanding grab but does
# not get the code re-sent yet, because a pendant that reports EMPTY on its way
# to the pickup station would otherwise collect a second 8 in its buffer -- the
# double grab the whole handshake exists to prevent. Has to be longer than a
# pick-up cycle: past it, a robot that really grabbed something has long since
# said GRABBED, so nothing is ever re-sent to it.
GRAB_RESEND_COOLDOWN = 10.0

# Shown verbatim in the GUI's error banner, so it is German like the rest of the
# interface. A named constant because the gravity check both sets and clears it
# by value -- two copies of the text could drift apart and leave the banner
# stuck on screen.
GRAVITY_ERROR = "Schwerkraft-Prüfung fehlgeschlagen! (Hand im Weg oder schwebender Stein)"

DEFAULT_SETTINGS = {
    "simulation_mode": False,
    "debounce_time": 0.5,
    "ai_enabled": True,
    "difficulty": "medium",
    "nfc_timeout": 15.0,
}

def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE) as f:
            stored = json.load(f)
    except FileNotFoundError:
        return settings
    except (OSError, ValueError) as e:
        print(f"Could not read {SETTINGS_FILE} ({e}); falling back to defaults")
        return settings

    if not isinstance(stored, dict):
        print(f"Ignoring malformed {SETTINGS_FILE}; falling back to defaults")
        return settings

    for key in DEFAULT_SETTINGS:
        if key in stored:
            settings[key] = stored[key]

    settings["simulation_mode"] = bool(settings["simulation_mode"])
    settings["ai_enabled"] = bool(settings["ai_enabled"])
    settings["difficulty"] = str(settings["difficulty"])
    try:
        settings["debounce_time"] = float(settings["debounce_time"])
    except (TypeError, ValueError):
        settings["debounce_time"] = DEFAULT_SETTINGS["debounce_time"]
    try:
        settings["nfc_timeout"] = float(settings["nfc_timeout"])
    except (TypeError, ValueError):
        settings["nfc_timeout"] = DEFAULT_SETTINGS["nfc_timeout"]
    return settings

def save_settings():
    settings = {
        "simulation_mode": state.simulation_mode,
        "debounce_time": state.debounce_time,
        "ai_enabled": state.ai_enabled,
        "difficulty": robot_controller.difficulty_name,
        "nfc_timeout": state.nfc_timeout,
    }
    tmp_path = f"{SETTINGS_FILE}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp_path, SETTINGS_FILE)
    except OSError as e:
        print(f"Could not save settings: {e}")

settings = load_settings()

# Global services
vision_service = VisionService()
robot_controller = RobotController(
    simulate=settings["simulation_mode"],
    difficulty=settings["difficulty"],
)

# Application state
class GameState:
    def __init__(self, settings):
        self.internal_board = [[0 for _ in range(7)] for _ in range(6)]
        self.virtual_board = [[0 for _ in range(7)] for _ in range(6)]
        self.turn = "human" # "human" or "robot"
        self.robot_state = "idle" # "idle", "analyzing", "thinking", "moving"
        self.simulation_mode = settings["simulation_mode"]
        self.game_over = False
        self.winner = None
        self.match_state = "in_game" # "in_game", "finished"

        # Validation State
        self.debounce_time = settings["debounce_time"]
        self.pending_board = None
        self.pending_board_time = 0
        self.error_msg = None
        self.invalid_stones = []

        # AI Config
        self.ai_enabled = settings["ai_enabled"]
        self.robot_target_col = None
        # Guards the idle -> analyzing transition so only one move thread runs.
        self.robot_move_lock = threading.Lock()
        # True once the grab code went out and no GRABBED has come back yet.
        # Only ever cleared by something that really does invalidate it: the
        # robot reconnecting (its gripper state is reported from scratch), the
        # robot saying EMPTY, or the token landing on the board. Not by a reset
        # -- see reset_game_state().
        self.robot_stone_requested = False
        # True once the robot said GRABBED, either in answer to a grab code or
        # unprompted right after connecting to report a token it was already
        # holding. The column of a move is held back until then, so the grab
        # code and the column can never sit unread in the pendant's buffer
        # together -- without terminators it would read them as one garbled
        # message. Cleared when the robot says EMPTY.
        self.robot_stone_held = False
        # True once the robot has stated its gripper on this connection, either
        # way round. Until then all we have is silence, which CONNECT_ANNOUNCE_GRACE
        # has to sit out before reading it as "empty".
        self.robot_gripper_reported = False
        # When the outstanding grab code went out, so a missing ack can be
        # complained about on a slow beat instead of silently.
        self.robot_stone_requested_time = 0.0
        self.robot_grab_warned = 0.0
        # Set when an EMPTY cancels a grab code that may still be in progress:
        # the code is re-sent once the robot has had GRAB_RESEND_COOLDOWN to
        # finish the pick-up it was asked for, not before.
        self.robot_grab_blocked_until = 0.0
        self.robot_absent_logged = 0.0
        # When the current robot connection was accepted, so CONNECT_ANNOUNCE_GRACE
        # can be measured from it.
        self.robot_connected_at = 0.0

        # NFC Reader State
        self.nfc_timeout = settings["nfc_timeout"]
        self.nfc_data = None
        self.nfc_scan_time = 0
        self.nfc_invalid_scan_time = 0

state = GameState(settings)

def difficulty_toggle_allowed():
    # The button may only change difficulty between games, i.e. while the
    # physical board is cleared and no move is in flight.
    return count_tokens(state.internal_board) == 0 and state.robot_state == "idle"

def on_difficulty_changed(name):
    save_settings()

def on_robot_connected():
    # Step 2 of the handshake starts here. A new connection tells us nothing
    # about the gripper -- the pendant may have restarted empty, or it may have
    # been power-cycled mid-move and still be clamping a token. So we drop what
    # we believed and let the robot state it: GRABBED if it is holding one,
    # EMPTY if it is not. maintain_robot_stone() falls back on
    # CONNECT_ANNOUNCE_GRACE only for a pendant that says neither.
    state.robot_stone_requested = False
    state.robot_stone_held = False
    state.robot_gripper_reported = False
    state.robot_connected_at = time.time()
    state.robot_grab_warned = 0.0
    state.robot_grab_blocked_until = 0.0

def on_stone_grabbed():
    # Step 4, and also step 2: the same ack serves as the unprompted "I am
    # already holding one" report on a fresh connection, which is why it is not
    # gated on robot_stone_requested.
    state.robot_stone_held = True
    state.robot_stone_requested = False
    state.robot_gripper_reported = True
    # Whatever grab was outstanding has been answered, so nothing is owed a
    # cooldown any more.
    state.robot_grab_blocked_until = 0.0

def on_stone_empty():
    # The robot states an empty gripper: on connect, after a drop, and after its
    # reset routine returned a token. Believing it is the point -- it is the one
    # side that can see the gripper -- so the grab code goes out again from
    # maintain_robot_stone(), including after a reset, where the backend
    # deliberately keeps its own belief untouched (see reset_game_state()).
    if state.robot_stone_requested:
        # A grab we asked for that the robot has not (yet) turned into a token.
        # Cancel it so it stops blocking, but let the cooldown decide when to
        # ask again: an EMPTY reported on the way to the pickup station must not
        # buy a second 8.
        state.robot_grab_blocked_until = (
            state.robot_stone_requested_time + GRAB_RESEND_COOLDOWN
        )
    state.robot_stone_requested = False
    state.robot_stone_held = False
    state.robot_gripper_reported = True

robot_controller.difficulty_toggle_guard = difficulty_toggle_allowed
robot_controller.on_difficulty_changed = on_difficulty_changed
robot_controller.on_robot_connected = on_robot_connected
robot_controller.on_stone_grabbed = on_stone_grabbed
robot_controller.on_stone_empty = on_stone_empty
# Defined further down; the robot's reset button does exactly what the web UI's
# reset button does.
robot_controller.on_reset_requested = lambda: reset_game_state()

def start_robot_move():
    with state.robot_move_lock:
        if not state.ai_enabled or state.robot_state != "idle":
            return False
        state.robot_state = "analyzing"
    threading.Thread(target=execute_robot_move, daemon=True).start()
    return True

def on_nfc_scan(tag_data):
    # Only allow saving tag if the board is empty (no stones inserted)
    if count_tokens(state.internal_board) == 0 and state.robot_state == "idle":
        state.nfc_data = tag_data
        state.nfc_scan_time = time.time()
    else:
        # Tried scanning mid-game
        state.nfc_invalid_scan_time = time.time()

nfc_thread = start_nfc_reader(on_nfc_scan)

# We need to start/stop the vision service around the Flask app lifecycle.
# In a simple setup, we can start it on the first request, or right here.
vision_service.start()

import atexit
import signal

_cleanup_started = False

def cleanup():
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True
    vision_service.stop()
    robot_controller.stop_robot_server()

atexit.register(cleanup)

def handle_sigterm(signum, frame):
    # The default SIGTERM disposition skips atexit, which would leave the camera
    # and port 8000 held by a process that never ran cleanup().
    if _cleanup_started:
        # A shutdown is already in flight (SIGINT -> atexit). Interrupting it
        # while librealsense is mid-teardown is what dumps core, so let it run.
        return
    cleanup()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

def count_tokens(board):
    return sum(1 for row in board for cell in row if cell != 0)

def count_player_tokens(board, player_id):
    return sum(1 for row in board for cell in row if cell == player_id)

def check_winner(board):
    # Check horizontal
    for r in range(6):
        for c in range(4):
            if board[r][c] != 0 and board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                return board[r][c]
    # Check vertical
    for c in range(7):
        for r in range(3):
            if board[r][c] != 0 and board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                return board[r][c]
    # Check positive diagonal
    for r in range(3):
        for c in range(4):
            if board[r][c] != 0 and board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                return board[r][c]
    # Check negative diagonal
    for r in range(3, 6):
        for c in range(4):
            if board[r][c] != 0 and board[r][c] == board[r-1][c+1] == board[r-2][c+2] == board[r-3][c+3]:
                return board[r][c]
    
    # Check draw
    if count_tokens(board) == 42:
        return 0 # Draw
        
    return None

def merge_boards(cv_board, virtual_board):
    res = [[0]*7 for _ in range(6)]
    for r in range(6):
        for c in range(7):
            res[r][c] = cv_board[r][c] or virtual_board[r][c]
    return res

def process_board_update():
    # Advances the detection state machine one step. Driven by detection_loop()
    # rather than by the board-state request handler: when this ran inside the
    # handler, the debounce could only elapse across two polls, so a 0.2s
    # debounce cost a full poll interval (1s) or more before the robot started.
    cv_board = vision_service.get_board_state()

    # If physical board is completely cleared, reset the virtual board too
    if count_tokens(cv_board) == 0:
        state.virtual_board = [[0 for _ in range(7)] for _ in range(6)]
        
    merged_board = merge_boards(cv_board, state.virtual_board)
    
    # 1. Gravity check on merged_board
    gravity_violation = False
    for r in range(5):
        for c in range(7):
            if merged_board[r][c] != 0 and merged_board[r+1][c] == 0:
                gravity_violation = True
                break
        if gravity_violation:
            break
            
    if gravity_violation:
        state.error_msg = GRAVITY_ERROR
    else:
        # If gravity is resolved but we had a gravity error, clear it immediately
        if state.error_msg == GRAVITY_ERROR:
            state.error_msg = None
            
        # 2. Debounce logic
        if state.pending_board != merged_board:
            state.pending_board = merged_board
            state.pending_board_time = time.time()
            
        if time.time() - state.pending_board_time >= state.debounce_time:
            # 3. Validation
            new_total = count_tokens(merged_board)
            if new_total == 0:
                # User physically cleared the entire board
                state.internal_board = merged_board
                state.error_msg = None
                state.invalid_stones = []
                state.robot_target_col = None
                # A robot move that ends the game leaves robot_state at
                # "waiting_for_drop", since no further token detection follows to
                # advance the turn. Clearing the board is a fresh start, so the
                # robot is idle again -- without this, start_robot_move() would
                # refuse to ever run again.
                state.robot_state = "idle"
                state.game_over = False
                state.winner = None
                state.turn = "human"
                state.match_state = "in_game"
                # Same reason reset_game_state() drops it: a column left over
                # from the game just abandoned would otherwise be flushed to the
                # robot on the next GRABBED ack, sending it to a column of a
                # board that no longer exists.
                robot_controller.clear_pending_column()
                # robot_stone_requested/_held are deliberately NOT cleared here.
                # Clearing the board does not touch the arm, so a token it
                # grabbed for the game just abandoned is still in its gripper;
                # claiming otherwise would send it off to fetch a second one.
            else:
                # Strict superset check
                missing_stones = False
                new_stones = []
                for r in range(6):
                    for c in range(7):
                        if state.internal_board[r][c] != 0 and merged_board[r][c] == 0:
                            missing_stones = True
                        elif state.internal_board[r][c] != 0 and merged_board[r][c] != state.internal_board[r][c]:
                            missing_stones = True
                        elif state.internal_board[r][c] == 0 and merged_board[r][c] != 0:
                            new_stones.append([r, c, merged_board[r][c]])
                            
                if missing_stones:
                    state.error_msg = "Steine wurden unerwartet entfernt oder verändert! Bitte das Spielfeld wiederherstellen."
                    state.invalid_stones = []
                elif len(new_stones) > 1:
                    state.error_msg = "Zu viele Steine auf einmal eingeworfen! Bitte die überzähligen Steine entfernen."
                    state.invalid_stones = [[r, c] for r, c, p in new_stones]
                elif len(new_stones) == 1:
                    r, c, p = new_stones[0]
                    if state.match_state != "in_game":
                        state.error_msg = "Das Spiel läuft nicht. Bitte ein neues Spiel starten."
                        state.invalid_stones = [[r, c]]
                    elif (state.turn == "human" and p != 1) or (state.turn == "robot" and p != 2):
                        state.error_msg = f"Falscher Stein eingeworfen! Erwartet: {'Mensch' if state.turn == 'human' else 'Roboter'} (Spieler {1 if state.turn == 'human' else 2})."
                        state.invalid_stones = [[r, c]]
                    else:
                        # Valid single move!
                        state.internal_board = merged_board
                        state.error_msg = None
                        state.invalid_stones = []

                        if p == 2:
                            # The token the robot was holding is now on the
                            # board, so it needs a new one -- unless the game
                            # ends right here, which the check below decides.
                            # Seeing it land settles the gripper question, so a
                            # cooldown from an earlier EMPTY is stale too.
                            state.robot_stone_requested = False
                            state.robot_stone_held = False
                            state.robot_grab_blocked_until = 0.0

                        winner = check_winner(merged_board)
                        if winner is not None:
                            state.game_over = True
                            state.winner = winner
                            state.match_state = "finished"
                            state.robot_target_col = None
                            robot_controller.send_game_result(robot_won=(winner == 2))
                        else:
                            if state.turn == "human":
                                state.turn = "robot"
                                start_robot_move()
                            else:
                                state.turn = "human"
                                state.robot_state = "idle"
                                state.robot_target_col = None
                else:
                    # len(new_stones) == 0 -> Board hasn't changed.
                    # Since missing_stones is False and new_stones is 0, merged_board exactly equals internal_board.
                    # We can clear any existing transient errors.
                    state.error_msg = None
                    state.invalid_stones = []


def maintain_robot_stone():
    """Steps 1-3: make sure the robot is holding a token while a game runs.

    The robot is the only side that can see its own gripper, so this never
    assumes -- it waits to be told. Driven from the detection loop rather than
    from the turn change alone, so a grab code that could not be delivered (not
    connected yet, socket busy) is simply attempted again on the next pass
    instead of leaving the robot empty-handed forever.
    """
    if state.simulation_mode:
        # No pendant to hand a token to, and run_robot_move() skips the
        # handshake entirely. Sending grab codes into the simulator would only
        # pin the debug view at "grab requested, never acked".
        return
    if state.game_over or state.match_state != "in_game":
        return
    if state.robot_stone_held:
        # Step 5 is unblocked; nothing to do until the token leaves the gripper.
        return
    if state.robot_state == "waiting_for_drop":
        # The column went out and the token is on its way into the board. A
        # robot that reports EMPTY before the camera sees it land is describing
        # that drop, not asking for the next token -- and the drop may be the
        # winning move, in which case an 8 sent now would race the 9/0 result
        # code and leave two unread codes in the pendant's buffer. The detection
        # path clears the gripper flags itself once it sees the token, after it
        # has decided whether the game is over.
        return

    # Step 1: wait for the robot to connect. It is the side that dials us, so a
    # pendant program started before this server was listening never connects
    # and never retries. Staying silent about that is what makes a start-order
    # problem look like a game that refuses to move.
    if not robot_controller.is_robot_connected:
        state.robot_stone_requested = False
        now = time.time()
        if now - state.robot_absent_logged >= ROBOT_ABSENT_LOG_INTERVAL:
            state.robot_absent_logged = now
            print(f"[robot] Nothing connected on port {robot_controller.robot_server_port}; "
                  "waiting for the pendant program to dial in")
        return

    # Step 2: a fresh connection that has said neither GRABBED nor EMPTY gets a
    # moment to report a token already in the gripper before its silence is read
    # as "empty". A robot that reported either way has already answered.
    if (not state.robot_gripper_reported
            and time.time() - state.robot_connected_at < CONNECT_ANNOUNCE_GRACE):
        return

    if state.robot_stone_requested:
        # Step 4 outstanding. The grab code is deliberately NOT re-sent on a
        # timer: TCP already redelivers within a connection, and a second 8
        # overtaking a slow ack is precisely a double grab. Only the robot
        # itself can call the grab off, by reporting EMPTY.
        now = time.time()
        waited = now - state.robot_stone_requested_time
        if waited >= GRAB_ACK_WARN_INTERVAL and now - state.robot_grab_warned >= GRAB_ACK_WARN_INTERVAL:
            state.robot_grab_warned = now
            print(f"[robot] Grab code sent {waited:.0f}s ago, still no GRABBED "
                  "-- is the pendant program acking the grab?")
        return

    if time.time() < state.robot_grab_blocked_until:
        # An EMPTY cancelled a grab code that had only just gone out. Wait out
        # the pick-up it might still be performing before asking again.
        return

    # Step 3: as far as the robot has told us, the gripper is empty. Ask.
    if robot_controller.send_game_continues():
        state.robot_stone_requested = True
        state.robot_stone_requested_time = time.time()
        state.robot_grab_warned = 0.0


def detection_loop():
    while True:
        try:
            process_board_update()
            maintain_robot_stone()
            
            # Clear expired NFC data
            if state.nfc_data and time.time() - state.nfc_scan_time > state.nfc_timeout:
                state.nfc_data = None
                
        except Exception as e:
            print(f"[detection] error: {e}")
        time.sleep(0.02)

threading.Thread(target=detection_loop, daemon=True).start()

@app.route("/api/board-state", methods=["GET"])
def get_board_state():
    # Pure read of state; the detection thread owns the transitions.
    return jsonify({
        "board": state.internal_board,
        "turn": state.turn,
        "robot_state": state.robot_state,
        "simulation_mode": state.simulation_mode,
        "game_over": state.game_over,
        "winner": state.winner,
        "match_state": state.match_state,
        "error_msg": state.error_msg,
        "invalid_stones": state.invalid_stones,
        "debounce_time": state.debounce_time,
        "ai_enabled": state.ai_enabled,
        "difficulty": robot_controller.difficulty_name,
        "robot_target_col": state.robot_target_col,
        "tcp_connected": robot_controller.is_robot_connected,
        "nfc_connected": os.path.exists('/dev/ttyUSB0'),
        "nfc_data": state.nfc_data,
        "nfc_invalid_scan_time": state.nfc_invalid_scan_time,
        "nfc_timeout": state.nfc_timeout,
        # Handshake state. A move sits in run_robot_move() until stone_held goes
        # true, so "grab_requested true, stone_held false" is the signature of a
        # column being withheld because the robot never sent GRABBED.
        "grab_requested": state.robot_stone_requested,
        "stone_held": state.robot_stone_held,
        "robot_connects": robot_controller.connection_count
    })

@app.route("/api/player-move", methods=["POST"])
def player_move():
    data = request.json
    col = data.get("column")
    player = data.get("player")
    
    if col is None or col < 0 or col > 6:
        return jsonify({"error": "Invalid column"}), 400
        
    if state.turn == "human":
        return jsonify({"error": "Not human's turn"}), 400
        
    if state.match_state != "in_game":
        return jsonify({"error": "Game not started"}), 400
        
    for row in range(5, -1, -1):
        if state.internal_board[row][col] == 0:
            state.internal_board[row][col] = player
            state.virtual_board[row][col] = player
            break
            
    winner = check_winner(state.internal_board)
    if winner is not None:
        state.game_over = True
        state.winner = winner
        state.match_state = "finished"
        state.robot_target_col = None
        robot_controller.send_game_result(robot_won=(winner == 2))
    else:
        state.turn = "robot"
        start_robot_move()

    return jsonify({"status": "success", "board": state.internal_board})

def execute_robot_move():
    try:
        run_robot_move()
    except Exception:
        traceback.print_exc()
        # A thread that dies mid-move would leave robot_state at "thinking" or
        # "moving", and start_robot_move() only ever leaves "idle" -- the robot
        # would never move again for the rest of the game.
        state.robot_state = "idle"
        state.robot_target_col = None

def run_robot_move():
    print(f"\n[AI] execute_robot_move triggered. AI Enabled: {state.ai_enabled}")

    # Wait until the physical board is valid before computing a move
    if state.error_msg is not None:
        print("[AI] Physical board in error state. Waiting...")
    while state.error_msg is not None:
        time.sleep(0.1)

    state.robot_state = "thinking"
    board_for_ai = state.internal_board
    if not state.simulation_mode:
        board_for_ai = vision_service.get_board_state()
        
    print("[AI] Getting AI move from solver...")
    best_move = robot_controller.get_ai_move(board_for_ai)
    print(f"[AI] Solver returned move: {best_move}")
    
    if best_move is None:
        state.robot_state = "idle"
        return
        
    state.robot_target_col = best_move
    state.robot_state = "moving"
    print(f"[AI] Target column set to {best_move}. Simulation Mode: {state.simulation_mode}")
    
    # Execute movement (either TCP or wait for manual drop)
    if not state.simulation_mode:
        # Step 5: hold the column back until the robot confirmed it is holding a
        # token, even if the human answered before it reached the pickup
        # station. maintain_robot_stone() is what drives steps 1-4 in the
        # background; this only waits for the result.
        start = time.time()
        last_warn = 0.0
        while not state.robot_stone_held:
            if state.game_over or state.match_state != "in_game" or state.robot_state != "moving":
                # Reset or board cleared while waiting; this move is stale.
                if state.robot_state == "moving":
                    state.robot_state = "idle"
                state.robot_target_col = None
                return
            # Keep saying it. maintain_robot_stone() logs why it is stuck (no
            # connection, or a grab code with no ack); this says what that is
            # costing, namely a computed move nobody can act on.
            waited = time.time() - start
            if waited > 15 and waited - last_warn >= 15:
                last_warn = waited
                print(f"[AI] Column {best_move + 1} withheld for {waited:.0f}s, "
                      "waiting for the robot to confirm it holds a token")
            time.sleep(0.05)
        robot_controller.send_robot_column(best_move)

    state.robot_state = "waiting_for_drop"
    # Thread now exits! `get_board_state` will naturally advance the turn when the physical token is detected.

@app.route("/api/robot-move", methods=["POST"])
def trigger_robot_move():
    if state.turn != "robot" or not state.ai_enabled:
        return jsonify({"error": "Not robot's turn or AI disabled"}), 400

    if not start_robot_move():
        return jsonify({"status": "already computing"})

    return jsonify({"status": "started computation"})

@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.get_json() or {}
    
    if "ai_enabled" in data:
        state.ai_enabled = data["ai_enabled"]
        
    if "simulate" in data:
        simulate = data["simulate"]
        if not simulate and state.simulation_mode:
            state.virtual_board = [[0 for _ in range(7)] for _ in range(6)]
        state.simulation_mode = simulate
        robot_controller.simulate = simulate
        
        if simulate:
            robot_controller.stop_robot_server()
        else:
            robot_controller.start_robot_server()
            
    if "difficulty" in data:
        robot_controller.set_difficulty(data["difficulty"])
        
    if "debounce_time" in data:
        try:
            state.debounce_time = float(data["debounce_time"])
        except ValueError:
            pass

    if "nfc_timeout" in data:
        try:
            state.nfc_timeout = float(data["nfc_timeout"])
        except ValueError:
            pass

    save_settings()
    return jsonify({"status": "success"})

def reset_game_state():
    state.internal_board = [[0]*7 for _ in range(6)]
    state.virtual_board = [[0]*7 for _ in range(6)]
    state.turn = "human"
    state.robot_state = "idle"
    state.robot_target_col = None
    state.game_over = False
    state.winner = None
    state.match_state = "in_game"
    state.error_msg = None
    state.invalid_stones = []

    # Drop the column of the move the reset just cancelled, so it cannot be
    # flushed to the robot on the next GRABBED ack and send it to a column from
    # the game we just abandoned.
    robot_controller.clear_pending_column()
    # robot_stone_requested/_held are deliberately NOT cleared, for the same
    # reason the board-cleared path above leaves them alone: a reset is a game
    # event, not a robot command -- nothing here tells the arm to open its
    # gripper, so a token it grabbed for the abandoned game is still in there.
    # Clearing the belief would send it off to fetch a second one, which is the
    # failure this handshake exists to prevent. The robot is the side that knows
    # what its reset routine did with the token: if that routine put it back, it
    # says EMPTY, on_stone_empty() clears the belief, and the grab code goes out
    # for the new game.

@app.route("/api/reset", methods=["POST"])
def reset_game():
    reset_game_state()
    return jsonify({"status": "success"})

def gen_frames():
    while True:
        frame = vision_service.get_annotated_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route("/api/video-feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
