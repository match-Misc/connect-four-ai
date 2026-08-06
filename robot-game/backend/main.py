from flask import Flask, request, jsonify, Response
import json
import os
from flask_cors import CORS
import threading
import time
import traceback

from vision_service import VisionService
from robot_controller import RobotController

app = Flask(__name__)
CORS(app)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# How long to wait for the robot's GRABBED ack before assuming the grab code
# never reached it and sending it again. Generous on purpose: it has to cover
# the arm's whole trip to the pickup station, and a retry that overtakes a slow
# ack makes the robot fetch a second token.
GRAB_ACK_TIMEOUT = 20.0

# How often to mention that no robot is on the wire. The detection loop notices
# 50x a second, which is far too often to print.
ROBOT_ABSENT_LOG_INTERVAL = 15.0

DEFAULT_SETTINGS = {
    "simulation_mode": False,
    "debounce_time": 0.5,
    "ai_enabled": True,
    "difficulty": "medium",
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
    return settings

def save_settings():
    settings = {
        "simulation_mode": state.simulation_mode,
        "debounce_time": state.debounce_time,
        "ai_enabled": state.ai_enabled,
        "difficulty": robot_controller.difficulty_name,
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
        # True once the robot has been told to pick up a token and has not yet
        # dropped it. Cleared on a reset: the pendant's reset returns the arm to
        # its start position with an empty gripper, so a token we believe it
        # grabbed before the reset is gone.
        self.robot_stone_requested = False
        # True once the robot ACKED the grab (sent GRABBED). The column of a
        # move is held back until then, so the grab code and the column can
        # never sit unread in the pendant's buffer together -- without
        # terminators it would read them as one garbled message. Cleared on a
        # reset for the same reason as robot_stone_requested.
        self.robot_stone_held = False
        # When the outstanding grab code went out, so a missing ack can be
        # retried instead of stalling the game for good.
        self.robot_stone_requested_time = 0.0
        self.robot_absent_logged = 0.0

state = GameState(settings)

def difficulty_toggle_allowed():
    # The button may only change difficulty between games, i.e. while the
    # physical board is cleared and no move is in flight.
    return count_tokens(state.internal_board) == 0 and state.robot_state == "idle"

def on_difficulty_changed(name):
    save_settings()

def on_robot_connected():
    # The pendant program is back at the top of its loop with an empty gripper,
    # so any token we think it grabbed before the restart is gone.
    state.robot_stone_requested = False
    state.robot_stone_held = False

def on_stone_grabbed():
    state.robot_stone_held = True

robot_controller.difficulty_toggle_guard = difficulty_toggle_allowed
robot_controller.on_difficulty_changed = on_difficulty_changed
robot_controller.on_robot_connected = on_robot_connected
robot_controller.on_stone_grabbed = on_stone_grabbed
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
        state.error_msg = "Gravity check failed! (Hand in the way or floating chip)"
    else:
        # If gravity is resolved but we had a gravity error, clear it immediately
        if state.error_msg == "Gravity check failed! (Hand in the way or floating chip)":
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
                    state.error_msg = "Stone(s) removed or altered unexpectedly! Please restore the board."
                    state.invalid_stones = []
                elif len(new_stones) > 1:
                    state.error_msg = "Too many stones inserted at once! Please remove the extra stones."
                    state.invalid_stones = [[r, c] for r, c, p in new_stones]
                elif len(new_stones) == 1:
                    r, c, p = new_stones[0]
                    if state.match_state != "in_game":
                        state.error_msg = "Game is not active. Please reset the game."
                        state.invalid_stones = [[r, c]]
                    elif (state.turn == "human" and p != 1) or (state.turn == "robot" and p != 2):
                        state.error_msg = f"Wrong token inserted! Expected Player {1 if state.turn == 'human' else 2} ({state.turn})."
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
                            state.robot_stone_requested = False
                            state.robot_stone_held = False

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
    """Keep a token in the robot's gripper while a game is running.

    Driven from the detection loop rather than from the turn change alone, so a
    signal that did not reach the robot (not connected yet, socket busy) is just
    retried on the next pass instead of leaving it empty-handed forever.
    """
    if state.game_over or state.match_state != "in_game":
        return
    if state.robot_stone_held:
        return
    if state.robot_stone_requested:
        # The grab code went out but no GRABBED came back yet. A send the kernel
        # accepted is no proof the robot read it: it may be powered off behind a
        # half-open socket, restarting, or have skipped the grab because its
        # gripper was still full. Gating on robot_stone_requested alone deadlocks
        # the game for good in that case -- execute_robot_move() waits on an ack
        # that never arrives, so the column is never sent and the robot just
        # stands there.
        if time.time() - state.robot_stone_requested_time < GRAB_ACK_TIMEOUT:
            return
        print(f"[robot] No GRABBED ack after {GRAB_ACK_TIMEOUT:.0f}s; re-sending the grab code")

    if robot_controller.send_game_continues():
        state.robot_stone_requested = True
        state.robot_stone_requested_time = time.time()
    else:
        # Not delivered (robot not connected, socket busy) -- the next pass
        # through the detection loop tries again. Say so on a slow beat: the
        # robot is the side that dials us, so a pendant program started before
        # this server was listening never connects and never retries. Staying
        # silent about it is what makes a start-order problem look like a game
        # that refuses to move.
        state.robot_stone_requested = False
        now = time.time()
        if now - state.robot_absent_logged >= ROBOT_ABSENT_LOG_INTERVAL:
            state.robot_absent_logged = now
            print(f"[robot] Nothing connected on port {robot_controller.robot_server_port}; "
                  "waiting for the pendant program to dial in")


def detection_loop():
    while True:
        try:
            process_board_update()
            maintain_robot_stone()
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
        # Handshake: hold the column back until the robot confirmed the grab,
        # even if the human answered before it reached the pickup station.
        start = time.time()
        last_warn = 0.0
        while not state.robot_stone_held:
            if state.game_over or state.match_state != "in_game" or state.robot_state != "moving":
                # Reset or board cleared while waiting; this move is stale.
                if state.robot_state == "moving":
                    state.robot_state = "idle"
                state.robot_target_col = None
                return
            # Keep saying it: maintain_robot_stone() re-sends the grab code, so a
            # wait that outlives a couple of those means the pendant never acks
            # at all, which is the one thing the retry cannot fix.
            waited = time.time() - start
            if waited > 15 and waited - last_warn >= 15:
                last_warn = waited
                print(f"[AI] Still waiting for the robot's GRABBED ack after {waited:.0f}s "
                      "-- is the pendant program sending it?")
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

    # Drop the column of the move the reset just cancelled before forgetting the
    # grab, so it cannot be flushed to the robot on the next GRABBED ack and
    # send it to a column from the game we just abandoned.
    robot_controller.clear_pending_column()
    # A reset puts the pendant back at its start position with an empty gripper,
    # so whatever we believed about the token in it no longer holds. Without
    # this, maintain_robot_stone() still thinks the robot is holding one and
    # never sends the grab code again, leaving it empty-handed for good.
    state.robot_stone_requested = False
    state.robot_stone_held = False

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
