from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time

from vision_service import VisionService
from robot_controller import RobotController

app = Flask(__name__)
CORS(app)

# Global services
vision_service = VisionService()
robot_controller = RobotController(simulate=True)

# Application state
class GameState:
    def __init__(self):
        self.internal_board = [[0 for _ in range(7)] for _ in range(6)]
        self.virtual_board = [[0 for _ in range(7)] for _ in range(6)]
        self.turn = "human" # "human" or "robot"
        self.robot_state = "idle" # "idle", "analyzing", "thinking", "moving"
        self.simulation_mode = True
        self.game_over = False
        self.winner = None
        self.match_state = "idle" # "idle", "in_game", "finished"
        
        # Validation State
        self.debounce_time = 1.0
        self.pending_board = None
        self.pending_board_time = 0
        self.error_msg = None
        self.invalid_stones = []
        
        # AI Config
        self.ai_enabled = True
        self.robot_target_col = None

state = GameState()

# We need to start/stop the vision service around the Flask app lifecycle.
# In a simple setup, we can start it on the first request, or right here.
vision_service.start()

import atexit
def cleanup():
    vision_service.stop()
    robot_controller.stop_robot_server()

atexit.register(cleanup)

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

@app.route("/api/board-state", methods=["GET"])
def get_board_state():
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
                if state.game_over:
                    state.game_over = False
                    state.winner = None
                    state.turn = "human"
                    state.match_state = "idle"
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
                        state.error_msg = "Game is not active. Please Start/Reset the game."
                        state.invalid_stones = [[r, c]]
                    elif (state.turn == "human" and p != 1) or (state.turn == "robot" and p != 2):
                        state.error_msg = f"Wrong token inserted! Expected Player {1 if state.turn == 'human' else 2} ({state.turn})."
                        state.invalid_stones = [[r, c]]
                    else:
                        # Valid single move!
                        state.internal_board = merged_board
                        state.error_msg = None
                        state.invalid_stones = []
                        
                        winner = check_winner(merged_board)
                        if winner is not None:
                            state.game_over = True
                            state.winner = winner
                            state.match_state = "finished"
                            state.robot_target_col = None
                        else:
                            if state.turn == "human":
                                state.turn = "robot"
                                if state.ai_enabled and state.robot_state == "idle":
                                    thread = threading.Thread(target=execute_robot_move, daemon=True)
                                    thread.start()
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
        "robot_target_col": state.robot_target_col
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
    else:
        state.turn = "robot"
        if state.ai_enabled and state.robot_state == "idle":
            thread = threading.Thread(target=execute_robot_move, daemon=True)
            thread.start()
            
    return jsonify({"status": "success", "board": state.internal_board})

def execute_robot_move():
    print(f"\n[AI] execute_robot_move triggered. AI Enabled: {state.ai_enabled}")
    if not state.ai_enabled:
        return
        
    state.robot_state = "analyzing"
    
    # Wait until the physical board is valid before computing a move
    if state.error_msg is not None:
        print("[AI] Physical board in error state. Waiting...")
    while state.error_msg is not None:
        time.sleep(0.5)
        
    time.sleep(0.5) # Fake delay for UI
    
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
        robot_controller.send_robot_column(best_move)
        
    state.robot_state = "waiting_for_drop"
    # Thread now exits! `get_board_state` will naturally advance the turn when the physical token is detected.

@app.route("/api/robot-move", methods=["POST"])
def trigger_robot_move():
    if state.turn != "robot" or not state.ai_enabled:
        return jsonify({"error": "Not robot's turn or AI disabled"}), 400
    
    # Start a background thread to execute the move without blocking the HTTP response
    thread = threading.Thread(target=execute_robot_move, daemon=True)
    thread.start()
    
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
            
    return jsonify({"status": "success"})

@app.route("/api/reset", methods=["POST"])
def reset_game():
    state.internal_board = [[0]*7 for _ in range(6)]
    state.virtual_board = [[0]*7 for _ in range(6)]
    state.turn = "human"
    state.robot_state = "idle"
    state.robot_target_col = None
    state.game_over = False
    state.winner = None
    state.match_state = "idle"
    return jsonify({"status": "success"})

@app.route("/api/start", methods=["POST"])
def start_game():
    # In the future this can be expanded, for now it ensures game state is active
    state.game_over = False
    state.winner = None
    state.match_state = "in_game"
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
