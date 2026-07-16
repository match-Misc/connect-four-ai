import threading
import socket
import os
import time
from typing import Optional, List

try:
    from connect_four_ai import AIPlayer, Difficulty, Position
except ImportError as e:
    print(f"Warning: could not import connect_four_ai ({e})")
    AIPlayer = None
    Difficulty = None
    Position = None

class RobotController:
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        
        # AI Player
        self.ai_player = None
        self.difficulty = None
        if AIPlayer:
            self.set_difficulty("medium")
            
        # Robot Server TCP
        self.robot_server_host = os.environ.get("C4_SERVER_HOST", "0.0.0.0")
        self.robot_server_port = int(os.environ.get("C4_SERVER_PORT", "30020"))
        self.robot_server_socket = None
        self.robot_client_socket = None
        self.robot_server_thread = None
        self.robot_server_running = False
        self.robot_conn_lock = threading.Lock()
        self.pending_column = None

        if not self.simulate:
            self.start_robot_server()

    def set_difficulty(self, diff_str: str):
        if not AIPlayer:
            return
        diff_map = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
            "impossible": Difficulty.IMPOSSIBLE
        }
        self.difficulty = diff_map.get(diff_str.lower(), Difficulty.MEDIUM)
        self.ai_player = AIPlayer(self.difficulty)

    def get_ai_move(self, board_state: List[List[int]]) -> Optional[int]:
        print(f"[Solver] get_ai_move called. AI Player initialized: {self.ai_player is not None}")
        if not self.ai_player:
            return None
        # Convert 2D list to board string for AI
        board_chars = []
        for row in range(6):
            for col in range(7):
                cell = board_state[row][col]
                if cell == 1:
                    board_chars.append("o")  # Opponent
                elif cell == 2:
                    board_chars.append("x")  # Current player (AI)
                else:
                    board_chars.append(".")
        board_string = "".join(board_chars)
        print(f"[Solver] Formatted board string:\n{board_string[:7]}\n{board_string[7:14]}\n{board_string[14:21]}\n{board_string[21:28]}\n{board_string[28:35]}\n{board_string[35:42]}")
        pos = Position.from_board_string(board_string)
        print("[Solver] Parsed Position successfully")
        try:
            move = self.ai_player.get_move(pos)
            print(f"[Solver] get_move returned: {move}")
            return move
        except Exception as e:
            import traceback
            print(f"[Solver] Error getting AI move: {e}")
            traceback.print_exc()
            return None

    def start_robot_server(self):
        try:
            self.robot_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.robot_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.robot_server_socket.bind((self.robot_server_host, self.robot_server_port))
            self.robot_server_socket.listen(1)
            self.robot_server_running = True

            def accept_loop():
                while self.robot_server_running:
                    try:
                        self.robot_server_socket.settimeout(1.0)
                        conn, addr = self.robot_server_socket.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    
                    with self.robot_conn_lock:
                        if self.robot_client_socket:
                            try:
                                self.robot_client_socket.close()
                            except:
                                pass
                        try:
                            conn.setblocking(False)
                        except:
                            pass
                        self.robot_client_socket = conn
                    
                    print(f"Robot connected from {addr}")
                    with self.robot_conn_lock:
                        pending = self.pending_column
                        send_conn = self.robot_client_socket
                        self.pending_column = None
                        
                    if pending is not None and send_conn:
                        try:
                            msg = str(int(pending)).encode("ascii")
                            send_conn.send(msg)
                            print(f"Sent pending column to robot: {pending}")
                        except Exception as e:
                            print(f"Failed to send pending column: {e}")
                            with self.robot_conn_lock:
                                self.pending_column = pending

            self.robot_server_thread = threading.Thread(target=accept_loop, daemon=True)
            self.robot_server_thread.start()
            print(f"Robot server listening on {self.robot_server_host}:{self.robot_server_port}")
        except Exception as e:
            print(f"Failed to start robot server: {e}")

    def stop_robot_server(self):
        self.robot_server_running = False
        with self.robot_conn_lock:
            if self.robot_client_socket:
                try:
                    self.robot_client_socket.close()
                except:
                    pass
                self.robot_client_socket = None
        if self.robot_server_socket:
            try:
                self.robot_server_socket.close()
            except:
                pass
            self.robot_server_socket = None
        if self.robot_server_thread and self.robot_server_thread.is_alive():
            self.robot_server_thread.join(timeout=2)

    def send_robot_column(self, col: int):
        if self.simulate:
            print(f"Simulating robot move to column {col}")
            time.sleep(1.0) # simulate movement delay
            return

        col_to_send = int(col) + 1
        with self.robot_conn_lock:
            conn = self.robot_client_socket
            
        if not conn:
            with self.robot_conn_lock:
                self.pending_column = col_to_send
            print("No robot connected; queued move.")
            return
            
        msg = str(col_to_send).encode("ascii")
        try:
            conn.send(msg)
            print(f"Sent column to robot: {col_to_send}")
        except (BlockingIOError, TimeoutError):
            with self.robot_conn_lock:
                self.pending_column = col_to_send
            print("Socket not ready; queued.")
        except Exception as e:
            print(f"Disconnected while sending: {e}")
            try:
                conn.close()
            except:
                pass
            with self.robot_conn_lock:
                self.pending_column = col_to_send
                self.robot_client_socket = None

    def __del__(self):
        if not self.simulate:
            self.stop_robot_server()
