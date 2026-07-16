import threading
import socket
import select
import os
import time
from typing import Callable, Optional, List

try:
    from connect_four_ai import AIPlayer, Difficulty, Position
except ImportError as e:
    print(f"Warning: could not import connect_four_ai ({e})")
    AIPlayer = None
    Difficulty = None
    Position = None

DIFFICULTY_NAMES = ("easy", "medium", "hard", "impossible")

# What the robot's button threads send. No terminator: the robot's keepalives
# have none either, so a token has to stand on its own in the stream.
TOGGLE_TOKEN = b"TOGGLE"
RESET_TOKEN = b"RESET"
ROBOT_TOKENS = (TOGGLE_TOKEN, RESET_TOKEN)
MAX_TOKEN_LEN = max(len(t) for t in ROBOT_TOKENS)

class RobotController:
    def __init__(self, simulate: bool = True, difficulty: str = "medium"):
        self.simulate = simulate

        # AI Player
        self.ai_player = None
        self.difficulty = None
        self.difficulty_name = "medium"
        self.set_difficulty(difficulty)

        # Robot Server TCP
        self.robot_server_host = os.environ.get("C4_SERVER_HOST", "0.0.0.0")
        self.robot_server_port = int(os.environ.get("C4_SERVER_PORT", "30020"))
        self.robot_server_socket = None
        self.robot_client_socket = None
        self.robot_server_thread = None
        self.robot_server_running = False
        self.robot_conn_lock = threading.Lock()
        self.pending_column = None

        # Difficulty button on the robot. The guard decides whether a TOGGLE is
        # accepted right now; while it is unset every toggle is rejected, so that
        # a button press arriving before main.py has wired up the game state
        # cannot swap the AI mid-move.
        self.difficulty_toggle_guard: Optional[Callable[[], bool]] = None
        self.on_difficulty_changed: Optional[Callable[[str], None]] = None

        # Reset button on the robot. Unguarded on purpose, to match the reset
        # button in the web UI: a reset must work whatever state we are in.
        self.on_reset_requested: Optional[Callable[[], None]] = None

        if not self.simulate:
            self.start_robot_server()

    def set_difficulty(self, diff_str: str):
        name = str(diff_str).lower()
        if name not in DIFFICULTY_NAMES:
            name = "medium"
        self.difficulty_name = name
        if not AIPlayer:
            return
        diff_map = {
            "easy": Difficulty.EASY,
            "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD,
            "impossible": Difficulty.IMPOSSIBLE
        }
        self.difficulty = diff_map[name]
        self.ai_player = AIPlayer(self.difficulty)

    def toggle_difficulty(self) -> Optional[str]:
        """Advance to the next difficulty, if the guard currently allows it."""
        guard = self.difficulty_toggle_guard
        if guard is None:
            print("Ignoring difficulty toggle: no guard installed")
            return None
        try:
            allowed = guard()
        except Exception as e:
            print(f"Difficulty toggle guard raised ({e}); ignoring toggle")
            return None
        if not allowed:
            print("Ignoring difficulty toggle: not allowed while a game is running")
            return None

        idx = DIFFICULTY_NAMES.index(self.difficulty_name)
        new_name = DIFFICULTY_NAMES[(idx + 1) % len(DIFFICULTY_NAMES)]
        self.set_difficulty(new_name)
        print(f"Difficulty toggled from robot button: {new_name}")

        if self.on_difficulty_changed:
            try:
                self.on_difficulty_changed(new_name)
            except Exception as e:
                print(f"on_difficulty_changed callback failed: {e}")
        return new_name

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
        if self.robot_server_running:
            return
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
                    threading.Thread(
                        target=self.robot_reader_loop, args=(conn,), daemon=True
                    ).start()
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

    def robot_reader_loop(self, conn):
        """Read what the robot sends us on its connection.

        This direction is not just button events: the robot also streams
        unterminated "2" keepalives to check we are still alive. So there is
        nothing to frame on -- we scan the stream for TOGGLE and drop the rest.
        """
        buf = b""
        while self.robot_server_running:
            with self.robot_conn_lock:
                superseded = self.robot_client_socket is not conn
            if superseded:
                # accept_loop already closed this socket when it replaced it.
                return
            try:
                ready, _, _ = select.select([conn], [], [], 1.0)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                data = conn.recv(256)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as e:
                if self.robot_server_running:
                    print(f"Robot read failed: {e}")
                break
            if not data:
                break

            buf = self.consume_robot_messages(buf + data)

        self.drop_robot_connection(conn)

    def consume_robot_messages(self, buf: bytes) -> bytes:
        """Act on each token in the stream, in order; return the leftover tail."""
        while True:
            hit = None
            upper = buf.upper()
            for token in ROBOT_TOKENS:
                idx = upper.find(token)
                if idx != -1 and (hit is None or idx < hit[0]):
                    hit = (idx, token)
            if hit is None:
                break
            idx, token = hit
            # Anything before the token is keepalive noise.
            buf = buf[idx + len(token):]
            self.handle_robot_token(token)

        # Keep just enough tail to match a token split across two reads; the
        # keepalives would otherwise grow this buffer without bound.
        tail = MAX_TOKEN_LEN - 1
        return buf[-tail:] if len(buf) > tail else buf

    def handle_robot_token(self, token: bytes):
        if token == TOGGLE_TOKEN:
            self.toggle_difficulty()
        elif token == RESET_TOKEN:
            self.request_reset()

    def request_reset(self):
        if self.on_reset_requested is None:
            print("Ignoring reset from robot button: no handler installed")
            return
        try:
            self.on_reset_requested()
            print("Game reset from robot button")
        except Exception as e:
            print(f"Reset handler failed: {e}")

    def drop_robot_connection(self, conn):
        with self.robot_conn_lock:
            if self.robot_client_socket is conn:
                self.robot_client_socket = None
                print("Robot disconnected")
        try:
            conn.close()
        except OSError:
            pass

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
