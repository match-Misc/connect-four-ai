#!/usr/bin/env python3
"""
Connect Four Game Wrapper

This script provides a complete game interface that:
1. Starts the detection system in a subprocess
2. Shows a difficulty selection menu
3. Visualizes the current board state
4. Integrates with the connect-four-ai package for AI moves
5. Waits for human moves via computer vision detection
"""

import json
import os
import socket
import subprocess
import sys
import time
import threading
from typing import Optional, Tuple
from pathlib import Path

import dearpygui.dearpygui as dpg
import pygame
try:
    from connect_four_ai import AIPlayer, Difficulty, Position
except Exception as e:
    # Provide a clearer, actionable error message when the Rust-backed
    # `connect_four_ai` Python package isn't installed or importable.
    print(
        "Could not import the 'connect_four_ai' Python package.\n"
        "This project provides a Rust-backed Python extension in `crates/python`.\n"
        "To make it available to Python, either install the package from PyPI or build & install it locally.\n\n"
        "Recommended local development steps (Linux / bash):\n"
        "  cd crates/python\n"
        "  python -m pip install --upgrade pip setuptools wheel maturin\n"
        "  maturin develop --release\n\n"
        "Or try installing from PyPI (if a prebuilt wheel exists for your Python):\n"
        "  python -m pip install connect_four_ai\n\n"
        "Original import error:" , e,
    )
    # Exit now since the rest of the module requires these symbols.
    raise
from pygame import Surface


class Board:
    """
    Stores the state of the current board as a 2D array which
    can be drawn to a Pygame surface.

    Empty spaces are represented as 0, with 1 and 2 representing
    the first and second player respectively.
    """

    ROWS: int = 6
    COLS: int = 7
    COLOURS: dict = {
        "background": (23, 32, 46),
        "board": (55, 65, 81),
        "empty": (31, 41, 55),
        "empty-outline": (28, 38, 53),
        "player1": (220, 38, 38),
        "player1-outline": (153, 27, 27),
        "player2": (251, 192, 36),
        "player2-outline": (245, 158, 11),
        "preview": (128, 128, 128),
        "preview-outline": (100, 100, 100),
        "hint-text": (156, 163, 175),  # gray-400
        "winning": (255, 255, 0),  # bright yellow for winning pieces
        "winning-outline": (200, 200, 0),
    }

    def __init__(self, surface_size=800) -> None:
        """Creates a new, empty board."""
        # Calculates values used for drawing the board
        self.surface_size = surface_size
        self.padding = surface_size // 40
        self.cell_size = (surface_size - 2 * self.padding) // 10
        self.board_width = Board.COLS * self.cell_size + self.padding * 2
        self.board_height = Board.ROWS * self.cell_size + self.padding * 2
        self.board_x = (surface_size - self.board_width) // 2
        self.board_y = (surface_size - self.board_height) // 2
        self.piece_radius = int(self.cell_size * 0.45)
        self.piece_interior = int(self.piece_radius * 0.85)

        # AI move preview
        self.ai_preview_column = -1

        # Hint scores display
        self.show_hints = False
        self.hint_scores = [None] * Board.COLS

        # Winning positions for highlighting
        self.winning_positions = set()  # Set of (row, col) tuples

        # Load colors from calibration.json
        self.COLOURS = Board.COLOURS.copy()
        try:
            with open("calibration.json", "r") as f:
                calib = json.load(f)
            # Convert BGR to RGB
            player1_rgb = tuple(calib["player1_color"][::-1])
            player2_rgb = tuple(calib["player2_color"][::-1])
            self.COLOURS["player1"] = player1_rgb
            self.COLOURS["player2"] = player2_rgb
            # Darken for outlines
            self.COLOURS["player1-outline"] = tuple(max(0, c - 50) for c in player1_rgb)
            self.COLOURS["player2-outline"] = tuple(max(0, c - 50) for c in player2_rgb)
        except Exception as e:
            print(f"Could not load calibration colors, using defaults: {e}")

        # Creates the surface used for drawing the board
        self.surface = Surface((surface_size, surface_size))
        self.reset()

    def is_playable(self, col: int) -> bool:
        """Returns whether the given column is playable."""
        if col is None or col < 0 or col >= Board.COLS:
            return False
        return self.heights[col] < Board.ROWS

    def play(self, col: int, player: int):
        """
        Places a player's tile in the specified column.

        Assumes a 0-indexed column and player are given.
        """
        # Plays the piece
        if not self.is_playable(col):
            raise Exception("attempted to play a move in a full column.")
        row = Board.ROWS - 1 - self.heights[col]
        self.grid[row][col] = player + 1
        self.heights[col] += 1
        self.update()

    def reset(self):
        """Resets the board state."""
        self.grid = [[0 for _ in range(Board.COLS)] for _ in range(Board.ROWS)]
        self.heights = [0 for _ in range(Board.COLS)]
        self.winning_positions.clear()
        self.update()

    def update(self):
        """Updates the board's surface to reflect its current state."""
        self.surface.fill(self.COLOURS["background"])
        pygame.draw.rect(
            self.surface,
            self.COLOURS["board"],
            (self.board_x, self.board_y, self.board_width, self.board_height),
            border_radius=self.cell_size // 2,
        )
        for i, row in enumerate(self.grid):
            for j, n in enumerate(row):
                x = self.padding + self.board_x + self.cell_size * (j + 0.5)
                y = self.padding + self.board_y + self.cell_size * (i + 0.5)
                is_winning = (i, j) in self.winning_positions
                if is_winning:
                    piece_type = "winning"
                    outline_color = self.COLOURS["winning-outline"]
                    fill_color = self.COLOURS["winning"]
                else:
                    piece_type = ["empty", "player1", "player2"][n]
                    outline_color = self.COLOURS[f"{piece_type}-outline"]
                    fill_color = self.COLOURS[f"{piece_type}"]
                pygame.draw.circle(
                    self.surface,
                    outline_color,
                    (x, y),
                    self.piece_radius,
                )
                pygame.draw.circle(
                    self.surface,
                    fill_color,
                    (x, y),
                    self.piece_interior,
                )

        # Draw AI move preview if active
        if self.ai_preview_column >= 0 and self.ai_preview_column < Board.COLS:
            # Find the top empty row in the preview column
            preview_row = Board.ROWS - 1 - self.heights[self.ai_preview_column]
            if preview_row >= 0:
                x = (
                    self.padding
                    + self.board_x
                    + self.cell_size * (self.ai_preview_column + 0.5)
                )
                y = self.padding + self.board_y + self.cell_size * (preview_row + 0.5)
                # Use gray preview colors
                pygame.draw.circle(
                    self.surface,
                    self.COLOURS["preview-outline"],
                    (x, y),
                    self.piece_radius,
                )
                pygame.draw.circle(
                    self.surface,
                    self.COLOURS["preview"],
                    (x, y),
                    self.piece_interior,
                )

        # Draw hint scores if enabled
        if self.show_hints:
            font = pygame.font.SysFont(None, int(self.cell_size * 0.5))
            for col in range(Board.COLS):
                if self.hint_scores[col] is not None:
                    x = self.padding + self.board_x + self.cell_size * (col + 0.5)
                    y = self.padding + self.board_y - self.cell_size * 0.6
                    text = font.render(
                        str(self.hint_scores[col]), True, self.COLOURS["hint-text"]
                    )
                    text_rect = text.get_rect(center=(x, y))
                    self.surface.blit(text, text_rect)

    def draw(self, target: Surface, pos: tuple = (0, 0)):
        """Draws the board onto the given target surface."""
        target.blit(self.surface, pos)


class GameWrapper:
    """Main game wrapper class that orchestrates the entire game flow."""

    def __init__(self):
        self.difficulty: Optional[Difficulty] = None
        self.ai_player: Optional[AIPlayer] = None
        self.position: Position = Position()
        self.board: Board = Board()
        self.detection_process: Optional[subprocess.Popen] = None
        self.socket_client: Optional[socket.socket] = None
        self.last_bitmasks = (0, 0)
        self.game_running = False
        self.ai_move_displayed = False
        self.ai_move_column = -1
        self.human_turn = True
        self.game_won = False
        self.win_start_time = None

        # Robot/server integration (PC acts as server, robot as client)
        # This PC will listen on a TCP port and send the AI's chosen column as a string (e.g., "3").
        self.robot_server_enabled: bool = True
        self.robot_server_host: str = os.environ.get("C4_SERVER_HOST", "0.0.0.0")
        self.robot_server_port: int = int(os.environ.get("C4_SERVER_PORT", "30020"))
        self.robot_server_socket: Optional[socket.socket] = None
        self.robot_client_socket: Optional[socket.socket] = None
        self.robot_server_thread: Optional[threading.Thread] = None
        self.robot_server_running: bool = False
        self.robot_conn_lock = threading.Lock()
        self.pending_column = None

        # GUI elements
        self.window_id = None
        self.difficulty_window_id = None
        self.game_window_id = None
        self.status_text_id = None
        self.ai_move_text_id = None
        self.hint_button_id = None

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Connect Four - Human vs AI")
        self.clock = pygame.time.Clock()

    def start_robot_server(self) -> bool:
        """Start a simple TCP server that accepts a single client (robot).

        When connected, send_robot_column will write the column as a string.
        """
        if not self.robot_server_enabled:
            return True
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
                            except Exception:
                                pass
                        # Make client non-blocking to avoid blocking game loop on send
                        try:
                            conn.setblocking(False)
                        except Exception:
                            pass
                        self.robot_client_socket = conn
                    print(f"Robot client connected from {addr}")
                    # If there is a pending column (chosen before connection), send it now
                    with self.robot_conn_lock:
                        pending = self.pending_column
                        send_conn = self.robot_client_socket
                        self.pending_column = None
                    if pending is not None and send_conn:
                        try:
                            msg = (str(int(pending))).encode("ascii")
                            # Non-blocking send; may raise BlockingIOError if not ready
                            send_conn.send(msg)
                            print(f"Sent pending column to robot: {pending}")
                        except Exception as e:
                            print(f"Failed to send pending column to robot: {e}")
                            # Re-queue if send didn't go through
                            with self.robot_conn_lock:
                                self.pending_column = pending

            self.robot_server_thread = threading.Thread(target=accept_loop, daemon=True)
            self.robot_server_thread.start()
            print(f"Robot server listening on {self.robot_server_host}:{self.robot_server_port}")
            return True
        except Exception as e:
            print(f"Failed to start robot server: {e}")
            return False

    def stop_robot_server(self):
        """Stop the robot TCP server and any client connection."""
        self.robot_server_running = False
        with self.robot_conn_lock:
            if self.robot_client_socket:
                try:
                    self.robot_client_socket.close()
                except Exception:
                    print("Failed to close robot client socket")
                    pass
                self.robot_client_socket = None
        if self.robot_server_socket:
            try:
                self.robot_server_socket.close()
            except Exception:
                print("Failed to close robot server socket")
                pass
            self.robot_server_socket = None
        if self.robot_server_thread and self.robot_server_thread.is_alive():
            self.robot_server_thread.join(timeout=2)

    def send_robot_column(self, col: int):
        """Send the chosen column to the connected robot client as a string (e.g., "3")."""
        if not self.robot_server_enabled:
            return
        # Convert to 1-indexed for robot
        col_to_send = int(col) + 1
        # Grab current client under lock, but perform send without holding the lock
        with self.robot_conn_lock:
            conn = self.robot_client_socket
        if not conn:
            # No client yet; queue the column to send on next connect
            with self.robot_conn_lock:
                self.pending_column = col_to_send
            print("No robot client connected; column queued.")
            return
        msg = (str(col_to_send)).encode("ascii")
        try:
            # Non-blocking send of small payload
            conn.send(msg)
            print(f"Sent column to robot: {col_to_send}")
        except (BlockingIOError, TimeoutError):
            # Try again on next connect or future opportunity
            with self.robot_conn_lock:
                self.pending_column = col_to_send
            print("Robot socket not ready; column queued for resend.")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"Robot client disconnected while sending column: {e}")
            try:
                conn.close()
            except Exception:
                pass
            # Queue to send on next reconnect
            with self.robot_conn_lock:
                self.pending_column = col_to_send
                self.robot_client_socket = None


    def bitmasks_to_grid(self, player1_mask: int, player2_mask: int) -> list[list[int]]:
        """Convert bitmasks to 2D grid representation."""
        grid = [[0 for _ in range(7)] for _ in range(6)]

        for row in range(6):
            for col in range(7):
                # Bit position: (5 - row) * 7 + col
                bit_pos = (5 - row) * 7 + col
                if player1_mask & (1 << bit_pos):
                    grid[row][col] = 1  # Player 1
                elif player2_mask & (1 << bit_pos):
                    grid[row][col] = 2  # Player 2 (AI)

        return grid

    def update_board_from_bitmasks(self, player1_mask: int, player2_mask: int):
        """Update the board visualization from bitmasks."""
        grid = self.bitmasks_to_grid(player1_mask, player2_mask)

        # Update board state
        self.board.grid = grid
        self.board.heights = [0] * 7
        for col in range(7):
            for row in range(6):
                if grid[row][col] != 0:
                    self.board.heights[col] += 1

        # Check if AI move has been detected (AI just played)
        if not self.human_turn and self.ai_move_displayed and self.ai_move_column >= 0:
            # Check if there's a new piece in the AI's intended column
            if self.board.heights[self.ai_move_column] > 0:
                # Check if the top piece in that column is player2 (AI)
                top_row = 6 - self.board.heights[self.ai_move_column]
                if grid[top_row][self.ai_move_column] == 2:
                    # AI move detected, reset preview
                    self.board.ai_preview_column = -1

        self.board.update()

    def start_detection(self) -> bool:
        """Start the detection subprocess."""
        try:
            # Always run detection.py from this file's directory to find calibration.json
            det_cwd = str(Path(__file__).parent)
            self.detection_process = subprocess.Popen(
                [sys.executable, "detection.py"],
                cwd=det_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Started detection subprocess")

            # Wait a bit for the socket server to start
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Failed to start detection: {e}")
            return False

    def stop_detection(self):
        """Stop the detection subprocess."""
        if self.detection_process:
            self.detection_process.terminate()
            try:
                self.detection_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.detection_process.kill()
            print("Stopped detection subprocess")

    def connect_to_detection(self) -> bool:
        """Connect to the detection socket server."""
        # Retry for a few seconds to allow the detector to initialize webcam & socket
        deadline = time.time() + 10.0
        last_err = None
        while time.time() < deadline:
            # If the detection process died, surface its stderr/stdout for debugging
            if self.detection_process and self.detection_process.poll() is not None:
                try:
                    out, err = self.detection_process.communicate(timeout=1)
                except Exception:
                    out, err = b"", b""
                if out:
                    print("[detection stdout]\n" + out.decode(errors="ignore"))
                if err:
                    print("[detection stderr]\n" + err.decode(errors="ignore"))
                print("Detection process terminated unexpectedly.")
                return False

            try:
                self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket_client.settimeout(1.0)
                self.socket_client.connect(("localhost", 65432))
                self.socket_client.settimeout(None)
                print("Connected to detection socket server")
                return True
            except Exception as e:
                last_err = e
                time.sleep(0.5)

        print(f"Failed to connect to detection server: {last_err}")
        print(
            "Tip: Ensure detection dependencies are installed in the active venv:\n"
            "  pip install -e /home/match-mover/Documents/connect-four-ai/robot-game\n"
            "And try running detection directly for diagnostics:\n"
            "  python /home/match-mover/Documents/connect-four-ai/robot-game/detection.py\n"
        )
        return False

    def disconnect_from_detection(self):
        """Disconnect from the detection socket server."""
        if self.socket_client:
            self.socket_client.close()
            self.socket_client = None

    def get_current_bitmasks(self) -> Tuple[int, int]:
        """Get current bitmasks from detection server."""
        if not self.socket_client:
            return self.last_bitmasks

        try:
            # Send request
            self.socket_client.sendall(b"request")

            # Receive response
            data = self.socket_client.recv(1024)
            if data:
                bitmasks = json.loads(data.decode("utf-8"))
                self.last_bitmasks = (bitmasks["player1"], bitmasks["player2"])
                return self.last_bitmasks
        except Exception as e:
            print(f"Error getting bitmasks: {e}")

        return self.last_bitmasks

    def check_board_changed(
        self, old_p1: int, old_p2: int, new_p1: int, new_p2: int
    ) -> bool:
        """Check if the board state has changed."""
        return old_p1 != new_p1 or old_p2 != new_p2

    def get_ai_move(self) -> Optional[int]:
        """Get the AI's next move."""
        if not self.ai_player:
            return None

        try:
            move = self.ai_player.get_move(self.position)
            return move
        except Exception as e:
            print(f"Error getting AI move: {e}")
            return None

    def update_position_from_bitmasks(
        self, player1_mask: int, player2_mask: int, human_turn: bool
    ):
        """Update the Position object from bitmasks."""
        # Convert bitmasks to board string
        board_chars = []
        for row in range(6):
            for col in range(7):
                bit_pos = (5 - row) * 7 + col
                if player1_mask & (1 << bit_pos):
                    if human_turn:
                        board_chars.append("x")  # Current player (human)
                    else:
                        board_chars.append("o")  # Opponent (human)
                elif player2_mask & (1 << bit_pos):
                    if human_turn:
                        board_chars.append("o")  # Opponent (AI)
                    else:
                        board_chars.append("x")  # Current player (AI)
                else:
                    board_chars.append(".")

        board_string = "".join(board_chars)
        self.position = Position.from_board_string(board_string)

    def find_winning_positions(
        self, player1_mask: int, player2_mask: int
    ) -> set[tuple[int, int]]:
        """Find all positions that are part of a winning four-in-a-row combination."""
        winning_positions = set()

        # Convert bitmasks to grid for easier checking
        grid = self.bitmasks_to_grid(player1_mask, player2_mask)

        # Check all possible four-in-a-row combinations
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]  # horizontal, vertical, diagonal1, diagonal2

        for row in range(6):
            for col in range(7):
                if grid[row][col] == 0:
                    continue

                player = grid[row][col]

                for dr, dc in directions:
                    # Check if we can form a line of 4 in this direction
                    positions = []
                    for i in range(4):
                        r, c = row + i * dr, col + i * dc
                        if 0 <= r < 6 and 0 <= c < 7 and grid[r][c] == player:
                            positions.append((r, c))
                        else:
                            break

                    if len(positions) == 4:
                        winning_positions.update(positions)

        return winning_positions

    def show_difficulty_menu(self):
        """Show the difficulty selection menu using Dear PyGui."""
        dpg.create_context()
        dpg.create_viewport(
            title="Connect Four - Select Difficulty", width=400, height=300
        )

        with dpg.window(
            label="Select Difficulty", width=400, height=300
        ) as self.difficulty_window_id:
            dpg.add_text("Choose AI Difficulty:", pos=(20, 20))

            with dpg.group(pos=(20, 50)):
                dpg.add_button(
                    label="Easy",
                    width=360,
                    height=40,
                    callback=self.set_difficulty_easy,
                )
                dpg.add_button(
                    label="Medium",
                    width=360,
                    height=40,
                    callback=self.set_difficulty_medium,
                )
                dpg.add_button(
                    label="Hard",
                    width=360,
                    height=40,
                    callback=self.set_difficulty_hard,
                )
                dpg.add_button(
                    label="Impossible",
                    width=360,
                    height=40,
                    callback=self.set_difficulty_impossible,
                )

        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Wait for difficulty selection
        while dpg.is_dearpygui_running() and self.difficulty is None:
            dpg.render_dearpygui_frame()
            time.sleep(0.01)

        dpg.destroy_context()

    def set_difficulty_easy(self):
        self.difficulty = Difficulty.EASY
        self.ai_player = AIPlayer(self.difficulty)

    def set_difficulty_medium(self):
        self.difficulty = Difficulty.MEDIUM
        self.ai_player = AIPlayer(self.difficulty)

    def set_difficulty_hard(self):
        self.difficulty = Difficulty.HARD
        self.ai_player = AIPlayer(self.difficulty)

    def set_difficulty_impossible(self):
        self.difficulty = Difficulty.IMPOSSIBLE
        self.ai_player = AIPlayer(self.difficulty)

    def show_game_gui(self):
        """Show the game GUI with status and AI move display."""
        dpg.create_context()
        dpg.create_viewport(title="Connect Four - Game Status", width=600, height=400)

        with dpg.window(
            label="Game Status", width=600, height=400
        ) as self.game_window_id:
            self.status_text_id = dpg.add_text("Initializing game...", pos=(10, 30))
            self.ai_move_text_id = dpg.add_text("AI Move: Waiting...", pos=(30, 70))
            self.hint_button_id = dpg.add_button(
                label="Show Hints",
                pos=(10, 100),
                width=100,
                height=30,
                callback=self.toggle_hints,
            )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update_game_status(self, status: str):
        """Update the game status text."""
        if self.status_text_id:
            dpg.set_value(self.status_text_id, status)

    def update_ai_move_display(self, move: Optional[int]):
        """Update the AI move display."""
        if self.ai_move_text_id:
            if move is not None:
                dpg.set_value(
                    self.ai_move_text_id, f"AI will play in column: {move + 1}"
                )
            else:
                dpg.set_value(self.ai_move_text_id, "AI Move: Calculating...")

    def toggle_hints(self):
        """Toggle hint display on/off."""
        self.board.show_hints = not self.board.show_hints
        if self.hint_button_id:
            dpg.set_item_label(
                self.hint_button_id,
                "Hide Hints" if self.board.show_hints else "Show Hints",
            )
        if self.board.show_hints:
            self.update_hint_scores()
        else:
            self.board.hint_scores = [None] * Board.COLS

    def update_hint_scores(self):
        """Update the hint scores for display."""
        if not self.ai_player or not self.board.show_hints:
            self.board.hint_scores = [None] * Board.COLS
            return

        try:
            # Get scores from AI's perspective, but we want to show them for the human player
            # So we need to negate the scores since AI maximizes and human minimizes
            scores = self.ai_player.get_all_move_scores(self.position)
            # For human player (who minimizes), higher scores are better moves
            # So we show the negated AI scores to make higher numbers better for human
            self.board.hint_scores = [
                -score if score is not None else None for score in scores
            ]
        except Exception as e:
            print(f"Error getting hint scores: {e}")
            self.board.hint_scores = [None] * Board.COLS

    def run_game(self):
        """Main game loop."""
        # Show difficulty selection
        self.show_difficulty_menu()

        if not self.difficulty:
            print("No difficulty selected, exiting")
            return

        # Start detection
        if not self.start_detection():
            print("Failed to start detection, exiting")
            return

        # Connect to detection server
        if not self.connect_to_detection():
            print("Failed to connect to detection server, exiting")
            self.stop_detection()
            return

        # Show game GUI
        self.show_game_gui()

        # Start robot server (PC acts as server; robot connects as client)
        self.start_robot_server()

        # Initialize game state
        self.game_running = True
        last_p1, last_p2 = 0, 0
        human_turn = True  # Human goes first

        self.update_game_status("Game started! Human's turn (you go first)")

        try:
            while self.game_running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_running = False

                # Get current board state
                current_p1, current_p2 = self.get_current_bitmasks()

                # Update board visualization
                self.update_board_from_bitmasks(current_p1, current_p2)

                # Update position for AI calculations
                self.update_position_from_bitmasks(current_p1, current_p2, human_turn)

                # Check for game end
                won = self.position.is_won_position()
                if won and not self.game_won:
                    # Game just ended, highlight winning positions
                    self.game_won = True
                    self.win_start_time = time.time()
                    winning_positions = self.find_winning_positions(
                        current_p1, current_p2
                    )
                    self.board.winning_positions = winning_positions
                    self.board.update()

                    if human_turn:
                        self.update_game_status("Game Over - You win!")
                    else:
                        self.update_game_status("Game Over - AI wins!")

                # Check if win highlight period is over (10 seconds)
                if self.game_won and time.time() - self.win_start_time >= 10:
                    self.game_running = False
                    break

                # Check if board is empty (game start)
                is_empty = current_p1 == 0 and current_p2 == 0
                if is_empty:
                    human_turn = True
                    self.ai_move_displayed = False
                    self.update_game_status("Board is empty. Your turn!")
                    self.update_ai_move_display(None)
                    # Update hint scores for empty board
                    if self.board.show_hints:
                        self.update_hint_scores()

                # Check for human move (only if game not won)
                elif not self.game_won:
                    changed = self.check_board_changed(
                        last_p1, last_p2, current_p1, current_p2
                    )
                    if changed:
                        if human_turn:
                            # Human made a move, now it's AI's turn
                            human_turn = False
                            self.ai_move_displayed = False
                            self.update_game_status("AI is thinking...")
                            self.update_ai_move_display(None)
                            # Reset preview when human moves
                            self.board.ai_preview_column = -1
                        else:
                            # AI move was made, now human's turn
                            human_turn = True
                            self.ai_move_displayed = False
                            self.update_game_status("Your turn!")
                            # Reset preview when AI moves
                            self.board.ai_preview_column = -1
                            # Update hint scores after AI move
                            if self.board.show_hints:
                                self.update_hint_scores()

                # AI's turn (only if game not won)
                if not human_turn and not self.ai_move_displayed and not self.game_won:
                    ai_move = self.get_ai_move()
                    if ai_move is not None:
                        self.ai_move_column = ai_move
                        self.board.ai_preview_column = ai_move
                        self.ai_move_displayed = True
                        self.update_game_status(
                            f"AI will play in column {ai_move + 1}. Make your move!"
                        )
                        self.update_ai_move_display(ai_move)
                        # Send move to robot controller once per AI decision
                        self.send_robot_column(ai_move)

                # Update last state
                last_p1, last_p2 = current_p1, current_p2

                # Draw board
                self.screen.fill((0, 0, 0))
                self.board.draw(self.screen)
                pygame.display.flip()

                # Render GUI
                if dpg.is_dearpygui_running():
                    dpg.render_dearpygui_frame()

                self.clock.tick(30)

        except KeyboardInterrupt:
            print("Game interrupted")
        except Exception as e:
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.disconnect_from_detection()
        self.stop_detection()
        self.stop_robot_server()

        if dpg.is_dearpygui_running():
            dpg.destroy_context()

        pygame.quit()


def main():
    """Main entry point."""
    game = GameWrapper()
    game.run_game()


if __name__ == "__main__":
    main()
