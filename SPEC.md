### **SPEC: Connect Four Robot Game Implementation**

-----

#### 1\. Overview

This document specifies the technical implementation for a complete Connect Four game where a human player competes against a Universal Robot (UR) arm. The project will be structured as a single, installable Python application, managed by `uv`, and launched via a single command.

The system will integrate existing computer vision and AI components to create a seamless gameplay loop: one-time calibration, GUI-based difficulty selection, vision-based detection of human moves, AI-powered robot moves communicated via TCP, and clear visual feedback through a Pygame interface.

-----

#### 2\. Project Structure

To achieve a clean, maintainable, and single-script-executable application, the repository will be restructured. The existing `computer-vision` and `python-demo` directories will be consolidated into a new, unified Python project structure.

```
/
├── .github/                      # (Existing CI/CD)
├── crates/                         # (Existing Rust crates)
├── web-demo/                       # (Existing web demo)
├── config.json                     # NEW: Robot server configuration.
├── main.py                         # NEW: Single entry point to run the game.
├── pyproject.toml                  # NEW: Project definition and dependencies for uv.
├── README.md                       # (To be updated with new instructions)
└── src/
    ├── __init__.py
    ├── game_logic.py             # NEW: Manages game state and AI interaction.
    ├── pygame_visualizer.py      # NEW: Adapted from python-demo for all GUI elements.
    ├── robot_server.py           # NEW: TCP server for UR robot communication.
    └── vision.py                 # NEW: Consolidated module from computer-vision.
```

-----

#### 3\. System Architecture

The application is orchestrated by `main.py`, which manages the game state and coordinates all modules.

```mermaid
graph TD
    subgraph Connect Four Application
        A[main.py: Game Loop] -- Reads --> Z[config.json];
        A -- Manages --> B[src/vision.py: Calibrator & Detector];
        A -- Manages --> E[src/pygame_visualizer.py: GUI];
        A -- Manages --> H[src/robot_server.py: TCP Server];
        A -- Manages --> D[src/game_logic.py: State Manager];

        D -- Uses --> F[connect-four-ai: AIPlayer];
    end

    subgraph External Hardware
        I[Human Player] -- Physically places tile --> J[Physical Board];
        K[Webcam] -- Captures video of Board --> B;
        L[Robot Arm] -- Executes move --> J;
        M[Robot Controller (URScript Client)] -- TCP connects to --> H;
    end
    
    H -- "Send Move ('3\\n')" --> M;
    M -- "Send Ack ('done\\n')" --> H;

    style I fill:#cde,stroke:#333,stroke-width:2px
    style L fill:#cde,stroke:#333,stroke-width:2px
    style J fill:#f9f,stroke:#333,stroke-width:2px
```

-----

#### 4\. Detailed Component Specification

##### 4.1. `pyproject.toml` (Root)

This file defines the project for `uv` and `pip`.

  * **Content**:
    ```toml
    [project]
    name = "connect-four-robot-game"
    version = "1.0.0"
    description = "A Connect Four game against a physical robot using computer vision and AI."
    requires-python = ">=3.11"
    dependencies = [
        "connect-four-ai>=1.0.0",
        "pygame>=2.6.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "dearpygui>=1.10.0"
    ]

    [project.scripts]
    play-connect-four = "main:main"
    ```
  * This setup allows the project to be installed via `uv pip install -e .` and run with the command `play-connect-four`.

##### 4.2. `config.json` (Root)

Stores network configuration for the robot server.

  * **Structure**:
    ```json
    {
      "robot_server": {
        "host": "0.0.0.0",
        "port": 30002
      }
    }
    ```

##### 4.3. `main.py` (Root)

The single entry point and central orchestrator.

  * **Responsibilities**:
    1.  Load configuration from `config.json`.
    2.  Check for `calibration.json`. If it doesn't exist, instantiate and run `vision.Calibrator`.
    3.  Initialize all modules: `PygameVisualizer`, `RobotServer`, `vision.Detector`, and `GameLogic`.
    4.  Manage the main game state machine (see Section 5).
    5.  Handle user inputs for restarting (`R`) or quitting (`Q`).
    6.  Ensure graceful shutdown of the TCP server, camera, and Pygame.

##### 4.4. `src/vision.py`

Consolidates and refactors the existing `computer-vision` scripts into a reusable module.

  * **Class `Calibrator`**: Adapted from `calibration.py`. To be run as a blocking process from `main.py` if needed.
  * **Class `Detector`**: Adapted from `detection.py`.
      * `__init__(calibration_file)`: Loads calibration and starts the webcam thread.
      * `get_stable_board_state() -> (int, int)`: Continuously captures frames. Returns the bitboards `(player1, player2)` only after the board state has been identical for a set duration (e.g., 1.5 seconds) to ensure a stable reading after a move.
      * `shutdown()`: Stops the webcam thread and releases the camera.
  * **Function `get_move_from_boards(mask_before: int, mask_after: int) -> int`**: Calculates the column of the new move by comparing two board masks.

##### 4.5. `src/robot_server.py`

Manages the TCP server for communication with the UR robot.

  * **Class `RobotServer`**:
      * `__init__`, `wait_for_connection`, `send_move_and_wait_for_ack`, `close` methods as defined in v1.2.
      * The `send_move_and_wait_for_ack` method will handle sending the **1-indexed** column and waiting for a `"done\n"` acknowledgment.

##### 4.6. `src/game_logic.py`

Encapsulates game state rules and AI interaction.

  * **Class `GameLogic`**:
      * `__init__(difficulty: Difficulty)`: Initializes `connect_four_ai.AIPlayer`.
      * `get_robot_move(p1_bitboard: int, p2_bitboard: int)`: Creates a `Position` object from the current bitboards and gets the AI's next move.
      * `find_winning_line(p1_bitboard: int)`: Iterates through the winner's pieces and checks in all 4 directions for a line of four to return the coordinates for highlighting.

##### 4.7. `src/pygame_visualizer.py`

Handles all graphical output.

  * **Class `PygameVisualizer`**:
      * `__init__`: Initializes Pygame, fonts, and colors.
      * `show_difficulty_selection() -> Difficulty`: Renders a menu and waits for a mouse click on a difficulty button. Returns the selected `Difficulty`.
      * `draw_board_from_state(p1_bitboard: int, p2_bitboard: int)`: Renders the game board from bitboards.
      * `draw_game_status(message: str)`: Displays status messages (e.g., "Your Turn", "Waiting for Robot...").
      * `draw_winning_line(winning_tiles: list[tuple])`: Highlights the four winning tiles.

-----

#### 5\. Game Flow State Machine

The `main.py` script will implement the following state machine:

1.  **INITIALIZING**: Load `config.json`. Check for `calibration.json`. If missing, transition to `CALIBRATING`. Otherwise, transition to `WAITING_FOR_ROBOT`.
2.  **CALIBRATING**: Run the `vision.Calibrator` GUI. Block until calibration is saved. Transition to `WAITING_FOR_ROBOT`.
3.  **WAITING\_FOR\_ROBOT**: Start the `RobotServer`. Display "Waiting for robot connection..." in the GUI. Block until the robot connects. Transition to `DIFFICULTY_SELECTION`.
4.  **DIFFICULTY\_SELECTION**: Call `visualizer.show_difficulty_selection()`. Wait for user input. Once selected, initialize `GameLogic` with the chosen difficulty. Transition to `GAME_START`.
5.  **GAME\_START**: Reset all game variables. Set player turn to Human. Transition to `HUMAN_TURN`.
6.  **HUMAN\_TURN**:
      * Store `board_before = detector.get_stable_board_state()`.
      * Display "Your Turn".
      * Loop until `board_after = detector.get_stable_board_state()` is different from `board_before`.
      * Identify the move and update the game state. Check for win/draw.
      * If game over, transition to `GAME_OVER`. Otherwise, transition to `ROBOT_TURN`.
7.  **ROBOT\_TURN**:
      * Display "Robot is thinking...".
      * Get move from `game_logic.get_robot_move()`.
      * Display "Robot is moving...".
      * Send move to robot and wait for acknowledgment via `robot_server`.
      * Validate the physical move using the same stable state detection as the human turn.
      * Update game state. Check for win/draw.
      * If game over, transition to `GAME_OVER`. Otherwise, transition to `HUMAN_TURN`.
8.  **GAME\_OVER**:
      * Display result ("You Win\!", "You Lose\!", "Draw").
      * Call `game_logic.find_winning_line()` and pass the result to `visualizer.draw_winning_line()`.
      * Display "Press 'R' to play again".
      * Wait for 'R' keypress. On press, transition to `DIFFICULTY_SELECTION`.

-----

#### 6\. Setup and Execution Instructions

1.  **Prerequisites**:

      * Python 3.11+
      * `uv` package manager (`pip install uv`)
      * A connected webcam.
      * A Universal Robot on the same network, configured with the client URScript.

2.  **Setup**:

      * Open a terminal in the project root directory.
      * Create a virtual environment: `uv venv`
      * Activate the environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).
      * Install the project and its dependencies: `uv pip install -e .`

3.  **Configuration**:

      * Edit `config.json`. Set `"host"` to the IP address of the computer running the Python script. The robot's URScript must point to this IP address.
      * Ensure the `"port"` matches the port used in the URScript.

4.  **Execution**:

      * Run the game using the installed script command: `play-connect-four`
      * Alternatively, run directly: `python main.py`

-----

#### 7\. Acceptance Criteria

1.  [ ] The project can be installed with `uv pip install -e .` and runs via a single command.
2.  [ ] If `calibration.json` is missing, the calibration GUI launches and must be completed.
3.  [ ] The application waits for a TCP connection from the robot before showing the difficulty menu.
4.  [ ] Human moves are detected **only by computer vision** after the board state stabilizes.
5.  [ ] Robot moves are calculated by the AI, sent via TCP, and acknowledged by the robot.
6.  [ ] The physical moves of both the human and the robot are validated by the vision system.
7.  [ ] The Pygame window accurately reflects the game state, including status messages and a highlighted winning line at the end.
8.  [ ] The game correctly identifies win/loss/draw outcomes.
9.  [ ] After a game ends, pressing 'R' successfully returns the user to the difficulty selection screen for a new game.
10. [ ] The application handles a lost robot connection or communication timeout gracefully.