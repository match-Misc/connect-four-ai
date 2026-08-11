# Comprehensive System Explanation

This document explains how the physical showcase components fit together for the interactive Connect Four Showcase. 

## Hardware Architecture

The Connect Four Showcase relies on three main physical components interacting over the local network and TCP:
- **PC/Laptop**: The central brain running the backend (`Flask`), frontend (`Vite/React`), and computer vision loop.
- **Intel RealSense Camera**: Placed above/in-front of the board to capture the board state in real-time.
- **Robot Arm**: A serial robot arm that communicates with the PC over a raw TCP socket connection on port `30020`. 

## Camera & Vision Setup

The RealSense camera continuously captures frames of the 7x6 Connect Four board. 
The system uses pre-configured calibration settings saved in the config file (accessible via the `calibration_server`). 
- It relies on grid mapping and max RGB thresholding to determine where tokens are dropped.
- The `vision_service.py` is responsible for parsing these frames into a 2D integer array representing the board.

## Robot Control & Communication

The robot arm acts as a TCP client connecting to the PC on port `30020` (as defined by `C4_SERVER_PORT`).

Communication Protocol (No Terminators):
- **Outbound (PC to Robot)**: The backend sends single-character codes.
  - `1` through `7`: Target column for a drop.
  - `8` (GRAB_CODE): Tells the robot to pick up a token.
  - `9` (WIN_CODE): Robot won (celebrate).
  - `0` (LOSE_CODE): Human won or draw (no celebration).
- **Inbound (Robot to PC)**: The robot sends specific byte sequences.
  - `TOGGLE`: The physical difficulty button was pressed.
  - `RESET`: The physical reset button was pressed.
  - `GRABBED`: The robot is holding a token and is ready for a column command.

### The Grab Handshake

The robot is the only side that can see its own gripper, so the backend never
assumes what is in it — it waits to be told. A column is never sent to a robot
that has not confirmed it is holding a token, which is what keeps the grab code
and a column from sitting unread in the pendant's buffer at the same time (with
no terminators, the pendant would read them as one garbled message).

1. **Wait for the robot.** The robot is the side that dials in. Until it
   connects, the backend just waits and logs on a slow beat.
2. **The robot announces its gripper.** Immediately after connecting, the robot
   sends `GRABBED` *if it is already holding a token*, and sends nothing if it
   is not. This is what lets the game survive a backend restart or a pendant
   reconnect mid-game without sending an already-loaded arm off to fetch a
   second token.
3. **The backend asks for a token.** If no announcement arrives within
   `CONNECT_ANNOUNCE_GRACE` (2 s), the gripper is taken to be empty and `8` goes
   out.
4. **The robot acks.** It picks up a token and sends `GRABBED`.
5. **Only now is a column sent.** `1`–`7` goes out, and the robot places the
   token.
6. **Repeat.** When the camera sees the robot's token land on the board, the
   gripper counts as empty again and step 3 starts over. This continues until
   the game is won, lost, or reset.

> [!IMPORTANT]
> Step 2 is what the pendant program has to get right: an unprompted `GRABBED`
> on connect whenever a token is already clamped. An empty gripper is signalled
> by staying silent.

> [!NOTE]
> `8` is never re-sent while an ack is outstanding — a second grab code
> overtaking a slow `GRABBED` is exactly how the robot ends up fetching two
> tokens. If the connection is genuinely dead, the socket errors out, the
> pendant reconnects, and the handshake restarts from step 2. Likewise, a reset
> (from the web UI or the robot's button) does **not** clear what the backend
> believes about the gripper: nothing in a reset commands the arm to open, so a
> token it was holding is still there. If the pendant's own reset routine
> returns its token, it must reconnect so step 2 runs again.

## Game Loop Logic

The turn-by-turn flow is orchestrated by the `detection_loop` thread in the backend:

1. **Human Plays**: The human physically drops a token.
2. **Vision Detection**: The camera sees the new token and updates the internal board state after a debounce period (e.g. 0.5s).
3. **Turn Validation**: The backend checks for illegal moves (e.g. multiple tokens dropped, gravity violations). If valid, it passes the turn to the Robot.
4. **AI Computation**: The Rust-based AI engine calculates the optimal move for the Robot.
5. **Robot Execution**: The backend sends the target column to the robot arm via TCP — but only once the robot has confirmed it is holding a token (see [The Grab Handshake](#the-grab-handshake)). The robot physically places the stone.
6. **Confirmation**: Once the camera detects the robot's stone in the board, the turn passes back to the human, and the robot is asked to grab the next token.

> [!CAUTION]
> If a gravity violation occurs (e.g. hand blocking the camera or a token gets stuck mid-air), the system pauses processing until the board stabilizes. The physical board must exactly match the expected state.

## Simulation Mode

For testing without physical hardware, the system includes a `SIMULATE_ROBOT_MODE`. 
- **Enabling**: It can be toggled on via the Web UI Settings or via `settings.json`.
- **Effect**: It disables the TCP server entirely. When the AI decides on a move, it waits 1 second and then directly modifies the "virtual board", simulating a perfect physical drop.
