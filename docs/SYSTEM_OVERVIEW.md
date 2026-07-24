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
  - `GRABBED`: Handshake acknowledgment that the robot has picked up a token and is ready for a column command.

## Game Loop Logic

The turn-by-turn flow is orchestrated by the `detection_loop` thread in the backend:

1. **Human Plays**: The human physically drops a token.
2. **Vision Detection**: The camera sees the new token and updates the internal board state after a debounce period (e.g. 0.5s).
3. **Turn Validation**: The backend checks for illegal moves (e.g. multiple tokens dropped, gravity violations). If valid, it passes the turn to the Robot.
4. **AI Computation**: The Rust-based AI engine calculates the optimal move for the Robot.
5. **Robot Execution**: The backend sends the target column to the robot arm via TCP. The robot physically places the stone.
6. **Confirmation**: Once the camera detects the robot's stone in the board, the turn passes back to the human.

> [!CAUTION]
> If a gravity violation occurs (e.g. hand blocking the camera or a token gets stuck mid-air), the system pauses processing until the board stabilizes. The physical board must exactly match the expected state.

## Simulation Mode

For testing without physical hardware, the system includes a `SIMULATE_ROBOT_MODE`. 
- **Enabling**: It can be toggled on via the Web UI Settings or via `settings.json`.
- **Effect**: It disables the TCP server entirely. When the AI decides on a move, it waits 1 second and then directly modifies the "virtual board", simulating a perfect physical drop.
