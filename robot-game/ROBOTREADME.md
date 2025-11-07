# Connect Four Robot Game

This project implements a physical Connect Four game system that uses computer vision to detect board state and integrates with an AI opponent. The system consists of three main Python scripts: `calibration.py`, `detection.py`, and `game.py`, which work together to provide a complete human-vs-AI Connect Four experience.

## Overview

The robot game allows you to play Connect Four against an AI opponent using a physical board and webcam. The system uses computer vision to detect piece placements in real-time, eliminating the need for manual input. The AI is powered by the `connect_four_ai` package, providing various difficulty levels from easy to impossible.

### Key Features
- **Computer Vision Detection**: Real-time board state recognition using webcam
- **Color Calibration**: Automatic player color detection for robust piece identification
- **AI Integration**: Multiple difficulty levels with advanced Connect Four AI
- **Visual Feedback**: Live board visualization with move previews and hints
- **Socket Communication**: Efficient data exchange between detection and game components

## Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Physical Connect Four board
- Red and yellow game pieces (or any two distinct colors)
- Well-lit environment for optimal computer vision performance

### Required Python Packages
- `opencv-python` (OpenCV for computer vision)
- `dearpygui` (GUI framework)
- `numpy` (numerical computations)
- `pygame` (game visualization)
- `connect_four_ai` (AI engine - included in the project)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd robot-game
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or if using uv:
   uv sync
   ```

3. **Ensure the AI package is available**:
   The `connect_four_ai` package should be installed or available in your Python path. If using the parent project structure, it should be accessible.

## Usage

1. **Activate the Python environment**:
   Only needed when running on the nuc.
   ```bash
   cd ~/connect-four-ai
   source ~/.pyenv/versions/connect4-3.13/bin/activate
   ```

### Step 1: Board Calibration

Before playing, you must calibrate the system to recognize your specific board setup.

1. **Run the calibration script**:
   ```bash
   python calibration.py
   ```

2. **Define board corners**:
   - Click on the four corners of your Connect Four board in the webcam feed
   - Corners should be clicked in order: top-left, top-right, bottom-left, bottom-right

3. **Adjust hole parameters**:
   - Use the sliders to set hole diameter and spacing to match your board
   - Adjust contrast, saturation, and brightness for optimal image quality

4. **Calibrate player colors**:
   - Place one red piece and one yellow piece in the first two columns
   - Click "Calibrate Colors" to sample the piece colors
   - The system will automatically detect and store color profiles

5. **Save calibration**:
   - Click "Save Calibration" to store all settings in `calibration.json`

### Step 2: Test Detection

Verify that the detection system works correctly with your calibrated setup.

1. **Run the detection script**:
   ```bash
   python detection.py
   ```

2. **Check real-time detection**:
   - The GUI will show the webcam feed with detected pieces highlighted
   - Place pieces on the board and verify they are correctly identified
   - The bitboard display shows the current board state as numerical masks

3. **Adjust detection threshold if needed**:
   - The detection threshold (default 65) can be modified in the code for fine-tuning
   - Lower values increase sensitivity but may cause false positives

### Step 3: Play the Game

Start the full game experience with AI opponent.

1. **Start the robot program**:
   If you want to play against a physical robot, start the robot program before running the game script.
   ```bash
   vier_gewinnt.urp
   ```

2. **Run the game script**:
   ```bash
   python game.py
   ```

3. **Select difficulty**:
   - Choose from Easy, Medium, Hard, or Impossible AI difficulty

4. **Play the game**:
   - The system will detect your moves automatically via webcam
   - Wait for the AI to calculate and display its move
   - Make your physical move on the board when prompted
   - The game continues until someone wins or the board is full

5. **Optional features**:
   - Click "Show Hints" to display AI evaluation scores for each column
   - Winning positions are highlighted when the game ends

## How It Works

### System Architecture

The three scripts work together in a coordinated system:

1. **`calibration.py`**: One-time setup script that captures board geometry and color profiles
2. **`detection.py`**: Background service that continuously processes webcam feed and exposes board state via socket
3. **`game.py`**: Main game orchestrator that manages AI, visualization, and game flow

### Computer Vision Pipeline

1. **Image Capture**: Webcam feed captured at 30 FPS
2. **Image Adjustment**: Contrast, saturation, and brightness corrections applied
3. **Perspective Transform**: Board corners used to warp image to rectangular grid
4. **Color Detection**: Euclidean distance comparison against calibrated player colors
5. **Piece Classification**: Threshold-based detection with gravity enforcement
6. **Bitboard Encoding**: 49-bit representation (7 columns × 6 rows) for efficient processing

### Bitboard Representation

The system uses bitboards for compact board state representation:
```
  6 13 20 27 34 41 48
 ---------------------
| 5 12 19 26 33 40 47 |
| 4 11 18 25 32 39 46 |
| 3 10 17 24 31 38 45 |
| 2  9 16 23 30 37 44 |
| 1  8 15 22 29 36 43 |
| 0  7 14 21 28 35 42 |
 ---------------------
```
- Bit 0: Bottom-left hole
- Bits increase right then up
- Player 1 and Player 2 have separate bitboards

### AI Integration

The game integrates with the `connect_four_ai` package:
- **Position Object**: Converts bitboards to AI-compatible board representation
- **AI Player**: Calculates optimal moves based on selected difficulty
- **Move Preview**: Shows AI's intended move before physical execution
- **Hint System**: Displays evaluation scores for strategic guidance

### Communication Protocol

- **Socket Server**: `detection.py` runs a TCP server on port 65432
- **JSON Messages**: Bitmasks exchanged as `{"player1": int, "player2": int}`
- **Real-time Updates**: Game polls detection server at 30 FPS

## Troubleshooting

### Common Issues

1. **Webcam Not Detected**:
   - Ensure webcam is connected and not used by other applications
   - Try changing camera index in code (default is 0)

2. **Poor Detection Accuracy**:
   - Improve lighting conditions
   - Recalibrate colors and board geometry
   - Adjust detection threshold in `detection.py`
   - Ensure pieces are fully visible and not overlapping

3. **Calibration Issues**:
   - Click corners precisely on board edges
   - Use high contrast between board and background
   - Place calibration pieces clearly in first two columns

4. **Socket Connection Errors**:
   - Ensure no firewall blocks port 65432
   - Wait for detection server to fully start (2-3 seconds)
   - Check that `detection.py` is running before starting `game.py`

5. **AI Not Responding**:
   - Verify `connect_four_ai` package is properly installed
   - Check console for error messages during AI move calculation

6. **Performance Issues**:
   - Close other applications using CPU/GPU resources
   - Reduce webcam resolution if needed
   - Ensure adequate RAM (4GB+ recommended)

### Debug Mode

For troubleshooting, you can run detection without the GUI:
```python
detector = ConnectFourDetector()
p1, p2 = detector.get_bitboards()
print(f"Player 1: {p1}, Player 2: {p2}")
```

### Resetting Calibration

To start fresh:
1. Delete `calibration.json`
2. Re-run `calibration.py`
3. Complete full calibration process

## Technical Notes

- **Detection Robustness**: Uses temporal consistency checking to prevent false detections
- **Gravity Enforcement**: Ensures pieces follow Connect Four physics (no floating pieces)
- **Color Spaces**: Operates in BGR color space for OpenCV compatibility
- **Threading**: Separate threads for webcam capture and GUI updates
- **Memory Management**: Efficient frame buffering to prevent memory leaks

## Contributing

When modifying the vision algorithms:
- Test calibration accuracy across different lighting conditions
- Validate detection robustness with various piece colors
- Ensure socket communication remains backward compatible

## License

This project follows the same license as the parent Connect Four AI project.