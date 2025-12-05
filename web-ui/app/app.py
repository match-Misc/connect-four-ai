import json
import socket
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__, template_folder='../templates', static_folder='../static')
socketio = SocketIO(app, cors_allowed_origins="*")

# Detection connection
detection_socket = None
last_bitmasks = (0, 0)

def connect_to_detection():
    global detection_socket
    try:
        detection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        detection_socket.connect(("localhost", 65432))
        print("Flask app connected to detection socket")
        return True
    except Exception:
        return False

def get_current_bitmasks():
    global detection_socket, last_bitmasks
    if not detection_socket:
        if not connect_to_detection():
            return last_bitmasks

    try:
        detection_socket.sendall(b"request")
        data = detection_socket.recv(1024)
        if data:
            bitmasks = json.loads(data.decode("utf-8"))
            last_bitmasks = (bitmasks["player1"], bitmasks["player2"])
            return last_bitmasks
    except Exception:
        # Try to reconnect
        detection_socket = None

    return last_bitmasks

def bitmasks_to_grid(player1_mask, player2_mask):
    grid = [[0 for _ in range(7)] for _ in range(6)]
    for row in range(6):
        for col in range(7):
            bit_pos = (5 - row) * 7 + col
            if player1_mask & (1 << bit_pos):
                grid[row][col] = 2  # Player 1 (red)
            elif player2_mask & (1 << bit_pos):
                grid[row][col] = 1  # Player 2 (yellow)
    return grid

def detection_poller():
    print("Detection poller started")
    while True:
        try:
            current_p1, current_p2 = get_current_bitmasks()
            grid = bitmasks_to_grid(current_p1, current_p2)
            
            # Count pieces in grid
            piece_count = sum(sum(1 for cell in row if cell != 0) for row in grid)
            
            # Log every 10th update to avoid spam
            if not hasattr(detection_poller, 'counter'):
                detection_poller.counter = 0
            detection_poller.counter += 1
            if detection_poller.counter % 10 == 0:
                print(f"Polling update #{detection_poller.counter}: p1={current_p1}, p2={current_p2}, pieces={piece_count}")
                if piece_count > 0:
                    print(f"  Grid sample (row 5): {grid[5]}")
                    print(f"  Emitting to clients: {grid}")
            
            socketio.emit('board_update', {'grid': grid})
            socketio.sleep(0.5)  # Poll every 500ms
        except Exception as e:
            print(f"Error in detection poller: {e}")
            socketio.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected to web-ui')
    # Send initial board state
    current_p1, current_p2 = get_current_bitmasks()
    grid = bitmasks_to_grid(current_p1, current_p2)
    print(f"Sending initial board state: p1={current_p1}, p2={current_p2}")
    print(f"Grid: {grid}")
    emit('board_update', {'grid': grid})

if __name__ == '__main__':
    # Start detection poller in background
    socketio.start_background_task(detection_poller)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)