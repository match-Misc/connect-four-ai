from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import json
import os
import threading
import time
import pyrealsense2 as rs

app = Flask(__name__)

class ConnectFourCalibrator:
    def __init__(self):
        self.corners = []
        self.hole_diameter = 30
        self.h_spacing = 60
        self.v_spacing = 60
        self.player1_color = None
        self.player2_color = None
        self.calibration_complete = False
        self.contrast = 100
        self.saturation = 100
        self.brightness = 0
        self.max_r = 255
        self.max_g = 255
        self.max_b = 255
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.running = True
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.current_depth_m = None
        self.frame_width = 640
        self.frame_height = 480
        self.status_text = "Ready. Click on the image to define corners."

        # Load existing calibration
        self.load_calibration()

    def load_calibration(self):
        filename = "../config/calibration.json"
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                corners_dict = data.get("corners", {})
                self.corners = [
                    tuple(corners_dict.get("top_left", ())),
                    tuple(corners_dict.get("top_right", ())),
                    tuple(corners_dict.get("bottom_left", ())),
                    tuple(corners_dict.get("bottom_right", ())),
                ]
                self.corners = [c for c in self.corners if len(c) == 2]
                self.hole_diameter = data.get("hole_diameter", self.hole_diameter)
                self.h_spacing = data.get("horizontal_spacing", self.h_spacing)
                self.v_spacing = data.get("vertical_spacing", self.v_spacing)
                self.contrast = data.get("contrast", self.contrast)
                self.saturation = data.get("saturation", self.saturation)
                self.brightness = data.get("brightness", self.brightness)
                p1 = data.get("player1_color")
                p2 = data.get("player2_color")
                if p1 and p2:
                    self.player1_color = p1
                    self.player2_color = p2
                    self.calibration_complete = True
                self.max_r = data.get("max_r", self.max_r)
                self.max_g = data.get("max_g", self.max_g)
                self.max_b = data.get("max_b", self.max_b)
                self.status_text = "Loaded previous calibration."
        except Exception as e:
            self.status_text = f"Failed to load calibration: {e}"

    def start_webcam(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.frame_width = color_profile.width()
            self.frame_height = color_profile.height()
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
            print("RealSense pipeline started successfully")
        except Exception as e:
            self.status_text = f"Error: Could not start RealSense pipeline ({e})"
            print(self.status_text)
            return False

        def capture_loop():
            while self.running:
                try:
                    frames = self.pipeline.wait_for_frames()
                    if self.align:
                        frames = self.align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    if self.depth_scale:
                        depth_m = depth_raw.astype(np.float32) * self.depth_scale
                    else:
                        depth_m = depth_raw.astype(np.float32)
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.current_depth_m = depth_m
                except Exception as e:
                    self.status_text = f"RealSense error: {e}"
                    print(self.status_text)
                time.sleep(0.001)

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_webcam(self):
        self.running = False
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

    def adjust_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (self.saturation / 100.0), 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        alpha = self.contrast / 100.0
        beta = self.brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return frame

    def draw_corners(self, frame):
        for i, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i+1}", (corner[0] + 10, corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_hole_grid(self, frame):
        if len(self.corners) == 4:
            adjusted_frame = self.adjust_image(frame)
            corners = np.array(self.corners)
            dst_points = np.array([[0, 0], [6 * self.h_spacing, 0], [0, 5 * self.v_spacing], [6 * self.h_spacing, 5 * self.v_spacing]], dtype=np.float32)
            src_points = corners.astype(np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            for row in range(6):
                for col in range(7):
                    grid_x = col * self.h_spacing
                    grid_y = row * self.v_spacing
                    grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(grid_point.reshape(1, 1, 2), np.linalg.inv(M))
                    x, y = transformed[0, 0].astype(int)
                    if 0 <= x < adjusted_frame.shape[1] and 0 <= y < adjusted_frame.shape[0]:
                        color = (255, 0, 0)
                        if col == 0 and self.calibration_complete and self.player1_color:
                            color = tuple(self.player1_color)
                        elif col == 1 and self.calibration_complete and self.player2_color:
                            color = tuple(self.player2_color)
                        elif col == 0:
                            color = (0, 0, 255)
                        elif col == 1:
                            color = (0, 255, 255)
                        cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, color, 2)
            return adjusted_frame
        return frame

    def get_frame_image(self):
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        self.draw_corners(frame)
        if len(self.corners) == 4:
            frame = self.draw_hole_grid(frame)
        return frame

    def handle_click(self, x, y):
        if len(self.corners) < 4:
            self.corners.append((int(x), int(y)))
            self.status_text = f"Corner {len(self.corners)} set at ({int(x)}, {int(y)})"
            if len(self.corners) == 4:
                # Sort corners robustly based on x and y coordinates
                sorted_by_x = sorted(self.corners, key=lambda p: p[0])
                left_corners = sorted(sorted_by_x[:2], key=lambda p: p[1])
                right_corners = sorted(sorted_by_x[2:], key=lambda p: p[1])
                
                self.corners = [
                    left_corners[0],   # top-left
                    right_corners[0],  # top-right
                    left_corners[1],   # bottom-left
                    right_corners[1]   # bottom-right
                ]
                self.status_text = "All corners defined. Adjust parameters and calibrate colors."

    def update_params(self, params):
        self.hole_diameter = int(params.get('hole_diameter', self.hole_diameter))
        self.h_spacing = int(params.get('h_spacing', self.h_spacing))
        self.v_spacing = int(params.get('v_spacing', self.v_spacing))
        self.contrast = int(params.get('contrast', self.contrast))
        self.saturation = int(params.get('saturation', self.saturation))
        self.brightness = int(params.get('brightness', self.brightness))
        self.max_r = int(params.get('max_r', self.max_r))
        self.max_g = int(params.get('max_g', self.max_g))
        self.max_b = int(params.get('max_b', self.max_b))

    def calibrate_colors(self):
        if len(self.corners) != 4 or self.current_frame is None:
            self.status_text = "Define corners and ensure camera is running."
            return False
        adjusted_frame = self.adjust_image(self.current_frame)
        corners = np.array(self.corners)
        dst_points = np.array([[0, 0], [6 * self.h_spacing, 0], [0, 5 * self.v_spacing], [6 * self.h_spacing, 5 * self.v_spacing]], dtype=np.float32)
        src_points = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        player1_samples = []
        player2_samples = []
        for col in range(2):
            for row in range(6):
                grid_x = col * self.h_spacing
                grid_y = row * self.v_spacing
                grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(grid_point.reshape(1, 1, 2), np.linalg.inv(M))
                x, y = transformed[0, 0].astype(int)
                if 0 <= x < adjusted_frame.shape[1] and 0 <= y < adjusted_frame.shape[0]:
                    radius = max(3, self.hole_diameter // 4)
                    roi = adjusted_frame[max(0, y - radius):min(adjusted_frame.shape[0], y + radius), max(0, x - radius):min(adjusted_frame.shape[1], x + radius)]
                    if roi.size > 0:
                        avg_color = cv2.mean(roi)[:3]
                        if col == 0:
                            player1_samples.append(avg_color)
                        else:
                            player2_samples.append(avg_color)
        if player1_samples and player2_samples:
            self.player1_color = np.mean(player1_samples, axis=0).astype(int).tolist()
            self.player2_color = np.mean(player2_samples, axis=0).astype(int).tolist()
            self.calibration_complete = True
            self.status_text = f"Colors calibrated. Player 1: {self.player1_color}, Player 2: {self.player2_color}"
            return True
        self.status_text = "Calibration failed."
        return False

    def save_calibration(self):
        if not self.calibration_complete:
            self.status_text = "Calibrate colors first."
            return False
        corner_dict = {"top_left": self.corners[0], "top_right": self.corners[1], "bottom_left": self.corners[2], "bottom_right": self.corners[3]}
        data = {
            "corners": corner_dict,
            "hole_diameter": self.hole_diameter,
            "horizontal_spacing": self.h_spacing,
            "vertical_spacing": self.v_spacing,
            "player1_color": self.player1_color,
            "player2_color": self.player2_color,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "brightness": self.brightness,
            "max_r": self.max_r,
            "max_g": self.max_g,
            "max_b": self.max_b,
        }
        try:
            with open("../config/calibration.json", "w") as f:
                json.dump(data, f, indent=2)
            self.status_text = "Calibration saved."
            return True
        except Exception as e:
            self.status_text = f"Save failed: {e}"
            return False

    def reset_corners(self):
        self.corners = []
        self.calibration_complete = False
        self.status_text = "Corners reset."

    def save_grid_only(self):
        if len(self.corners) != 4:
            self.status_text = "Define all 4 corners first."
            return False
        
        corner_dict = {"top_left": self.corners[0], "top_right": self.corners[1], "bottom_left": self.corners[2], "bottom_right": self.corners[3]}
        
        data = {}
        if os.path.exists("../config/calibration.json"):
            with open("../config/calibration.json", "r") as f:
                try:
                    data = json.load(f)
                except:
                    pass
                    
        data["corners"] = corner_dict
        data["horizontal_spacing"] = self.h_spacing
        data["vertical_spacing"] = self.v_spacing
        
        try:
            os.makedirs("../config", exist_ok=True)
            with open("../config/calibration.json", "w") as f:
                json.dump(data, f, indent=2)
            self.status_text = "Grid configuration saved."
            return True
        except Exception as e:
            self.status_text = f"Save failed: {e}"
            return False

    def save_max_rgb(self):
        data = {}
        if os.path.exists("../config/calibration.json"):
            with open("../config/calibration.json", "r") as f:
                try:
                    data = json.load(f)
                except:
                    pass
                    
        data["max_r"] = self.max_r
        data["max_g"] = self.max_g
        data["max_b"] = self.max_b
        
        try:
            os.makedirs("../config", exist_ok=True)
            with open("../config/calibration.json", "w") as f:
                json.dump(data, f, indent=2)
            self.status_text = "Max RGB settings saved."
            return True
        except Exception as e:
            self.status_text = f"Save failed: {e}"
            return False

calibrator = ConnectFourCalibrator()

@app.route('/')
def index():
    return render_template('detection-calibration.html',
                           hole_diameter=calibrator.hole_diameter,
                           contrast=calibrator.contrast,
                           saturation=calibrator.saturation,
                           brightness=calibrator.brightness,
                           max_r=calibrator.max_r,
                           max_g=calibrator.max_g,
                           max_b=calibrator.max_b,
                           status=calibrator.status_text,
                           p1_color=calibrator.player1_color,
                           p2_color=calibrator.player2_color)

@app.route('/frame')
def frame():
    img = calibrator.get_frame_image()
    if img is None:
        # Return a placeholder image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "No camera feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/click', methods=['POST'])
def click():
    data = request.json
    x = data['x']
    y = data['y']
    calibrator.handle_click(x, y)
    return jsonify({'status': calibrator.status_text})

@app.route('/update_params', methods=['POST'])
def update_params():
    params = request.form
    calibrator.update_params(params)
    return jsonify({'status': 'Parameters updated'})

@app.route('/update_single', methods=['POST'])
def update_single():
    name = request.form.get('name')
    value = request.form.get('value')
    if name == 'hole_diameter':
        calibrator.hole_diameter = int(value)
    elif name == 'contrast':
        calibrator.contrast = int(value)
    elif name == 'saturation':
        calibrator.saturation = int(value)
    elif name == 'brightness':
        calibrator.brightness = int(value)
    elif name == 'max_r':
        calibrator.max_r = int(value)
    elif name == 'max_g':
        calibrator.max_g = int(value)
    elif name == 'max_b':
        calibrator.max_b = int(value)
    return jsonify({'status': f'{name} updated'})

@app.route('/calibrate_colors', methods=['POST'])
def calibrate_colors():
    calibrator.calibrate_colors()
    return jsonify({'status': calibrator.status_text, 'p1_color': calibrator.player1_color, 'p2_color': calibrator.player2_color})

@app.route('/save', methods=['POST'])
def save():
    calibrator.save_calibration()
    return jsonify({'status': calibrator.status_text})

@app.route('/reset', methods=['POST'])
def reset():
    calibrator.reset_corners()
    return jsonify({'status': calibrator.status_text})

@app.route('/save_grid', methods=['POST'])
def save_grid():
    calibrator.save_grid_only()
    return jsonify({'status': calibrator.status_text})

@app.route('/save_hole', methods=['POST'])
def save_hole():
    # Save only hole diameter
    if os.path.exists("../config/calibration.json"):
        with open("../config/calibration.json", "r") as f:
            data = json.load(f)
        data["hole_diameter"] = calibrator.hole_diameter
        with open("../config/calibration.json", "w") as f:
            json.dump(data, f, indent=2)
        calibrator.status_text = "Hole diameter saved."
    else:
        calibrator.status_text = "No config file to update."
    return jsonify({'status': calibrator.status_text})

@app.route('/save_color', methods=['POST'])
def save_color():
    # Save image adjustments
    if os.path.exists("../config/calibration.json"):
        with open("../config/calibration.json", "r") as f:
            data = json.load(f)
        data["contrast"] = calibrator.contrast
        data["saturation"] = calibrator.saturation
        data["brightness"] = calibrator.brightness
        with open("../config/calibration.json", "w") as f:
            json.dump(data, f, indent=2)
        calibrator.status_text = "Color calibration saved."
    else:
        calibrator.status_text = "No config file to update."
    return jsonify({'status': calibrator.status_text})

@app.route('/save_max_rgb', methods=['POST'])
def save_max_rgb():
    calibrator.save_max_rgb()
    return jsonify({'status': calibrator.status_text})

if __name__ == '__main__':
    if calibrator.start_webcam():
        try:
            app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
        finally:
            calibrator.stop_webcam()
    else:
        print("Failed to start camera. Exiting.")