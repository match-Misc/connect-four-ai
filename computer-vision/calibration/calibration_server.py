import cv2
import numpy as np
import json
import os
import threading
import time
import pyrealsense2 as rs
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import moondream as md
from PIL import Image
import io
import base64
# Moondream models will be initialized on-demand per request based on the selected mode.
moondream_local = None
moondream_cloud = None

app = Flask(__name__, static_folder="frontend/dist", static_url_path="/static")
CORS(app)

class UnifiedCalibrator:
    def __init__(self):
        import collections
        self.show_occupancy_overlay = False
        self.temporal_smoothing = 10
        self.hole_depth_history = [[] for _ in range(42)]
        
        # Detection properties
        self.corners = []
        self.hole_diameter = 30
        self.occupancy_threshold = 0.3
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

        # RealSense properties
        self.exposure = 1500
        self.gain = 16
        self.laser_power = 150
        self.visual_preset = 0
        self.min_depth = 0
        self.max_depth = 2000
        self.emitter = 1

        # RealSense hardware
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.depth_sensor = None
        self.colorizer = rs.colorizer()

        # Threading
        self.running = True
        self.frame_lock = threading.Lock()
        self.current_color_frame = None
        self.current_depth_frame = None
        self.current_raw_depth_frame = None
        self.status_text = "Ready."
        self.is_autocalibrating = False
        self.autocalibrate_state = 0 # 0: IDLE, 1: SCANNING_EMPTY, 2: WAITING_FILLED, 3: SCANNING_FILLED, 4: QUICK
        self.autocalibrate_progress = 0.0
        self.autocalibrate_results = []
        self.empty_scan_results = {}

        self.load_detection_calibration()
        self.load_realsense_calibration()

    def load_detection_calibration(self):
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
                self.occupancy_threshold = data.get("occupancy_threshold", self.occupancy_threshold)
                self.temporal_smoothing = data.get("temporal_smoothing", self.temporal_smoothing)
                self.show_occupancy_overlay = data.get("show_occupancy_overlay", self.show_occupancy_overlay)
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
        except Exception as e:
            print(f"Failed to load detection calibration: {e}")

    def load_realsense_calibration(self):
        filename = "../config/calibrate_realsense.json"
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                self.exposure = data.get("exposure", self.exposure)
                self.gain = data.get("gain", self.gain)
                self.laser_power = data.get("laser_power", self.laser_power)
                self.visual_preset = data.get("visual_preset", self.visual_preset)
                self.min_depth = data.get("min_depth_mm", self.min_depth)
                self.max_depth = data.get("max_depth_mm", self.max_depth)
                self.emitter = data.get("emitter_enabled", self.emitter)
        except Exception as e:
            print(f"Failed to load realsense calibration: {e}")

    def apply_realsense_params(self):
        if self.depth_sensor:
            def safe_set(opt, val, delay=0.1):
                try:
                    self.depth_sensor.set_option(opt, float(val))
                    time.sleep(delay)
                except Exception as e:
                    print(f"Error setting param: {e}")
            
            # Setting visual preset first as it's a macro that changes many other settings internally
            safe_set(rs.option.visual_preset, self.visual_preset, 0.2)
            safe_set(rs.option.exposure, self.exposure)
            safe_set(rs.option.gain, self.gain)
            safe_set(rs.option.laser_power, self.laser_power)
            safe_set(rs.option.emitter_enabled, self.emitter)

    def start_webcam(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.apply_realsense_params()
            self.depth_scale = float(self.depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
            print("RealSense pipeline started successfully")
        except Exception as e:
            print(f"Error: Could not start RealSense pipeline ({e})")
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
                    
                    c_frame = np.asanyarray(color_frame.get_data())
                    filtered_depth = rs.threshold_filter(max(0.001, self.min_depth / 1000.0), max(0.001, self.max_depth / 1000.0)).process(depth_frame)
                    d_frame = np.asanyarray(self.colorizer.colorize(filtered_depth).get_data())
                    raw_d = np.asanyarray(filtered_depth.get_data())

                    with self.frame_lock:
                        self.current_color_frame = c_frame.copy()
                        self.current_depth_frame = d_frame.copy()
                        self.current_raw_depth_frame = raw_d.copy()
                except Exception as e:
                    pass
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

    def get_hole_coordinates(self):
        if len(self.corners) < 4:
            return []
        corners = np.array(self.corners)
        dst_points = np.array([[0, 0], [6 * self.h_spacing, 0], [0, 5 * self.v_spacing], [6 * self.h_spacing, 5 * self.v_spacing]], dtype=np.float32)
        src_points = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        coords = []
        for row in range(6):
            for col in range(7):
                grid_x = col * self.h_spacing
                grid_y = row * self.v_spacing
                grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(grid_point.reshape(1, 1, 2), np.linalg.inv(M))
                coords.append((int(transformed[0, 0, 0]), int(transformed[0, 0, 1])))
        return coords

    def draw_hole_grid(self, frame, is_depth_frame=False):
        if len(self.corners) == 4:
            adjusted_frame = self.adjust_image(frame)
            coords = self.get_hole_coordinates()
            
            with self.frame_lock:
                depth_frame = None
                if self.show_occupancy_overlay and self.current_raw_depth_frame is not None:
                    depth_frame = self.current_raw_depth_frame.copy()

            idx = 0
            for row in range(6):
                for col in range(7):
                    x, y = coords[idx]
                    
                    if 0 <= x < adjusted_frame.shape[1] and 0 <= y < adjusted_frame.shape[0]:
                        if self.show_occupancy_overlay and depth_frame is not None:
                            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                                radius = self.hole_diameter // 2
                                y_min, y_max = max(0, y - radius), min(depth_frame.shape[0], y + radius)
                                x_min, x_max = max(0, x - radius), min(depth_frame.shape[1], x + radius)
                                roi = depth_frame[y_min:y_max, x_min:x_max]
                                valid_pixels = roi[(roi >= self.min_depth) & (roi <= self.max_depth)]
                                
                                if valid_pixels.size > 0 and roi.size > 0 and (valid_pixels.size / roi.size) >= self.occupancy_threshold:
                                    import numpy as np
                                    d = np.median(valid_pixels)
                                else:
                                    d = 0
                                
                                self.hole_depth_history[idx].append(d)
                                self.hole_depth_history[idx] = self.hole_depth_history[idx][-self.temporal_smoothing:]
                                
                                if len(self.hole_depth_history[idx]) > 0:
                                    import numpy as np
                                    median_d = np.median(self.hole_depth_history[idx])
                                    if median_d > 0:
                                        token_color = (0, 0, 255)
                                        is_calibrated = False
                                        if not is_depth_frame and self.calibration_complete and self.player1_color and self.player2_color:
                                            c_radius = max(3, self.hole_diameter // 4)
                                            c_roi = adjusted_frame[max(0, y - c_radius):min(adjusted_frame.shape[0], y + c_radius), max(0, x - c_radius):min(adjusted_frame.shape[1], x + c_radius)]
                                            if c_roi.size > 0:
                                                avg_c = cv2.mean(c_roi)[:3]
                                                dist1 = sum((a - b) ** 2 for a, b in zip(avg_c, self.player1_color))
                                                dist2 = sum((a - b) ** 2 for a, b in zip(avg_c, self.player2_color))
                                                token_color = tuple(int(c) for c in self.player1_color) if dist1 < dist2 else tuple(int(c) for c in self.player2_color)
                                                is_calibrated = True
                                                
                                        if is_calibrated:
                                            cv2.circle(adjusted_frame, (x, y), max(2, self.hole_diameter // 2 - 4), token_color, -1)
                                            cv2.circle(adjusted_frame, (x, y), max(2, self.hole_diameter // 2 - 4), (255, 255, 255), 2)
                                        else:
                                            cv2.line(adjusted_frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
                                            cv2.line(adjusted_frame, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 2)
                                    else:
                                        cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, (0, 255, 0), 2)
                        else:
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
                    idx += 1
            return adjusted_frame
        return frame

    def get_color_frame_image(self):
        with self.frame_lock:
            if self.current_color_frame is None:
                return None
            frame = self.current_color_frame.copy()
        self.draw_corners(frame)
        if len(self.corners) == 4:
            frame = self.draw_hole_grid(frame)
        return frame

    def get_raw_color_frame_image(self):
        with self.frame_lock:
            if self.current_color_frame is None:
                return None
            return self.current_color_frame.copy()

    def get_depth_frame_image(self):
        with self.frame_lock:
            if self.current_depth_frame is None:
                return None
            frame = self.current_depth_frame.copy()
        
        self.draw_corners(frame)
        if len(self.corners) == 4:
            frame = self.draw_hole_grid(frame, is_depth_frame=True)
            
        return frame

    def handle_click(self, x, y):
        if len(self.corners) < 4:
            self.corners.append((int(x), int(y)))
        else:
            min_dist = float('inf')
            closest_idx = -1
            for i, (cx, cy) in enumerate(self.corners):
                dist = (cx - x)**2 + (cy - y)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            if closest_idx != -1:
                self.corners[closest_idx] = (int(x), int(y))
            self.status_text = f"Corner {len(self.corners)} set at ({int(x)}, {int(y)})"
            if len(self.corners) == 4:
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

    def calibrate_colors(self):
        if len(self.corners) != 4 or self.current_color_frame is None:
            self.status_text = "Define corners and ensure camera is running."
            return False
        adjusted_frame = self.adjust_image(self.current_color_frame)
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
            self.status_text = "Colors calibrated."
        return True

    def _autocalibrate_thread(self, step=1, mode='step', params=None):
        if mode == 'single':
            self.autocalibrate_state = 4
            self.status_text = "Quick Scanning..."
        elif step == 1:
            self.autocalibrate_state = 1
            self.empty_scan_results = {}
            self.status_text = "Step 1: Scanning empty board..."
        elif step == 2:
            self.autocalibrate_state = 3
            self.status_text = "Step 2: Scanning filled board..."
            
        time.sleep(0.5)
        
        if not self.pipeline:
            self.status_text = "RealSense not connected"
            self.autocalibrate_state = 0
            return
            
        coords = self.get_hole_coordinates()
        if not coords:
            self.status_text = "Failed: Corners not set"
            self.autocalibrate_state = 0
            return
            
        if params is None:
            params = {}
            
        exp_min = params.get('exp_min', 1000)
        exp_max = params.get('exp_max', 8000)
        exp_step = params.get('exp_step', 1000)
        
        gain_min = params.get('gain_min', 16)
        gain_max = params.get('gain_max', 128)
        gain_step = params.get('gain_step', 16)
        
        laser_min = params.get('laser_min', 150)
        laser_max = params.get('laser_max', 360)
        laser_step = params.get('laser_step', 50)
            
        import numpy as np
        exposures = [int(x) for x in np.arange(exp_min, exp_max + 1, exp_step)]
        gains = [int(x) for x in np.arange(gain_min, gain_max + 1, gain_step)]
        lasers = [int(x) for x in np.arange(laser_min, laser_max + 1, laser_step)]
        
        if not exposures: exposures = [1000]
        if not gains: gains = [16]
        if not lasers: lasers = [150]
        
        best_score = -1
        best_var = 999999.0
        best_params = None
        
        orig_exposure = self.exposure
        orig_gain = self.gain
        orig_laser = self.laser_power
        
        self.autocalibrate_progress = 0.0
        if mode == 'single' or step == 2:
            self.autocalibrate_results = []
            
        total_combinations = len(exposures) * len(gains) * len(lasers)
        current_idx = 0
        
        radius = self.hole_diameter // 2
        
        for e in exposures:
            if self.autocalibrate_state == 'cancelled': break
            for g in gains:
                if self.autocalibrate_state == 'cancelled': break
                for l in lasers:
                    if self.autocalibrate_state == 'cancelled': break
                    
                    current_idx += 1
                    self.autocalibrate_progress = current_idx / total_combinations
                    
                    self.exposure = e
                    self.gain = g
                    self.laser_power = l
                    
                    # Apply ONLY the sweeping parameters, individually, so one failure doesn't block the rest
                    if self.depth_sensor:
                        def loop_safe_set(opt, val):
                            try:
                                self.depth_sensor.set_option(opt, float(val))
                                time.sleep(0.1)
                            except Exception as ex:
                                print(f"Sweep error on {opt} with val {val}: {ex}")
                        
                        loop_safe_set(rs.option.exposure, e)
                        loop_safe_set(rs.option.gain, g)
                        loop_safe_set(rs.option.laser_power, l)
                            
                    time.sleep(0.6)  # MUST be >0.5s to clear the camera's internal 16-frame buffer queue!
                    
                    frames_depths = []
                    for _ in range(3):
                        time.sleep(0.05)
                        with self.frame_lock:
                            if self.current_raw_depth_frame is not None:
                                frames_depths.append(self.current_raw_depth_frame.copy())
                                
                    if not frames_depths:
                        continue
                        
                    valid_holes = 0
                    variances = []
                    
                    for (cx, cy) in coords:
                        hole_vals = []
                        for fd in frames_depths:
                            y_min, y_max = max(0, cy - radius), min(fd.shape[0], cy + radius)
                            x_min, x_max = max(0, cx - radius), min(fd.shape[1], cx + radius)
                            roi = fd[y_min:y_max, x_min:x_max]
                            valid_pixels = roi[(roi >= self.min_depth) & (roi <= self.max_depth)]
                            if valid_pixels.size > 0 and roi.size > 0 and (valid_pixels.size / roi.size) >= self.occupancy_threshold:
                                hole_vals.append(np.mean(valid_pixels))
                                
                        if len(hole_vals) == len(frames_depths):
                            valid_holes += 1
                            variances.append(np.var(hole_vals))
                            
                    avg_var = float(np.mean(variances)) if variances else 999999.0
                    
                    if mode == 'single':
                        open_holes = 42 - valid_holes
                        self.autocalibrate_results.append({
                            'exposure': e, 'gain': g, 'laser': l, 'score': open_holes, 'var': avg_var
                        })
                        if open_holes > best_score or (open_holes == best_score and avg_var < best_var):
                            best_score = open_holes
                            best_var = avg_var
                            best_params = (e, g, l)
                    else:
                        if step == 1:
                            open_holes = 42 - valid_holes
                            self.empty_scan_results[(e, g, l)] = (open_holes, avg_var)
                        elif step == 2:
                            empty_score, empty_var = self.empty_scan_results.get((e, g, l), (0, 999999.0))
                            combined_score = min(empty_score, valid_holes)
                            combined_var = (avg_var + empty_var) / 2
                            
                            self.autocalibrate_results.append({
                                'exposure': e, 'gain': g, 'laser': l, 'score': combined_score, 'var': combined_var
                            })
                            
                            if combined_score > best_score or (combined_score == best_score and combined_var < best_var):
                                best_score = combined_score
                                best_var = combined_var
                                best_params = (e, g, l)
                                
        if self.autocalibrate_state == 'cancelled':
            self.exposure = orig_exposure
            self.gain = orig_gain
            self.laser_power = orig_laser
            self.apply_realsense_params()
            self.status_text = "Calibration Cancelled"
            self.autocalibrate_state = 0
            self.is_autocalibrating = False
            return
                            
        if mode == 'single' or step == 2:
            self.autocalibrate_results.sort(key=lambda x: (-x['score'], x['var']))
            self.autocalibrate_results = self.autocalibrate_results[:10]
                            
        if mode == 'single':
            if best_params:
                self.exposure, self.gain, self.laser_power = best_params
                self.apply_realsense_params()
                self.save_realsense_calibration()
                self.status_text = f"Quick Calibrated! (Cov: {best_score}/42)"
            else:
                self.status_text = "Quick Calibrate Failed"
            self.autocalibrate_state = 0
        elif step == 1:
            self.autocalibrate_state = 2
            self.status_text = "Step 1 Complete. Waiting for tokens..."
        elif step == 2:
            if best_params:
                self.exposure, self.gain, self.laser_power = best_params
                self.apply_realsense_params()
                self.save_realsense_calibration()
                self.status_text = f"Auto Calibrated! (Min Cov: {best_score}/42)"
            else:
                self.status_text = "Auto Calibrate Failed"
            self.autocalibrate_state = 0
            
        self.is_autocalibrating = False

    def save_detection_calibration(self):
        corner_dict = {}
        if len(self.corners) == 4:
            corner_dict = {"top_left": self.corners[0], "top_right": self.corners[1], "bottom_left": self.corners[2], "bottom_right": self.corners[3]}
            
        data = {
            "corners": corner_dict,
            "hole_diameter": self.hole_diameter,
            "occupancy_threshold": self.occupancy_threshold,
            "temporal_smoothing": self.temporal_smoothing,
            "show_occupancy_overlay": self.show_occupancy_overlay,
            "horizontal_spacing": self.h_spacing,
            "vertical_spacing": self.v_spacing,
            "player1_color": self.player1_color,
            "player2_color": self.player2_color,
            "calibration_complete": self.calibration_complete,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "brightness": self.brightness,
            "max_r": self.max_r,
            "max_g": self.max_g,
            "max_b": self.max_b
        }
        try:
            with open("../config/calibration.json", "w") as f:
                json.dump(data, f, indent=2)
            self.status_text = "Calibration saved."
            return True
        except Exception as e:
            self.status_text = f"Save failed: {e}"
            return False

    def save_realsense_calibration(self):
        settings = {
            "exposure": self.exposure,
            "gain": self.gain,
            "laser_power": self.laser_power,
            "visual_preset": self.visual_preset,
            "min_depth_mm": self.min_depth,
            "max_depth_mm": self.max_depth,
            "emitter_enabled": self.emitter,
            "auto_exposure": 0
        }
        with open("../config/calibrate_realsense.json", "w") as f:
            json.dump(settings, f, indent=4)
        self.status_text = "RealSense settings saved"

    def reset_corners(self):
        self.corners = []
        self.status_text = "Corners reset."

calibrator = UnifiedCalibrator()



@app.route('/api/status')
def get_status():
    return jsonify({
        "status_text": calibrator.status_text,
        "corners": calibrator.corners,
        "calibration_complete": calibrator.calibration_complete,
        "hole_diameter": calibrator.hole_diameter,
        "occupancy_threshold": calibrator.occupancy_threshold,
        "temporal_smoothing": calibrator.temporal_smoothing,
        "contrast": calibrator.contrast,
        "saturation": calibrator.saturation,
        "brightness": calibrator.brightness,
        "max_r": calibrator.max_r,
        "max_g": calibrator.max_g,
        "max_b": calibrator.max_b,
        "exposure": calibrator.exposure,
        "gain": calibrator.gain,
        "laser_power": calibrator.laser_power,
        "visual_preset": calibrator.visual_preset,
        "min_depth": calibrator.min_depth,
        "max_depth": calibrator.max_depth,
        "emitter": calibrator.emitter,
        "player1_color": calibrator.player1_color,
        "player2_color": calibrator.player2_color,
        "is_autocalibrating": calibrator.is_autocalibrating,
        "autocalibrate_state": calibrator.autocalibrate_state,
        "autocalibrate_progress": calibrator.autocalibrate_progress,
        "autocalibrate_results": calibrator.autocalibrate_results,
        "show_occupancy_overlay": calibrator.show_occupancy_overlay
    })

def generate_frames(stream_type):
    while True:
        if stream_type == 'color':
            img = calibrator.get_color_frame_image()
        elif stream_type == 'raw':
            img = calibrator.get_raw_color_frame_image()
        else:
            img = calibrator.get_depth_frame_image()
            
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"No {stream_type} feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/frame/color')
def frame_color():
    return Response(generate_frames('color'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame/raw')
def frame_raw():
    return Response(generate_frames('raw'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame/depth')
def frame_depth():
    return Response(generate_frames('depth'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/click', methods=['POST'])
def click():
    data = request.json
    x, y = data.get('x'), data.get('y')
    calibrator.handle_click(x, y)
    return jsonify({"status": "success"})

@app.route('/api/autocalibrate/cancel', methods=['POST'])
def cancel_autocalibrate():
    if calibrator.is_autocalibrating:
        calibrator.autocalibrate_state = 'cancelled'
        calibrator.is_autocalibrating = False
    return jsonify({'status': 'Autocalibration cancelled'})

@app.route('/api/depth_measure', methods=['POST'])
def depth_measure():
    data = request.json
    x, y = data.get('x'), data.get('y')
    depth_val = 0
    with calibrator.frame_lock:
        if calibrator.current_raw_depth_frame is not None:
            if 0 <= y < calibrator.current_raw_depth_frame.shape[0] and 0 <= x < calibrator.current_raw_depth_frame.shape[1]:
                depth_val = float(calibrator.current_raw_depth_frame[int(y), int(x)])
    return jsonify({"depth_mm": depth_val})

@app.route('/api/update_detection', methods=['POST'])
def update_detection():
    data = request.json
    for key, val in data.items():
        if hasattr(calibrator, key):
            existing_val = getattr(calibrator, key)
            if existing_val is not None:
                setattr(calibrator, key, type(existing_val)(val))
            else:
                setattr(calibrator, key, val)
    return jsonify({'status': 'Detection parameters updated'})

@app.route('/api/update_realsense', methods=['POST'])
def update_realsense():
    data = request.json
    for key, val in data.items():
        if hasattr(calibrator, key):
            setattr(calibrator, key, int(val))
    calibrator.apply_realsense_params()
    return jsonify({'status': 'RealSense parameters updated'})

@app.route('/api/autocalibrate_single', methods=['POST'])
def autocalibrate_single():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(1, 'single', params), daemon=True).start()
    return jsonify({"status": "started single"})

@app.route('/api/autocalibrate_step1', methods=['POST'])
def autocalibrate_step1():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(1, 'step', params), daemon=True).start()
    return jsonify({"status": "started step 1"})

@app.route('/api/autocalibrate_step2', methods=['POST'])
def autocalibrate_step2():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(2, 'step', params), daemon=True).start()
    return jsonify({"status": "started step 2"})

@app.route('/api/save_detection', methods=['POST'])
def save_detection():
    success = calibrator.save_detection_calibration()
    if success:
        return jsonify({"status": "saved"})
    return jsonify({"status": "failed", "message": calibrator.status_text}), 400

@app.route('/api/toggle_occupancy', methods=['POST'])
def toggle_occupancy():
    calibrator.show_occupancy_overlay = not calibrator.show_occupancy_overlay
    calibrator.save_detection_calibration()
    return jsonify({'status': 'Toggled occupancy overlay', 'show_occupancy_overlay': calibrator.show_occupancy_overlay})

@app.route('/api/save_realsense', methods=['POST'])
def save_realsense():
    settings = {
        "exposure": calibrator.exposure,
        "gain": calibrator.gain,
        "laser_power": calibrator.laser_power,
        "visual_preset": calibrator.visual_preset,
        "min_depth_mm": calibrator.min_depth,
        "max_depth_mm": calibrator.max_depth,
        "emitter_enabled": calibrator.emitter,
        "auto_exposure": 0
    }
    with open("../config/calibrate_realsense.json", "w") as f:
        json.dump(settings, f, indent=4)
    calibrator.status_text = "RealSense settings saved"
    return jsonify({'status': calibrator.status_text})

@app.route('/api/calibrate_colors', methods=['POST'])
def calibrate_colors():
    calibrator.calibrate_colors()
    return jsonify({'status': calibrator.status_text})

@app.route('/api/reset', methods=['POST'])
def reset():
    calibrator.reset_corners()
    return jsonify({'status': calibrator.status_text})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

class LocalMoondream:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading local Moondream model via transformers...")
        self.model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True).to("cpu")
        print("Local Moondream model loaded.")
        
    def detect(self, image, target):
        print(f"Running detection for '{target}'...")
        try:
            # Resize image to max 640x640 to match the UI resolution and drastically speed up CPU inference
            # The bounding boxes are normalized [0, 1] so this won't break coordinates
            img_copy = image.copy()
            img_copy.thumbnail((640, 640))
            
            results = self.model.detect(img_copy, target)
            print(f"RAW MOONDREAM OUTPUT for '{target}': {results}")
            
            boxes = []
            if results and "objects" in results:
                for obj in results["objects"]:
                    boxes.append([
                        obj.get('x_min', 0), 
                        obj.get('y_min', 0), 
                        obj.get('x_max', 0), 
                        obj.get('y_max', 0)
                    ])
            return boxes
        except Exception as e:
            print(f"Detect error: {e}")
            return []

@app.route('/api/moondream/detect', methods=['POST'])
def moondream_detect():
    global moondream_local, moondream_cloud
    print("==================================================")
    print("SINGLE SHOT REQUEST RECEIVED!")
    print("==================================================")
    try:
        data = request.json
        image_data = data.get('image')
        targets = data.get('targets', ['green chip', 'black chip'])
        mode = data.get('mode', 'cloud')
        api_key = data.get('api_key', '')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
            
        # Select and initialize the correct model
        try:
            if mode == 'local':
                if not moondream_local:
                    print("--> Initializing Local Moondream 2025-01-09...")
                    print("--> Note: This may take several minutes if it is downloading the weights for the first time!")
                    moondream_local = LocalMoondream()
                active_model = moondream_local
            else:
                # Always create a new cloud client if the API key changes or isn't cached
                active_model = md.vl(api_key=api_key)
        except Exception as e:
            return jsonify({"error": f"Failed to initialize {mode} model: {str(e)}"}), 500

        start_time = time.time()
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        prep_time = (time.time() - start_time) * 1000
        
        results = {}
        call_times = {}
        
        for target in targets:
            step_start = time.time()
            detections = active_model.detect(image, target)
            
            # Extract coordinates
            boxes = []
            if detections and hasattr(detections, 'objects'):
                for obj in detections.objects:
                    boxes.append([obj.x_min, obj.y_min, obj.x_max, obj.y_max])
            elif detections and isinstance(detections, dict) and 'objects' in detections:
                for obj in detections['objects']:
                    boxes.append([obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']])
            elif isinstance(detections, list):
                for obj in detections:
                    if isinstance(obj, dict):
                        boxes.append([obj.get('x_min'), obj.get('y_min'), obj.get('x_max'), obj.get('y_max')])
                    else:
                        boxes.append([obj.x_min, obj.y_min, obj.x_max, obj.y_max])
                        
            results[target] = boxes
            call_times[target] = (time.time() - step_start) * 1000
            
        total_latency = (time.time() - start_time) * 1000
        
        return jsonify({
            "detections": results,
            "latency": {
                "total": total_latency,
                "prep": prep_time,
                "calls": call_times
            }
        })
    except Exception as e:
        print(f"Moondream error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if calibrator.start_webcam():
        try:
            app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
        finally:
            calibrator.stop_webcam()
    else:
        print("Failed to start camera. Exiting.")
