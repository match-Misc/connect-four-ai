import cv2
import numpy as np
import json
import os
import threading
import time
import pyrealsense2 as rs
from typing import List

class VisionService:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            self.config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config"))
        else:
            self.config_dir = config_dir
        self.running = False
        
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
        self.temporal_smoothing = 10
        self.hole_depth_history = [[] for _ in range(42)]
        
        # RealSense properties
        self.exposure = 1500
        self.gain = 16
        self.laser_power = 150
        self.visual_preset = 0
        self.min_depth = 0
        self.max_depth = 2000
        self.emitter = 1
        # Keep the RGB sensor fixed to the values used during colour calibration.
        self.color_exposure = 5000
        self.color_gain = 16
        self.color_auto_exposure = 0

        # RealSense hardware
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_sensor = None
        self.color_sensor = None

        # Threading
        self.frame_lock = threading.Lock()
        self.current_color_frame = None
        self.current_raw_depth_frame = None
        self.capture_thread = None

        # Detection runs on the capture thread so the depth history advances at
        # camera rate. Feeding it from get_board_state() instead made the
        # smoothing window depend on how often callers polled.
        self.board_lock = threading.Lock()
        self.latest_board = [[0 for _ in range(7)] for _ in range(6)]

        self.load_calibration()
        self.load_realsense_calibration()

    def load_calibration(self):
        filename = os.path.join(self.config_dir, "calibration.json")
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
        except Exception as e:
            print(f"Failed to load detection calibration: {e}")

    def load_realsense_calibration(self):
        filename = os.path.join(self.config_dir, "calibrate_realsense.json")
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
                self.color_exposure = data.get("color_exposure", self.color_exposure)
                self.color_gain = data.get("color_gain", self.color_gain)
                self.color_auto_exposure = data.get("color_auto_exposure", self.color_auto_exposure)
        except Exception as e:
            print(f"Failed to load realsense calibration: {e}")

    @staticmethod
    def _sensor_supports(sensor, option):
        try:
            return sensor is not None and sensor.supports(option)
        except Exception:
            return False

    def _find_color_sensor(self, device):
        for sensor in device.query_sensors():
            try:
                if not sensor.is_depth_sensor() and sensor.supports(rs.option.enable_auto_exposure):
                    return sensor
            except Exception:
                continue
        return None

    def apply_realsense_params(self):
        def safe_set(sensor, option, value, delay=0.1):
            if not self._sensor_supports(sensor, option):
                return
            try:
                sensor.set_option(option, float(value))
                time.sleep(delay)
            except Exception as e:
                print(f"Error setting {option}: {e}")

        if self.depth_sensor:
            safe_set(self.depth_sensor, rs.option.visual_preset, self.visual_preset, 0.2)
            safe_set(self.depth_sensor, rs.option.exposure, self.exposure)
            safe_set(self.depth_sensor, rs.option.gain, self.gain)
            safe_set(self.depth_sensor, rs.option.laser_power, self.laser_power)
            safe_set(self.depth_sensor, rs.option.emitter_enabled, self.emitter)
        if self.color_sensor:
            safe_set(self.color_sensor, rs.option.enable_auto_exposure, self.color_auto_exposure)
            if not self.color_auto_exposure:
                safe_set(self.color_sensor, rs.option.exposure, self.color_exposure)
                safe_set(self.color_sensor, rs.option.gain, self.color_gain)

    def start(self):
        if self.running:
            return True
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.color_sensor = self._find_color_sensor(profile.get_device())
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            self.apply_realsense_params()
            self.align = rs.align(rs.stream.color)
            self.running = True
            print("VisionService: RealSense pipeline started successfully")
        except Exception as e:
            print(f"VisionService: Error starting RealSense pipeline ({e})")
            return False

        def capture_loop():
            while self.running:
                try:
                    # Bounded wait so stop() is observed promptly; a blocking
                    # wait_for_frames() would keep this thread inside librealsense
                    # long after running goes False.
                    ok, frames = self.pipeline.try_wait_for_frames(200)
                    if not ok:
                        continue
                    if self.align:
                        frames = self.align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    
                    c_frame = np.asanyarray(color_frame.get_data())
                    filtered_depth = rs.threshold_filter(max(0.001, self.min_depth / 1000.0), max(0.001, self.max_depth / 1000.0)).process(depth_frame)
                    raw_d = np.asanyarray(filtered_depth.get_data())

                    with self.frame_lock:
                        self.current_color_frame = c_frame.copy()
                        self.current_raw_depth_frame = raw_d.copy()

                    detected = self._detect_board()
                    with self.board_lock:
                        self.latest_board = detected
                except Exception:
                    pass

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop(self):
        self.running = False
        thread, self.capture_thread = self.capture_thread, None
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                # Stopping the pipeline while the capture thread is still inside
                # librealsense segfaults the process. Leave it open instead; the
                # OS reclaims the device when we exit.
                print("VisionService: capture thread did not exit; leaving pipeline open")
                return
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"VisionService: error stopping pipeline ({e})")
            self.pipeline = None
        self.depth_sensor = None
        self.color_sensor = None
        self.align = None

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

    def adjust_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (self.saturation / 100.0), 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        alpha = self.contrast / 100.0
        beta = self.brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return frame

    def circular_roi_pixels(self, frame, cx, cy, radius):
        """Return only pixels inside the requested physical circular ROI."""
        radius = max(1, int(radius))
        y_min, y_max = max(0, cy - radius), min(frame.shape[0], cy + radius + 1)
        x_min, x_max = max(0, cx - radius), min(frame.shape[1], cx + radius + 1)
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return np.array([], dtype=frame.dtype)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        return roi[mask]

    def depth_roi_measurement(self, depth_frame, cx, cy):
        """Use the same circular physical hole area as calibration and overlay."""
        radius = max(1, self.hole_diameter // 2)
        circle_pixels = self.circular_roi_pixels(depth_frame, cx, cy, radius)
        if circle_pixels.size == 0:
            return 0.0, np.array([], dtype=depth_frame.dtype)
        valid_pixels = circle_pixels[(circle_pixels > 0) & (circle_pixels >= self.min_depth) & (circle_pixels <= self.max_depth)]
        return float(valid_pixels.size / circle_pixels.size), valid_pixels

    def get_board_state(self) -> List[List[int]]:
        # Returns the most recent detection: a 6x7 array, 0 empty / 1 player1 /
        # 2 player2. Cheap and side-effect free -- callers may poll it freely.
        with self.board_lock:
            return [row[:] for row in self.latest_board]

    def _detect_board(self) -> List[List[int]]:
        # Runs on the capture thread only: mutates hole_depth_history.
        board = [[0 for _ in range(7)] for _ in range(6)]

        if len(self.corners) != 4 or not self.calibration_complete:
            return board

        with self.frame_lock:
            color_frame = self.current_color_frame.copy() if self.current_color_frame is not None else None
            depth_frame = self.current_raw_depth_frame.copy() if self.current_raw_depth_frame is not None else None

        if color_frame is None or depth_frame is None:
            return board

        adjusted_frame = self.adjust_image(color_frame)
        coords = self.get_hole_coordinates()
        
        idx = 0
        for row in range(6):
            for col in range(7):
                x, y = coords[idx]
                if 0 <= x < adjusted_frame.shape[1] and 0 <= y < adjusted_frame.shape[0]:
                    if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                        coverage, valid_pixels = self.depth_roi_measurement(depth_frame, x, y)
                        if valid_pixels.size > 0 and coverage >= self.occupancy_threshold:
                            d = np.median(valid_pixels)
                        else:
                            d = 0
                        
                        self.hole_depth_history[idx].append(d)
                        self.hole_depth_history[idx] = self.hole_depth_history[idx][-self.temporal_smoothing:]
                        
                        if len(self.hole_depth_history[idx]) > 0:
                            median_d = np.median(self.hole_depth_history[idx])
                            if median_d > 0:
                                # Found a token
                                c_radius = max(3, self.hole_diameter // 4)
                                c_pixels = self.circular_roi_pixels(adjusted_frame, x, y, c_radius)
                                if c_pixels.size > 0:
                                    avg_c = np.mean(c_pixels, axis=0)[:3]
                                    dist1 = sum((a - b) ** 2 for a, b in zip(avg_c, self.player1_color))
                                    dist2 = sum((a - b) ** 2 for a, b in zip(avg_c, self.player2_color))
                                    board[row][col] = 1 if dist1 < dist2 else 2
                idx += 1
        return board

    def get_annotated_frame(self):
        with self.frame_lock:
            if self.current_color_frame is None:
                return None
            frame = self.current_color_frame.copy()
            
        board = self.get_board_state()
        coords = self.get_hole_coordinates()
        
        for p in self.corners:
            cv2.circle(frame, p, 5, (0, 255, 0), -1)
            
        idx = 0
        for row in range(6):
            for col in range(7):
                if idx < len(coords):
                    x, y = coords[idx]
                    token = board[row][col]
                    color = (255, 255, 255) # Empty
                    if token == 1:
                        color = (0, 255, 0) # P1
                    elif token == 2:
                        color = (0, 0, 255) # P2
                        
                    cv2.circle(frame, (x, y), self.hole_diameter // 2, color, 2)
                idx += 1
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()
        return None
