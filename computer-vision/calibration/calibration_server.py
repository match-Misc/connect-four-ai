import cv2
import numpy as np
import json
import os
import threading
import time
import pyrealsense2 as rs
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../robot-game/backend')))
from nfc_reader import nfc_reader_connected, reader_connection, start_nfc_reader

app = Flask(__name__, static_folder="frontend/dist", static_url_path="/static")
CORS(app)

class UnifiedCalibrator:
    # A calibration layout only includes physically reachable positions: every
    # token sits on the bottom of its column or on another token. Alternating
    # columns make the expected colour unambiguous at every step.
    COLOR_CALIBRATION_PLANS = {
        "fast": {"label": "Fast", "frames": 8, "stages": (1, 6),
                 "candidate_ranges": (range(75, 176, 25), range(75, 176, 25), range(-20, 21, 20))},
        "standard": {"label": "Standard", "frames": 12, "stages": (1, 3, 6),
                     "candidate_ranges": (range(50, 201, 25), range(50, 201, 25), range(-40, 41, 20))},
        "thorough": {"label": "Thorough", "frames": 20, "stages": (1, 3, 6),
                     "candidate_ranges": (range(50, 251, 25), range(50, 251, 25), range(-60, 61, 20))},
    }

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
        # RGB controls are separate from the depth controls on D4xx cameras.
        # Keep RGB automatic exposure off so calibration remains reproducible.
        self.color_exposure = 5000
        self.color_gain = 16
        self.color_auto_exposure = 0

        # RealSense hardware
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.depth_sensor = None
        self.color_sensor = None
        self.color_sensor_available = False
        self.color_exposure_supported = False
        self.colorizer = rs.colorizer()

        # Threading
        self.running = True
        self.frame_lock = threading.Lock()
        self.current_color_frame = None
        self.current_raw_depth_frame = None
        self.frame_counter = 0
        self.status_text = "Ready."
        self.is_autocalibrating = False
        self.autocalibrate_state = 0 # 0: IDLE, 1: SCANNING_EMPTY, 2: WAITING_FILLED, 3: SCANNING_FILLED, 4: QUICK
        self.autocalibrate_progress = 0.0
        self.autocalibrate_mode = None
        self.autocalibrate_results = []
        self.empty_scan_results = {}
        self.autocalibrate_step_params = None
        self.is_color_autocalibrating = False
        self.is_color_capturing = False
        self.color_autocalibrate_progress = 0.0
        self.color_autocalibrate_result = None
        self.color_autocalibrate_results = []
        self.color_calibration_phase = "idle"
        self.color_calibration_operation_started_at = None
        self.color_calibration_capture_estimate_seconds = 0.0
        self.color_calibration_analysis_estimate_seconds = 0.0
        self.color_calibration_precision = "standard"
        self.color_calibration_stage_index = None
        self.color_calibration_captures = []
        self.color_calibration_rgb_candidates = []
        self.color_calibration_active_rgb_setting = None
        self.color_calibration_capture_progress = 0.0
        self.ui_mode = "define_board"  # define_board, color_calibration, detection_calibration
        
        self.cached_grid_coords = None
        self.cached_corners = []

        # NFC state
        self.nfc_last_tag = None
        
        # Start NFC thread
        def on_nfc_scan(tag_data):
            self.nfc_last_tag = tag_data
        self.nfc_thread = start_nfc_reader(on_nfc_scan)

        self.load_detection_calibration()
        self.load_realsense_calibration()

    def load_detection_calibration(self):
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/calibration.json")
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
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/calibrate_realsense.json")
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
        """Find the dedicated RGB sensor instead of applying RGB controls to depth."""
        for sensor in device.query_sensors():
            try:
                if not sensor.is_depth_sensor() and sensor.supports(rs.option.enable_auto_exposure):
                    return sensor
            except Exception:
                continue
        return None

    def apply_colour_camera_params(self):
        """Apply only RGB settings; used repeatedly by RGB auto-calibration."""
        if not self.color_sensor:
            return False
        try:
            if self._sensor_supports(self.color_sensor, rs.option.enable_auto_exposure):
                self.color_sensor.set_option(rs.option.enable_auto_exposure, float(self.color_auto_exposure))
            if not self.color_auto_exposure:
                if self._sensor_supports(self.color_sensor, rs.option.exposure):
                    self.color_sensor.set_option(rs.option.exposure, float(self.color_exposure))
                if self._sensor_supports(self.color_sensor, rs.option.gain):
                    self.color_sensor.set_option(rs.option.gain, float(self.color_gain))
            return True
        except Exception as e:
            print(f"Error setting RGB camera parameters: {e}")
            return False

    def apply_realsense_params(self):
        def safe_set(sensor, option, value, delay=0.1):
            if not self._sensor_supports(sensor, option):
                return False
            try:
                sensor.set_option(option, float(value))
                time.sleep(delay)
                return True
            except Exception as e:
                print(f"Error setting {option}: {e}")
                return False

        if self.depth_sensor:
            # Setting visual preset first as it's a macro that changes many other settings internally.
            safe_set(self.depth_sensor, rs.option.visual_preset, self.visual_preset, 0.2)
            safe_set(self.depth_sensor, rs.option.exposure, self.exposure)
            safe_set(self.depth_sensor, rs.option.gain, self.gain)
            safe_set(self.depth_sensor, rs.option.laser_power, self.laser_power)
            safe_set(self.depth_sensor, rs.option.emitter_enabled, self.emitter)

        self.color_sensor_available = self.color_sensor is not None
        self.color_exposure_supported = self._sensor_supports(self.color_sensor, rs.option.exposure)
        if self.color_sensor:
            # Setting exposure alone does not lock it; disable auto exposure first.
            self.apply_colour_camera_params()

    def start_webcam(self):
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
                        self.frame_counter += 1
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

    @staticmethod
    def _adjust_image_with_params(frame, contrast, saturation, brightness):
        """Apply the same image processing used by token detection for a candidate setting."""
        if saturation != 100:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Use cv2 for multiplication to avoid holding the Python GIL
            s = cv2.multiply(s, saturation / 100.0)
            hsv = cv2.merge([h, s, v])
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        if contrast != 100 or brightness != 0:
            alpha = contrast / 100.0
            beta = brightness
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
        return frame

    def adjust_image(self, frame):
        return self._adjust_image_with_params(frame, self.contrast, self.saturation, self.brightness)

    def draw_corners(self, frame):
        for i, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{i+1}", (corner[0] + 10, corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def get_hole_coordinates(self):
        if len(self.corners) < 4:
            return []
            
        if self.cached_corners == self.corners and self.cached_grid_coords is not None:
            return self.cached_grid_coords
            
        corners = np.array(self.corners)
        dst_points = np.array([[0, 0], [6 * self.h_spacing, 0], [0, 5 * self.v_spacing], [6 * self.h_spacing, 5 * self.v_spacing]], dtype=np.float32)
        src_points = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        grid_points = []
        for row in range(6):
            for col in range(7):
                grid_points.append([col * self.h_spacing, row * self.v_spacing])
                
        grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(grid_points, np.linalg.inv(M))
        
        coords = []
        for i in range(42):
            coords.append((int(transformed[i, 0, 0]), int(transformed[i, 0, 1])))
            
        self.cached_grid_coords = coords
        self.cached_corners = list(self.corners)
        return coords

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
        """Return depth coverage and valid samples from the circular hole ROI."""
        radius = max(1, self.hole_diameter // 2)
        circle_pixels = self.circular_roi_pixels(depth_frame, cx, cy, radius)
        if circle_pixels.size == 0:
            return 0.0, np.array([], dtype=depth_frame.dtype)

        valid_pixels = circle_pixels[(circle_pixels > 0) & (circle_pixels >= self.min_depth) & (circle_pixels <= self.max_depth)]
        return float(valid_pixels.size / circle_pixels.size), valid_pixels

    @staticmethod
    def _raw_occupancy_metrics(coverages, threshold, expected_closed):
        """Score raw frame decisions only; temporal smoothing is excluded."""
        per_hole = []
        total_errors = 0
        transitions = 0
        for samples in coverages:
            states = np.asarray(samples, dtype=float) >= threshold
            errors = ~states if expected_closed else states
            error_count = int(np.sum(errors))
            total_errors += error_count
            transitions += int(np.sum(states[1:] != states[:-1])) if len(states) > 1 else 0
            per_hole.append({
                "error_count": error_count,
                "error_rate": float(np.mean(errors)) if len(errors) else 1.0,
                "p95_coverage": float(np.percentile(samples, 95)) if samples else 0.0,
                "p05_coverage": float(np.percentile(samples, 5)) if samples else 0.0,
                "max_coverage": float(np.max(samples)) if samples else 0.0,
                "min_coverage": float(np.min(samples)) if samples else 0.0,
                "mean_coverage": float(np.mean(samples)) if samples else 0.0,
            })

        total_frames = sum(len(samples) for samples in coverages)
        reliable_holes = sum(item["error_count"] == 0 for item in per_hole)
        # Worst robust coverage is a useful reserve around the dynamic threshold:
        # empty boards want the largest p95 low; filled boards want the smallest
        # p05 high.  It is deliberately a percentile, not a debounce.
        safety_coverage = (
            min((item["p05_coverage"] for item in per_hole), default=0.0)
            if expected_closed else
            max((item["p95_coverage"] for item in per_hole), default=1.0)
        )
        return {
            "reliable_holes": reliable_holes,
            "total_holes": len(per_hole),
            "total_errors": total_errors,
            "total_frames": total_frames,
            "possible_transitions": sum(max(0, len(samples) - 1) for samples in coverages),
            "error_rate": float(total_errors / total_frames) if total_frames else 1.0,
            "worst_error_rate": max((item["error_rate"] for item in per_hole), default=1.0),
            "safety_coverage": safety_coverage,
            "transitions": transitions,
            "per_hole": per_hole,
        }

    @staticmethod
    def _performance_score(*metric_sets):
        """Return an interpretable 0–100 score for raw, unsmoothed detection.

        A perfect raw scan scores 100. The score strongly rewards every hole
        being reliable, then penalises individual bad frames, the worst hole and
        state changes (flicker). It is deliberately independent of debounce.
        """
        metric_sets = [metrics for metrics in metric_sets if metrics]
        if not metric_sets:
            return 0.0
        total_holes = sum(metrics.get('total_holes', 42) for metrics in metric_sets)
        reliable_ratio = sum(metrics['reliable_holes'] for metrics in metric_sets) / total_holes if total_holes else 0.0
        total_errors = sum(metrics['total_errors'] for metrics in metric_sets)
        total_frames = sum(metrics['total_frames'] for metrics in metric_sets)
        frame_accuracy = 1.0 - (total_errors / total_frames) if total_frames else 0.0
        worst_hole_accuracy = 1.0 - max(metrics['worst_error_rate'] for metrics in metric_sets)
        transitions = sum(metrics['transitions'] for metrics in metric_sets)
        possible_transitions = sum(metrics['possible_transitions'] for metrics in metric_sets)
        stability = 1.0 - (transitions / possible_transitions) if possible_transitions else 1.0
        return round(float(np.clip(
            60 * reliable_ratio + 25 * frame_accuracy + 10 * worst_hole_accuracy + 5 * stability,
            0, 100,
        )), 1)

    @staticmethod
    def _recommended_occupancy_threshold(empty_metrics, filled_metrics):
        """Suggest a value only when every observed empty/filled sample separates."""
        if not empty_metrics or not filled_metrics:
            return None
        # Unlike the displayed p95/p05 reserve, a setting written for the game
        # must cover every sampled raw frame, including the flicker outliers.
        empty_upper = max((item["max_coverage"] for item in empty_metrics["per_hole"]), default=1.0)
        filled_lower = min((item["min_coverage"] for item in filled_metrics["per_hole"]), default=0.0)
        if filled_lower <= empty_upper:
            return None
        return round(float(np.clip((empty_upper + filled_lower) / 2.0, 0.05, 0.95)), 2)

    def draw_hole_grid(self, frame, is_depth_frame=False):
        if len(self.corners) == 4:
            if not is_depth_frame and self.ui_mode in ["color_calibration", "detection_calibration"]:
                adjusted_frame = self.adjust_image(frame)
            else:
                adjusted_frame = frame
                
            coords = self.get_hole_coordinates()
            
            with self.frame_lock:
                depth_frame = None
                if self.ui_mode in ["detection_calibration", "realsense"] and self.current_raw_depth_frame is not None:
                    depth_frame = self.current_raw_depth_frame.copy()

            idx = 0
            for row in range(6):
                for col in range(7):
                    x, y = coords[idx]
                    
                    if 0 <= x < adjusted_frame.shape[1] and 0 <= y < adjusted_frame.shape[0]:
                        if self.ui_mode == "detection_calibration" and depth_frame is not None:
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
                                        token_color = (0, 0, 255)
                                        is_calibrated = False
                                        if not is_depth_frame and self.calibration_complete and self.player1_color and self.player2_color:
                                            c_radius = max(3, self.hole_diameter // 4)
                                            c_pixels = self.circular_roi_pixels(adjusted_frame, x, y, c_radius)
                                            if c_pixels.size > 0:
                                                avg_c = np.mean(c_pixels, axis=0)[:3]
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
                        elif self.ui_mode == "realsense" and depth_frame is not None:
                            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                                coverage, valid_pixels = self.depth_roi_measurement(depth_frame, x, y)
                                if valid_pixels.size > 0 and coverage >= self.occupancy_threshold:
                                    # Closed / Blocked
                                    cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, (0, 0, 255), 3)
                                    cv2.line(adjusted_frame, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 3)
                                    cv2.line(adjusted_frame, (x - 10, y + 10), (x + 10, y - 10), (0, 0, 255), 3)
                                else:
                                    # Open
                                    cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, (255, 255, 255), 2)
                        elif self.ui_mode == "color_calibration":
                            # Show only the slots required for the active guided
                            # step. Rings preserve the camera image so users can
                            # verify that the physical token has the right colour.
                            active_rows = self._current_colour_stage()
                            is_required = active_rows is None or row >= 6 - active_rows
                            if is_required:
                                color = (20, 20, 20) if col % 2 == 0 else (0, 200, 0)
                                cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, color, 3)
                                cv2.putText(adjusted_frame, "P1" if col % 2 == 0 else "P2", (x - 10, y + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                        else:
                            # Just draw the grid for define_board
                            cv2.circle(adjusted_frame, (x, y), self.hole_diameter // 2, (255, 255, 255), 2)
                            cv2.circle(adjusted_frame, (x, y), 2, (255, 255, 255), -1)
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

    def _colour_stage_slots(self, filled_rows):
        """Bottom-up slots for a legal Connect Four position."""
        return [(row, col) for row in range(6 - filled_rows, 6) for col in range(7)]

    def _current_colour_stage(self):
        if self.color_calibration_stage_index is None:
            return None
        plan = self.COLOR_CALIBRATION_PLANS[self.color_calibration_precision]
        if self.color_calibration_stage_index >= len(plan["stages"]):
            return None
        return plan["stages"][self.color_calibration_stage_index]

    def _colour_stage_instruction(self, filled_rows):
        return (
            f"Fill the bottom {filled_rows} row{'s' if filled_rows != 1 else ''}: "
            "green Player 1 tokens in columns 1, 3, 5 and 7; "
            "black Player 2 tokens in columns 2, 4 and 6."
        )

    def _colour_rgb_candidates(self):
        """Small, bounded RGB sweep around the current manual exposure."""
        baseline = int(np.clip(self.color_exposure, 41, 10000))
        if self.color_calibration_precision == "fast":
            offsets, gains = (-2000, 0, 2000), (16, 32)
        elif self.color_calibration_precision == "standard":
            offsets, gains = (-3000, -1500, 0, 1500, 3000), (16, 32)
        else:
            offsets, gains = (-4000, -2500, -1000, 0, 1000, 2500, 4000), (16, 32, 64)
        exposures = sorted({int(np.clip(baseline + offset, 41, 10000)) for offset in offsets})
        return [(exposure, gain) for exposure in exposures for gain in gains]

    def _colour_frames_per_rgb_setting(self):
        # The RGB sweep has many settings, so take a stable short burst at each;
        # the median across the burst is more useful than one long auto-exposed frame.
        return {"fast": 4, "standard": 4, "thorough": 6}[self.color_calibration_precision]

    def _colour_capture_estimate_seconds(self):
        return len(self.color_calibration_rgb_candidates) * (0.5 + self._colour_frames_per_rgb_setting() * 0.07)

    def _colour_analysis_estimate_seconds(self):
        ranges = self.COLOR_CALIBRATION_PLANS[self.color_calibration_precision]["candidate_ranges"]
        candidates = len(self.color_calibration_rgb_candidates) * len(ranges[0]) * len(ranges[1]) * len(ranges[2])
        # These operations are on small ROI sample arrays, not full camera frames.
        return max(2.0, candidates * 0.002)

    def colour_calibration_eta_seconds(self):
        if self.color_calibration_phase == "capturing":
            return max(0.0, self.color_calibration_capture_estimate_seconds * (1 - self.color_calibration_capture_progress))
        if self.color_calibration_phase == "optimising":
            if self.color_autocalibrate_progress > 0.02 and self.color_calibration_operation_started_at:
                elapsed = time.time() - self.color_calibration_operation_started_at
                return max(0.0, elapsed * (1 - self.color_autocalibrate_progress) / self.color_autocalibrate_progress)
            return max(0.0, self.color_calibration_analysis_estimate_seconds * (1 - self.color_autocalibrate_progress))
        return 0.0

    def _slot_samples(self, frame, slots):
        """Return the same circular centre samples used by game detection."""
        coords = self.get_hole_coordinates()
        samples, labels = [], []
        # The game classifies a token from the centre quarter-radius of its
        # circular hole ROI. Calibration must use that identical physical area;
        # sampling the former larger half-radius mixed in the hole rim/background.
        radius = max(3, self.hole_diameter // 4)
        for row, col in slots:
            index = row * 7 + col
            x, y = coords[index]
            if not (radius <= x < frame.shape[1] - radius and radius <= y < frame.shape[0] - radius):
                continue
            roi = frame[y - radius:y + radius + 1, x - radius:x + radius + 1]
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            samples.append(np.array(cv2.mean(roi, mask=mask)[:3], dtype=np.float32))
            labels.append(col % 2)
        return np.asarray(samples, dtype=np.float32), np.asarray(labels, dtype=np.int8)

    def _reference_slot_samples(self, frame):
        # Manual calibration retains the original full-board reference layout.
        return self._slot_samples(frame, self._colour_stage_slots(6))

    @staticmethod
    def _adjust_colour_samples(samples, contrast, saturation, brightness):
        """Apply the detector's colour transform to sampled BGR values."""
        pixels = np.clip(samples, 0, 255).astype(np.uint8).reshape(1, -1, 3)
        return UnifiedCalibrator._adjust_image_with_params(pixels, contrast, saturation, brightness).reshape(-1, 3).astype(np.float32)

    @staticmethod
    def _colour_calibration_score(samples, labels):
        """Leave-one-out accuracy and separation margin for labelled samples."""
        if len(samples) < 4 or len(np.unique(labels)) != 2:
            return None
        correct = 0
        margins = []
        indices = np.arange(len(samples))
        for i, sample in enumerate(samples):
            means = []
            for label in (0, 1):
                group = samples[(labels == label) & (indices != i)]
                if len(group) == 0:
                    return None
                means.append(np.median(group, axis=0))
            distances = np.array([np.linalg.norm(sample - mean) for mean in means])
            expected = labels[i]
            correct += int(np.argmin(distances) == expected)
            margins.append(distances[1 - expected] - distances[expected])
        return correct, float(np.mean(margins))

    def calibrate_colors(self):
        if len(self.corners) != 4 or self.current_color_frame is None:
            self.status_text = "Define corners and ensure camera is running."
            return False
        # A manual reference must be captured under the same locked RGB setting.
        self.color_auto_exposure = 0
        self.apply_realsense_params()
        with self.frame_lock:
            frame = self.current_color_frame.copy()
        samples, labels = self._reference_slot_samples(self.adjust_image(frame))
        if len(samples[labels == 0]) and len(samples[labels == 1]):
            self.player1_color = np.median(samples[labels == 0], axis=0).astype(int).tolist()
            self.player2_color = np.median(samples[labels == 1], axis=0).astype(int).tolist()
            self.calibration_complete = True
            self.status_text = "Colours calibrated from the full reference layout."
        return True

    def start_colour_calibration(self, precision):
        if precision not in self.COLOR_CALIBRATION_PLANS:
            self.status_text = "Choose Fast, Standard, or Thorough colour calibration."
            return False
        if len(self.corners) != 4 or self.current_color_frame is None:
            self.status_text = "Define corners and ensure camera is running."
            return False
        if self.is_color_capturing or self.is_color_autocalibrating:
            self.status_text = "Colour calibration is already running."
            return False
        if not self.color_sensor_available or not self.color_exposure_supported:
            self.status_text = "This camera does not expose manual RGB exposure; colour calibration cannot be locked."
            return False
        # A changing RGB exposure invalidates a colour reference. Always lock it
        # before collecting the first stage, even if the UI was toggled earlier.
        self.color_auto_exposure = 0
        self.apply_realsense_params()
        self.color_calibration_precision = precision
        self.color_calibration_stage_index = 0
        self.color_calibration_captures = []
        # The sweep depends on the selected precision and current RGB exposure,
        # so generate it at the start of every guided run.
        self.color_calibration_rgb_candidates = self._colour_rgb_candidates()
        self.color_calibration_active_rgb_setting = None
        self.color_calibration_capture_progress = 0.0
        self.color_autocalibrate_progress = 0.0
        self.color_autocalibrate_result = None
        self.color_autocalibrate_results = []
        self.color_calibration_phase = "waiting"
        self.color_calibration_capture_estimate_seconds = self._colour_capture_estimate_seconds()
        self.color_calibration_analysis_estimate_seconds = self._colour_analysis_estimate_seconds()
        self.status_text = "Step 1: " + self._colour_stage_instruction(self._current_colour_stage())
        return True

    def _capture_colour_stage_thread(self):
        try:
            filled_rows = self._current_colour_stage()
            if filled_rows is None:
                self.status_text = "Start colour calibration before capturing a stage."
                return
            slots = self._colour_stage_slots(filled_rows)
            settings = self.color_calibration_rgb_candidates
            frames_per_setting = self._colour_frames_per_rgb_setting()
            self.color_calibration_phase = "capturing"
            self.color_calibration_operation_started_at = time.time()
            self.color_calibration_capture_estimate_seconds = self._colour_capture_estimate_seconds()
            if not settings:
                self.status_text = "No RGB exposure settings are available for calibration."
                return

            for setting_index, (exposure, gain) in enumerate(settings):
                self.color_calibration_active_rgb_setting = {"exposure": exposure, "gain": gain}
                self.color_exposure, self.color_gain, self.color_auto_exposure = exposure, gain, 0
                if not self.apply_colour_camera_params():
                    self.status_text = "Could not apply the RGB exposure/gain setting."
                    return
                # Clear the sensor's already queued frames after changing exposure.
                time.sleep(0.5)
                frame_samples = []
                for frame_index in range(frames_per_setting):
                    with self.frame_lock:
                        frame = None if self.current_color_frame is None else self.current_color_frame.copy()
                    if frame is not None:
                        samples, labels = self._slot_samples(frame, slots)
                        if len(samples) == len(slots):
                            frame_samples.append(samples)
                    completed = setting_index * frames_per_setting + frame_index + 1
                    self.color_calibration_capture_progress = completed / (len(settings) * frames_per_setting)
                    time.sleep(0.07)
                if not frame_samples:
                    self.status_text = "No valid colour samples captured. Check the board outline and camera feed."
                    return
                sample_stack = np.stack(frame_samples)
                self.color_calibration_captures.append({
                    "filled_rows": filled_rows,
                    "samples": np.median(sample_stack, axis=0),
                    "labels": labels,
                    "rgb_exposure": exposure,
                    "rgb_gain": gain,
                    "frame_variance": float(np.mean(np.var(sample_stack, axis=0))),
                })

            self.color_calibration_active_rgb_setting = None
            self.color_calibration_stage_index += 1
            if self._current_colour_stage() is None:
                self.status_text = "All layouts captured. Optimising RGB camera and colour settings…"
                self.color_calibration_phase = "optimising"
                self.color_calibration_operation_started_at = time.time()
                self.is_color_autocalibrating = True
                threading.Thread(target=self._autocalibrate_colours_thread, daemon=True).start()
            else:
                step = self.color_calibration_stage_index + 1
                self.status_text = f"Step {step}: " + self._colour_stage_instruction(self._current_colour_stage())
        except Exception as exc:
            self.status_text = f"Colour capture failed: {exc}"
        finally:
            self.is_color_capturing = False
            self.color_calibration_capture_progress = 1.0
            if self.color_calibration_phase == "capturing":
                self.color_calibration_phase = "waiting"

    def capture_colour_stage(self):
        if self.is_color_capturing or self.is_color_autocalibrating:
            return False
        if self._current_colour_stage() is None:
            self.status_text = "Start a new colour calibration to capture another layout."
            return False
        if not self.color_calibration_rgb_candidates:
            self.status_text = "Keine RGB-Kandidaten vorbereitet. Kalibrierung bitte neu starten."
            return False
        self.is_color_capturing = True
        self.color_calibration_capture_progress = 0.0
        self.status_text = "Capturing stable colour samples… keep the board still."
        threading.Thread(target=self._capture_colour_stage_thread, daemon=True).start()
        return True

    def _autocalibrate_colours_thread(self):
        try:
            if not self.color_calibration_captures:
                self.status_text = "Capture all guided layouts before optimising colours."
                return

            # Each exposure/gain setting has samples from every requested board
            # occupancy. Evaluate it jointly with every software colour filter.
            captures_by_rgb = {}
            for capture in self.color_calibration_captures:
                key = (capture["rgb_exposure"], capture["rgb_gain"])
                captures_by_rgb.setdefault(key, []).append(capture)
            ranges = self.COLOR_CALIBRATION_PLANS[self.color_calibration_precision]["candidate_ranges"]
            image_candidates = [(contrast, saturation, brightness)
                                for contrast in ranges[0] for saturation in ranges[1] for brightness in ranges[2]]
            total = len(captures_by_rgb) * len(image_candidates)
            top_results = []
            completed = 0
            for (exposure, gain), captures in captures_by_rgb.items():
                raw_samples = np.concatenate([capture["samples"] for capture in captures])
                # These are tie-breakers only: prefer an unclipped, stable image
                # and lower gain when colour classification is otherwise equal.
                clipping = float(np.mean((raw_samples <= 3) | (raw_samples >= 252)))
                frame_variance = float(np.mean([capture["frame_variance"] for capture in captures]))
                for contrast, saturation, brightness in image_candidates:
                    stage_scores, stage_samples = [], []
                    for capture in captures:
                        samples = self._adjust_colour_samples(capture["samples"], contrast, saturation, brightness)
                        score = self._colour_calibration_score(samples, capture["labels"])
                        if score is None:
                            stage_scores = []
                            break
                        stage_scores.append((score, len(capture["labels"])))
                        stage_samples.append((samples, capture["labels"]))
                    if stage_scores:
                        # Every occupancy stage has equal influence. Without this,
                        # the 42-token layout would outweigh the 7-token layout.
                        accuracy = float(np.mean([score[0] / count for score, count in stage_scores]))
                        margin = float(np.mean([score[1] for score, _ in stage_scores]))
                        rank = (accuracy, margin, -clipping, -frame_variance, -gain)
                        record = {
                            "accuracy": round(accuracy * 100, 1),
                            "margin": round(margin, 2),
                            "clipping": round(clipping * 100, 2),
                            "frame_variance": round(frame_variance, 2),
                            "rgb_exposure": exposure,
                            "rgb_gain": gain,
                            "contrast": contrast,
                            "saturation": saturation,
                            "brightness": brightness,
                            "player1_color": np.median(np.stack([np.median(samples[labels == 0], axis=0) for samples, labels in stage_samples]), axis=0).astype(int).tolist(),
                            "player2_color": np.median(np.stack([np.median(samples[labels == 1], axis=0) for samples, labels in stage_samples]), axis=0).astype(int).tolist(),
                            "_rank": rank,
                        }
                        top_results.append(record)
                        top_results.sort(key=lambda result: result["_rank"], reverse=True)
                        del top_results[8:]
                    completed += 1
                    self.color_autocalibrate_progress = completed / total

            if not top_results:
                self.status_text = "Auto colour calibration failed: no valid reference slots."
                return
            selected = top_results[0]
            self.color_autocalibrate_results = [{key: value for key, value in result.items() if key != "_rank"} for result in top_results]
            self.color_exposure = selected["rgb_exposure"]
            self.color_gain = selected["rgb_gain"]
            self.contrast = selected["contrast"]
            self.saturation = selected["saturation"]
            self.brightness = selected["brightness"]
            self.color_auto_exposure = 0
            self.apply_colour_camera_params()
            self.player1_color = selected["player1_color"]
            self.player2_color = selected["player2_color"]
            self.calibration_complete = True
            self.color_autocalibrate_result = {
                **self.color_autocalibrate_results[0],
                "total": sum(len(capture["labels"]) for capture in captures_by_rgb[(self.color_exposure, self.color_gain)]),
                "rgb_candidates": len(captures_by_rgb),
                "precision": self.color_calibration_precision,
                "stages": sorted({capture["filled_rows"] for capture in self.color_calibration_captures}),
            }
            self.status_text = f"Auto calibrated RGB exposure {self.color_exposure}, gain {self.color_gain}, and colour filters ({selected['accuracy']:.1f}% stage-balanced accuracy)."
        except Exception as exc:
            self.status_text = f"Auto colour calibration failed: {exc}"
        finally:
            self.color_autocalibrate_progress = 1.0
            self.is_color_autocalibrating = False
            self.color_calibration_phase = "complete"

    def use_colour_calibration_result(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self.color_autocalibrate_results):
            self.status_text = "Unknown RGB calibration result."
            return False
        selected = self.color_autocalibrate_results[index]
        self.color_exposure = selected["rgb_exposure"]
        self.color_gain = selected["rgb_gain"]
        self.color_auto_exposure = 0
        self.contrast = selected["contrast"]
        self.saturation = selected["saturation"]
        self.brightness = selected["brightness"]
        self.player1_color = selected["player1_color"]
        self.player2_color = selected["player2_color"]
        self.calibration_complete = True
        self.apply_colour_camera_params()
        self.color_autocalibrate_result = {**selected, "selected_manually": index != 0}
        self.status_text = f"Applied RGB result {index + 1}. Save RGB settings and calibrated colours to persist it."
        return True

    def _autocalibrate_thread(self, step=1, mode='step', params=None):
        quick_modes = {'single', 'filled', 'partial'}
        if mode in quick_modes:
            self.autocalibrate_state = 4
            self.autocalibrate_mode = mode
            quick_label = {
                'single': 'empty board',
                'filled': 'fully filled board',
                'partial': 'partially filled board',
            }[mode]
            self.status_text = f"Quick scanning {quick_label}..."
        elif step == 1:
            self.autocalibrate_state = 1
            self.autocalibrate_mode = 'step'
            self.empty_scan_results = {}
            self.status_text = "Step 1: Scanning empty board..."
        elif step == 2:
            self.autocalibrate_state = 3
            self.autocalibrate_mode = 'step'
            self.status_text = "Step 2: Scanning filled board..."

        time.sleep(0.5)
        if not self.pipeline:
            self.status_text = "RealSense not connected"
            self.autocalibrate_state = 0
            self.autocalibrate_mode = None
            return

        coords = self.get_hole_coordinates()
        if not coords:
            self.status_text = "Failed: Corners not set"
            self.autocalibrate_state = 0
            self.autocalibrate_mode = None
            return

        # Step 2 must sweep precisely the same parameter grid as step 1; its
        # cached empty-board measurements are keyed by that exact tuple.
        if mode == 'step' and step == 2 and self.autocalibrate_step_params:
            params = self.autocalibrate_step_params
        else:
            params = params or {}
        if mode == 'step' and step == 1:
            self.autocalibrate_step_params = dict(params)
        exp_min, exp_max, exp_step = params.get('exp_min', 1000), params.get('exp_max', 8000), params.get('exp_step', 1000)
        gain_min, gain_max, gain_step = params.get('gain_min', 16), params.get('gain_max', 128), params.get('gain_step', 16)
        laser_min, laser_max, laser_step = params.get('laser_min', 150), params.get('laser_max', 360), params.get('laser_step', 50)
        duration = float(params.get('duration', 3.0))
        num_frames = max(3, int(duration * 10))
        sleep_between_frames = 0.1

        exposures = [int(x) for x in np.arange(exp_min, exp_max + 1, exp_step)] or [1000]
        gains = [int(x) for x in np.arange(gain_min, gain_max + 1, gain_step)] or [16]
        lasers = [int(x) for x in np.arange(laser_min, laser_max + 1, laser_step)] or [150]
        best_rank = None
        best_params = None
        orig_exposure, orig_gain, orig_laser = self.exposure, self.gain, self.laser_power

        # Partial quick calibration uses only holes that are already detected as
        # closed in several current frames. This avoids guessing positions, but
        # deliberately does not claim to validate the empty holes.
        measurement_coords = coords
        if mode == 'partial':
            reference_samples = [[] for _ in coords]
            for _ in range(5):
                with self.frame_lock:
                    reference_frame = (self.current_raw_depth_frame.copy()
                                       if self.current_raw_depth_frame is not None else None)
                if reference_frame is not None:
                    for index, (cx, cy) in enumerate(coords):
                        coverage, _ = self.depth_roi_measurement(reference_frame, cx, cy)
                        reference_samples[index].append(coverage)
                time.sleep(0.08)
            closed_indices = [
                index for index, samples in enumerate(reference_samples)
                if samples and float(np.median(samples)) >= self.occupancy_threshold
            ]
            if not closed_indices:
                self.status_text = ('Quick calibration failed: no reliably closed holes were found. '
                                    'Place stones in the board or use the empty-board quick scan.')
                self.autocalibrate_state = 0
                self.autocalibrate_mode = None
                self.is_autocalibrating = False
                return
            measurement_coords = [coords[index] for index in closed_indices]
            self.status_text = (f"Quick scanning {len(measurement_coords)}/42 currently closed reference holes...")

        self.autocalibrate_progress = 0.0
        if mode in quick_modes or step == 2:
            self.autocalibrate_results = []
        total_combinations = len(exposures) * len(gains) * len(lasers)
        current_idx = 0

        for e in exposures:
            if self.autocalibrate_state == 'cancelled': break
            for g in gains:
                if self.autocalibrate_state == 'cancelled': break
                for l in lasers:
                    if self.autocalibrate_state == 'cancelled': break
                    current_idx += 1
                    self.autocalibrate_progress = current_idx / total_combinations
                    self.exposure, self.gain, self.laser_power = e, g, l

                    if self.depth_sensor:
                        for option, value in ((rs.option.exposure, e), (rs.option.gain, g), (rs.option.laser_power, l)):
                            try:
                                self.depth_sensor.set_option(option, float(value))
                                time.sleep(0.1)
                            except Exception as ex:
                                print(f"Sweep error on {option} with value {value}: {ex}")
                    time.sleep(0.6)  # Clear the camera's internal frame queue.

                    frames_depths = []
                    for _ in range(num_frames):
                        time.sleep(sleep_between_frames)
                        with self.frame_lock:
                            if self.current_raw_depth_frame is not None:
                                frames_depths.append(self.current_raw_depth_frame.copy())
                    if not frames_depths:
                        continue

                    # The circular ROI and the current occupancy threshold are used
                    # exactly as in live detection.  This intentionally does NOT
                    # replay temporal smoothing: flicker must remain visible here.
                    coverages = []
                    depth_values = []
                    for cx, cy in measurement_coords:
                        hole_coverages = []
                        hole_depths = []
                        for frame in frames_depths:
                            coverage, valid_pixels = self.depth_roi_measurement(frame, cx, cy)
                            hole_coverages.append(coverage)
                            if valid_pixels.size:
                                hole_depths.append(float(np.mean(valid_pixels)))
                        coverages.append(hole_coverages)
                        depth_values.append(hole_depths)

                    expected_closed = mode in {'filled', 'partial'} or (step == 2 and mode != 'single')
                    metrics = self._raw_occupancy_metrics(
                        coverages, self.occupancy_threshold, expected_closed=expected_closed
                    )
                    depth_variances = [np.var(values) for values in depth_values if len(values) > 1]
                    avg_var = float(np.mean(depth_variances)) if depth_variances else 999999.0

                    if mode in quick_modes:
                        # A quick scan validates one known board state only. Empty
                        # scans seek raw-open holes; filled/partial scans seek
                        # raw-closed holes. Temporal smoothing stays excluded.
                        performance_score = self._performance_score(metrics)
                        reserve_rank = metrics['safety_coverage'] if expected_closed else -metrics['safety_coverage']
                        rank = (
                            performance_score, metrics['reliable_holes'], -metrics['total_errors'],
                            -metrics['worst_error_rate'], reserve_rank,
                            -metrics['transitions'], -avg_var,
                        )
                        result = {
                            'exposure': e, 'gain': g, 'laser': l,
                            'score': metrics['reliable_holes'], 'reference_holes': metrics['total_holes'],
                            'quick_target': 'closed' if expected_closed else 'open',
                            'quick_mode': mode,
                            'performance_score': performance_score, 'var': avg_var,
                            'raw_errors': metrics['total_errors'],
                            'worst_error_rate': metrics['worst_error_rate'],
                            'flicker_transitions': metrics['transitions'],
                            'suggested_occupancy_threshold': None,
                            '_rank': rank,
                        }
                        if expected_closed:
                            result['filled_p05_coverage'] = metrics['safety_coverage']
                        else:
                            result['empty_p95_coverage'] = metrics['safety_coverage']
                        self.autocalibrate_results.append(result)
                        if best_rank is None or rank > best_rank:
                            best_rank, best_params = rank, (e, g, l)
                    elif step == 1:
                        self.empty_scan_results[(e, g, l)] = {
                            'metrics': metrics,
                            'var': avg_var,
                        }
                    else:
                        empty = self.empty_scan_results.get((e, g, l))
                        if not empty:
                            continue
                        empty_metrics = empty['metrics']
                        # Both scans matter: all holes must be raw-open when empty
                        # and raw-closed when filled, before debounce can help.
                        score = min(empty_metrics['reliable_holes'], metrics['reliable_holes'])
                        raw_errors = empty_metrics['total_errors'] + metrics['total_errors']
                        worst_error_rate = max(empty_metrics['worst_error_rate'], metrics['worst_error_rate'])
                        flicker_transitions = empty_metrics['transitions'] + metrics['transitions']
                        coverage_gap = metrics['safety_coverage'] - empty_metrics['safety_coverage']
                        combined_var = (empty['var'] + avg_var) / 2
                        suggested_threshold = self._recommended_occupancy_threshold(empty_metrics, metrics)
                        performance_score = self._performance_score(empty_metrics, metrics)
                        rank = (
                            performance_score, score,
                            empty_metrics['reliable_holes'] + metrics['reliable_holes'],
                            -raw_errors, -worst_error_rate, coverage_gap,
                            -flicker_transitions, -combined_var,
                        )
                        result = {
                            'exposure': e, 'gain': g, 'laser': l, 'score': score, 'performance_score': performance_score, 'var': combined_var,
                            'empty_reliable_holes': empty_metrics['reliable_holes'],
                            'filled_reliable_holes': metrics['reliable_holes'],
                            'raw_errors': raw_errors,
                            'worst_error_rate': worst_error_rate,
                            'empty_p95_coverage': empty_metrics['safety_coverage'],
                            'filled_p05_coverage': metrics['safety_coverage'],
                            'coverage_gap': coverage_gap,
                            'flicker_transitions': flicker_transitions,
                            'suggested_occupancy_threshold': suggested_threshold,
                            '_rank': rank,
                        }
                        self.autocalibrate_results.append(result)
                        if best_rank is None or rank > best_rank:
                            best_rank, best_params = rank, (e, g, l)

        if self.autocalibrate_state == 'cancelled':
            self.exposure, self.gain, self.laser_power = orig_exposure, orig_gain, orig_laser
            self.apply_realsense_params()
            self.status_text = "Calibration Cancelled"
            self.autocalibrate_state = 0
            self.autocalibrate_mode = None
            self.is_autocalibrating = False
            return

        if mode in quick_modes or step == 2:
            self.autocalibrate_results.sort(key=lambda item: item['_rank'], reverse=True)
            for result in self.autocalibrate_results:
                result.pop('_rank', None)

        if mode in quick_modes:
            if best_params:
                self.exposure, self.gain, self.laser_power = best_params
                self.apply_realsense_params()
                self.save_realsense_calibration()
                target_label = 'raw-open' if mode == 'single' else 'raw-closed'
                reference_holes = len(measurement_coords)
                self.status_text = (f"Quick calibrated: {best_rank[0]}/100 performance; "
                                    f"{reference_holes} reference holes are {target_label}.")
            else:
                self.status_text = "Quick Calibrate Failed"
            self.autocalibrate_state = 0
            self.autocalibrate_mode = None
        elif step == 1:
            self.autocalibrate_state = 2
            self.status_text = "Step 1 complete. Fill every hole with a token, then scan step 2."
        else:
            if best_params:
                self.exposure, self.gain, self.laser_power = best_params
                self.apply_realsense_params()
                self.save_realsense_calibration()
                self.status_text = f"Auto calibrated: {best_rank[0]}/42 reliable in both raw scans."
            else:
                self.status_text = "Auto Calibrate Failed"
            self.autocalibrate_state = 0
            self.autocalibrate_mode = None
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
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/calibration.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
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
            "auto_exposure": 0,
            "color_exposure": self.color_exposure,
            "color_gain": self.color_gain,
            "color_auto_exposure": self.color_auto_exposure
        }
        try:
            filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/calibrate_realsense.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(settings, f, indent=4)
            self.status_text = "RealSense settings saved"
            return True
        except Exception as e:
            self.status_text = f"Save failed: {e}"
            return False

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
        "color_exposure": calibrator.color_exposure,
        "color_gain": calibrator.color_gain,
        "color_auto_exposure": calibrator.color_auto_exposure,
        "color_sensor_available": calibrator.color_sensor_available,
        "color_exposure_supported": calibrator.color_exposure_supported,
        "player1_color": calibrator.player1_color,
        "player2_color": calibrator.player2_color,
        "is_autocalibrating": calibrator.is_autocalibrating,
        "is_color_autocalibrating": calibrator.is_color_autocalibrating,
        "is_color_capturing": calibrator.is_color_capturing,
        "color_calibration_precision": calibrator.color_calibration_precision,
        "color_calibration_stage_index": calibrator.color_calibration_stage_index,
        "color_calibration_stage_rows": calibrator._current_colour_stage(),
        "color_calibration_stage_count": len(calibrator.COLOR_CALIBRATION_PLANS[calibrator.color_calibration_precision]["stages"]),
        "color_calibration_capture_progress": calibrator.color_calibration_capture_progress,
        "color_calibration_rgb_candidate_count": len(calibrator.color_calibration_rgb_candidates),
        "color_calibration_active_rgb_setting": calibrator.color_calibration_active_rgb_setting,
        "color_autocalibrate_progress": calibrator.color_autocalibrate_progress,
        "color_autocalibrate_result": calibrator.color_autocalibrate_result,
        "color_autocalibrate_results": calibrator.color_autocalibrate_results,
        "color_calibration_phase": calibrator.color_calibration_phase,
        "color_calibration_eta_seconds": round(calibrator.colour_calibration_eta_seconds(), 1),
        "color_calibration_capture_estimate_seconds": round(calibrator.color_calibration_capture_estimate_seconds, 1),
        "color_calibration_analysis_estimate_seconds": round(calibrator.color_calibration_analysis_estimate_seconds, 1),
        "autocalibrate_state": calibrator.autocalibrate_state,
        "autocalibrate_mode": calibrator.autocalibrate_mode,
        "autocalibrate_progress": calibrator.autocalibrate_progress,
        "autocalibrate_results": calibrator.autocalibrate_results,
        "ui_mode": calibrator.ui_mode,
        "nfc_connected": nfc_reader_connected(),
        "nfc_reader": reader_connection(),
        "nfc_last_tag": calibrator.nfc_last_tag
    })

def generate_frames(stream_type):
    last_frame = -1
    while True:
        try:
            if calibrator.frame_counter == last_frame:
                time.sleep(0.01)
                continue
                
            last_frame = calibrator.frame_counter

            if stream_type == 'color':
                img = calibrator.get_color_frame_image()
            else:
                img = calibrator.get_depth_frame_image()
                
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, f"No {stream_type} feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
            ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Generator error: {e}")
            time.sleep(0.5)

@app.route('/frame/color')
def frame_color():
    return Response(generate_frames('color'), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    if calibrator.autocalibrate_state in [1, 3, 4]:
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

@app.route('/api/set_ui_mode', methods=['POST'])
def set_ui_mode():
    data = request.json
    mode = data.get('ui_mode')
    if mode in ["define_board", "color_calibration", "detection_calibration", "realsense"]:
        calibrator.ui_mode = mode
    return jsonify({'status': 'UI mode updated'})

@app.route('/api/update_realsense', methods=['POST'])
def update_realsense():
    data = request.json
    for key, val in data.items():
        if hasattr(calibrator, key):
            setattr(calibrator, key, int(val))
    calibrator.apply_realsense_params()
    return jsonify({'status': 'RealSense parameters updated'})

@app.route('/api/autocalibrate/use_result', methods=['POST'])
def use_autocalibrate_result():
    data = request.get_json(silent=True) or {}
    index = data.get('index')
    if not isinstance(index, int) or not 0 <= index < len(calibrator.autocalibrate_results):
        return jsonify({'status': 'Invalid calibration result'}), 400
    result = calibrator.autocalibrate_results[index]
    suggested = result.get('suggested_occupancy_threshold')
    if data.get('apply_suggested_threshold') and suggested is None:
        return jsonify({'status': 'This result has no safe threshold suggestion'}), 400
    calibrator.exposure = int(result['exposure'])
    calibrator.gain = int(result['gain'])
    calibrator.laser_power = int(result['laser'])
    if data.get('apply_suggested_threshold'):
        calibrator.occupancy_threshold = float(suggested)
        if data.get('save_threshold_for_game') and not calibrator.save_detection_calibration():
            return jsonify({'status': calibrator.status_text}), 400
    calibrator.apply_realsense_params()
    return jsonify({'status': 'Result applied; the selected threshold was saved for the game.' if data.get('save_threshold_for_game') else 'Result applied temporarily; save the relevant configuration when ready.'})

@app.route('/api/autocalibrate_single', methods=['POST'])
def autocalibrate_single():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(1, 'single', params), daemon=True).start()
    return jsonify({"status": "started single"})

@app.route('/api/autocalibrate_filled', methods=['POST'])
def autocalibrate_filled():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(1, 'filled', params), daemon=True).start()
    return jsonify({"status": "started filled quick calibration"})

@app.route('/api/autocalibrate_partial', methods=['POST'])
def autocalibrate_partial():
    if calibrator.autocalibrate_state in [1, 3, 4]:
        return jsonify({"status": "already running"})
    params = request.get_json() if request.is_json else {}
    threading.Thread(target=calibrator._autocalibrate_thread, args=(1, 'partial', params), daemon=True).start()
    return jsonify({"status": "started partial quick calibration"})

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
    success = calibrator.save_realsense_calibration()
    if success:
        return jsonify({'status': calibrator.status_text})
    return jsonify({'status': 'failed', 'message': calibrator.status_text}), 400

@app.route('/api/calibrate_colors', methods=['POST'])
def calibrate_colors():
    calibrator.calibrate_colors()
    return jsonify({'status': calibrator.status_text})

@app.route('/api/color_calibration/start', methods=['POST'])
def start_colour_calibration():
    data = request.get_json(silent=True) or {}
    if not calibrator.start_colour_calibration(data.get("precision", "standard")):
        return jsonify({'status': calibrator.status_text}), 400
    return jsonify({'status': calibrator.status_text})

@app.route('/api/color_calibration/capture', methods=['POST'])
def capture_colour_calibration_stage():
    if not calibrator.capture_colour_stage():
        return jsonify({'status': calibrator.status_text}), 400
    return jsonify({'status': calibrator.status_text})

@app.route('/api/color_calibration/use_result', methods=['POST'])
def use_colour_calibration_result():
    data = request.get_json(silent=True) or {}
    if not calibrator.use_colour_calibration_result(data.get("index")):
        return jsonify({'status': calibrator.status_text}), 400
    return jsonify({'status': calibrator.status_text})

@app.route('/api/autocalibrate_colors', methods=['POST'])
def autocalibrate_colors():
    # Legacy endpoint for existing clients; new clients use the guided flow.
    if calibrator.is_color_autocalibrating or calibrator.is_color_capturing:
        return jsonify({'status': 'already running'}), 409
    if len(calibrator.corners) != 4:
        return jsonify({'status': 'Define all four board corners first.'}), 400
    return jsonify({'status': 'Use /api/color_calibration/start and /capture for guided RGB calibration.'}), 400

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

if __name__ == '__main__':
    if calibrator.start_webcam():
        try:
            app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
        finally:
            calibrator.stop_webcam()
    else:
        print("Failed to start camera. Exiting.")
