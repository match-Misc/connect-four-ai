#!/usr/bin/env python3
"""
Connect Four Board Detection Script

This script performs computer vision detection of the current state
of a Connect Four board using an Intel RealSense color stream and calibration data.
It outputs two bitboards representing the positions of Player 1 and Player 2 pieces.

Usage: python detection.py [--calibration CALIB_FILE]

The bitboard representation uses 49 bits for the 6x7 grid:
    6 13 20 27 34 41 48
   ---------------------
  | 5 12 19 26 33 40 47 |
  | 4 11 18 25 32 39 46 |
  | 3 10 17 24 31 38 45 |
  | 2  9 16 23 30 37 44 |
  | 1  8 15 22 29 36 43 |
  | 0  7 14 21 28 35 42 |
   ---------------------

Each bit position corresponds to a hole on the board, with bit 0 at bottom-left.
"""

import argparse
import json
import socket   
import sys
import threading
import time

import pyrealsense2 as rs
import cv2
import dearpygui.dearpygui as dpg
import numpy as np


class ConnectFourDetector:
    # Tunable parameters for detection
    DETECTION_THRESHOLD = 80  # Threshold on G channel in RGB (0-255). Below: black, above: green.
    CONSISTENCY_WINDOW = 2.5  # Seconds for detection consistency check
    DEPTH_TOLERANCE_M = 0.01  # +/- 1 cm tolerance for depth verification

    def __init__(self, calibration_file="calibration.json"):
        self.calibration_file = calibration_file
        self.calibration_data = None
        self.corners = None
        self.hole_diameter = None
        self.h_spacing = None
        self.v_spacing = None
        self.player1_color = None
        self.player2_color = None
        self.contrast = None
        self.saturation = None
        self.brightness = None

        # RealSense and threading
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        self.cap = None  # retained for compatibility, unused
        self.current_frame = None
        self.current_depth_m = None  # aligned to color, meters
        self.running = True
        self.frame_lock = threading.Lock()
        self.frame_width = 640  # Default, will be updated
        self.frame_height = 480  # Default, will be updated

        # GUI elements
        self.window_id = None
        self.texture_id = None
        self.bitboard_text_id = None

        # Detection results
        self.player1_bitboard = 0
        self.player2_bitboard = 0

        # Socket server for exposing bitmasks
        self.socket_server = None
        self.socket_thread = None
        self.socket_port = 65432  # Default port

        # Robustness improvements
        self.consistency_window = self.CONSISTENCY_WINDOW
        self.last_observed_state = [-1] * (6 * 7)  # -1 = uninitialized, 0 = empty, 1/2 = players
        self.last_observed_since = [0.0] * (6 * 7)
        self.validated_state = [0] * (6 * 7)

        # Dynamic detection threshold & last sampled G values for overlay/prints
        self.detection_threshold = self.DETECTION_THRESHOLD
        self.last_g_values = [[0.0 for _ in range(7)] for _ in range(6)]  # row-major top (0) to bottom (5)
        self.last_depth_values = [[None for _ in range(7)] for _ in range(6)]
        self.calib_depth_m = None  # 6x7 matrix from calibration
        
        # RGB max value thresholds for detection
        self.max_r_value = 250.0
        self.max_b_value = 250.0
        self.max_g_value = 200.0
        
        # RealSense camera settings
        self.realsense_settings = None

    @staticmethod
    def circular_roi_pixels(frame, cx, cy, radius):
        """Return pixels from the circular hole area, excluding square corners."""
        radius = max(1, int(radius))
        y_min, y_max = max(0, cy - radius), min(frame.shape[0], cy + radius + 1)
        x_min, x_max = max(0, cx - radius), min(frame.shape[1], cx + radius + 1)
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return np.array([], dtype=frame.dtype)
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        return roi[mask]

    def load_calibration(self):
        """Load calibration data from JSON file"""
        try:
            with open(self.calibration_file, "r") as f:
                self.calibration_data = json.load(f)

            # Load corners in the correct order: top_left, top_right, bottom_left, bottom_right
            corners_dict = self.calibration_data["corners"]
            self.corners = [
                corners_dict["top_left"],
                corners_dict["top_right"],
                corners_dict["bottom_left"],
                corners_dict["bottom_right"],
            ]
            self.hole_diameter = self.calibration_data["hole_diameter"]
            self.h_spacing = self.calibration_data["horizontal_spacing"]
            self.v_spacing = self.calibration_data["vertical_spacing"]
            self.player1_color = np.array(self.calibration_data["player1_color"]) if "player1_color" in self.calibration_data else None
            self.player2_color = np.array(self.calibration_data["player2_color"]) if "player2_color" in self.calibration_data else None
            self.contrast = self.calibration_data.get("contrast", 100)
            self.saturation = self.calibration_data.get("saturation", 100)
            self.brightness = self.calibration_data.get("brightness", 0)
            # Load calibration depth map if present
            if "depth_m" in self.calibration_data:
                self.calib_depth_m = self.calibration_data["depth_m"]

            # Load RealSense camera settings if available
            try:
                with open("realsense_settings.json", "r") as f:
                    self.realsense_settings = json.load(f)
                    print(f"Loaded RealSense settings from realsense_settings.json")
            except FileNotFoundError:
                print("No realsense_settings.json found, using default RealSense settings")
            except json.JSONDecodeError:
                print("Invalid JSON in realsense_settings.json, using default RealSense settings")

            self.detection_threshold = (self.player1_color[1]+self.player2_color[1])/3

            return True
        except FileNotFoundError:
            print(f"Error: Calibration file '{self.calibration_file}' not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in calibration file '{self.calibration_file}'")
            return False
        except KeyError as e:
            print(f"Error: Missing required calibration data: {e}")
            return False

    def start_webcam(self):
        """Start RealSense color stream capture in a separate thread (name kept for compatibility)."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            # Enable depth+color, align depth to color for per-pixel depth
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)
            # Get actual stream dimensions
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.frame_width = color_profile.width()
            self.frame_height = color_profile.height()
            print(f"RealSense resolution: {self.frame_width}x{self.frame_height}")
            # Depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            
            # Apply RealSense settings if loaded
            if self.realsense_settings:
                try:
                    if "auto_exposure" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.enable_auto_exposure, float(self.realsense_settings["auto_exposure"]))
                    if "visual_preset" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.visual_preset, float(self.realsense_settings["visual_preset"]))
                    if "exposure" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.exposure, float(self.realsense_settings["exposure"]))
                    if "gain" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.gain, float(self.realsense_settings["gain"]))
                    if "laser_power" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.laser_power, float(self.realsense_settings["laser_power"]))
                    if "emitter_enabled" in self.realsense_settings:
                        depth_sensor.set_option(rs.option.emitter_enabled, float(self.realsense_settings["emitter_enabled"]))
                    print("Applied RealSense settings from realsense_settings.json")
                except Exception as e:
                    print(f"Warning: Could not apply some RealSense settings: {e}")
            
            # Align depth to color
            self.align = rs.align(rs.stream.color)
        except Exception as e:
            print(f"Error: Could not start RealSense pipeline ({e})")
            return False

        def capture_loop():
            while self.running:
                try:
                    frames = self.pipeline.wait_for_frames()
                    if self.align is not None:
                        frames = self.align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
                    depth_m = (
                        depth_raw.astype(np.float32) * self.depth_scale
                        if self.depth_scale is not None
                        else depth_raw.astype(np.float32)
                    )
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.current_depth_m = depth_m
                except Exception:
                    pass
                time.sleep(0.001)

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_webcam(self):
        """Stop RealSense capture"""
        self.running = False
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass

    def start_socket_server(self):
        """Start socket server to expose bitmasks"""
        self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket_server.bind(("localhost", self.socket_port))
            self.socket_server.listen(1)
            print(f"Socket server listening on port {self.socket_port}")
        except OSError as e:
            print(f"Failed to start socket server: {e}")
            return False

        def server_loop():
            while self.running:
                try:
                    conn, addr = self.socket_server.accept()
                    with conn:
                        while self.running:
                            data = conn.recv(1024)
                            if not data:
                                break
                            # Send current bitmasks as JSON
                            response = json.dumps(
                                {
                                    "player1": self.player1_bitboard,
                                    "player2": self.player2_bitboard,
                                }
                            ).encode("utf-8")
                            conn.sendall(response)
                except OSError:
                    break  # Socket closed
            self.socket_server.close()

        self.socket_thread = threading.Thread(target=server_loop, daemon=True)
        self.socket_thread.start()
        return True

    def stop_socket_server(self):
        """Stop socket server"""
        if self.socket_server:
            self.socket_server.close()
        if self.socket_thread and self.socket_thread.is_alive():
            self.socket_thread.join()

    def adjust_image(self, frame):
        """Apply contrast, saturation, and brightness adjustments to the frame"""
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (self.saturation / 100.0), 0, 255)

        # Convert back to BGR
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Adjust contrast and brightness
        alpha = self.contrast / 100.0  # Contrast control (1.0-3.0)
        beta = self.brightness  # Brightness control (-100 to 100)

        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return frame

    def format_bitboard(self, bitboard):
        """Format bitboard as a visual grid string matching the bitboard representation"""
        lines = []
        # Top row (row 5, top of board)
        top_line = "  " + " ".join(
            str(1 if bitboard & (1 << (5 * 7 + col)) else 0) for col in range(7)
        )
        lines.append(top_line)
        lines.append("  ---------------------")
        # Rows 4 down to 0
        for row in range(4, -1, -1):
            row_line = (
                " | "
                + " ".join(
                    str(1 if bitboard & (1 << (row * 7 + col)) else 0)
                    for col in range(7)
                )
                + " |"
            )
            lines.append(row_line)
        lines.append("  ---------------------")
        return "\n".join(lines)

    def detect_pieces(self, frame):
        """Detect pieces on the board and return bitboards"""
        if not self.calibration_data:
            return 0, 0

        # Apply image adjustments
        adjusted_frame = self.adjust_image(frame)

        # Corners are already in correct order: top_left, top_right, bottom_left, bottom_right
        corners = np.array(self.corners)

        # Define destination points (grid coordinates)
        dst_points = np.array(
            [
                [0, 0],  # top-left
                [6 * self.h_spacing, 0],  # top-right
                [0, 5 * self.v_spacing],  # bottom-left
                [6 * self.h_spacing, 5 * self.v_spacing],  # bottom-right
            ],
            dtype=np.float32,
        )

        src_points = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        player1_mask = 0  # will represent Black
        player2_mask = 0  # will represent Green
        # Reset last values
        self.last_g_values = [[0.0 for _ in range(7)] for _ in range(6)]
        self.last_depth_values = [[None for _ in range(7)] for _ in range(6)]
        # Sample each hole position
        for row in range(6):  # 6 rows
            for col in range(7):  # 7 columns
                # Calculate grid position (hole center)
                grid_x = col * self.h_spacing
                grid_y = row * self.v_spacing

                # Transform to image coordinates
                grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(
                    grid_point.reshape(1, 1, 2), np.linalg.inv(M)
                )

                x, y = transformed[0, 0].astype(int)

                # Check bounds
                if (
                    0 <= x < adjusted_frame.shape[1]
                    and 0 <= y < adjusted_frame.shape[0]
                ):
                    # Sample a region around the hole center
                    radius = int(self.hole_diameter / 2)
                    roi_pixels = self.circular_roi_pixels(adjusted_frame, x, y, radius)

                    if roi_pixels.size > 0:
                        # Compute average colour from the same circular hole area.
                        avg_bgr = np.mean(roi_pixels, axis=0)[:3]
                        avg_g = float(avg_bgr[1])
                        self.last_g_values[row][col] = avg_g

                        # Depth gating: require measured depth within +/- tolerance of calibration depth
                        
                        depth_ok = False
                        measured_depth = None
                        if self.current_depth_m is not None:
                            # Sample depth using the identical circular hole area.
                            radius = int(self.hole_diameter / 2)
                            depth_pixels = self.circular_roi_pixels(self.current_depth_m, x, y, radius)
                            # Reject only if more than 5% of circular ROI pixels are missing depth.
                            if depth_pixels.size > 0:
                                valid_pixels = depth_pixels[depth_pixels > 0]
                                valid_ratio = valid_pixels.size / depth_pixels.size
                                if valid_ratio >= 0.95:
                                    measured_depth = float(np.mean(valid_pixels))
                                else:
                                    # More than 5% missing depth
                                    measured_depth = None
                        self.last_depth_values[row][col] = measured_depth

                        if (
                            measured_depth is not None
                            and self.calib_depth_m is not None
                        ):
                            calib_d = self.calib_depth_m[row][col]
                            if calib_d is not None:
                                if abs(measured_depth - float(calib_d)) <= self.DEPTH_TOLERANCE_M:
                                    depth_ok = True
                        # If no calibration depth or no measured depth, treat as not ok (cannot be successful)
                        
                        # Classify only if depth check passed
                        if depth_ok and float(avg_bgr[2]) < self.max_r_value and float(avg_bgr[0]) < self.max_b_value and float(avg_bgr[1]) < self.max_g_value:
                            threshold = self.detection_threshold
                            bit_pos = (5 - row) * 7 + col  # bottom-left is bit 0
                            if avg_g <= threshold:
                                player1_mask |= 1 << bit_pos  # Black -> player1
                            else:
                                player2_mask |= 1 << bit_pos  # Green -> player2


        return player1_mask, player2_mask

    def update_stable_board(self, player1_mask, player2_mask, current_time):
        """Update validated board state based on stability over the consistency window."""
        updated = False

        for pos in range(6 * 7):
            bit = 1 << pos
            if player1_mask & bit:
                current_status = 1
            elif player2_mask & bit:
                current_status = 2
            else:
                current_status = 0

            if self.last_observed_state[pos] != current_status:
                self.last_observed_state[pos] = current_status
                self.last_observed_since[pos] = current_time
            elif self.last_observed_since[pos] == 0.0:
                self.last_observed_since[pos] = current_time

            stable_duration = current_time - self.last_observed_since[pos]

            if (
                stable_duration >= self.consistency_window
                and self.validated_state[pos] != current_status
            ):
                self.validated_state[pos] = current_status
                updated = True

        stable_p1 = 0
        stable_p2 = 0
        for pos, status in enumerate(self.validated_state):
            if status == 1:
                stable_p1 |= 1 << pos
            elif status == 2:
                stable_p2 |= 1 << pos

        return stable_p1, stable_p2, updated

    def update_frame(self):
        """Update the displayed frame in the GUI"""

        with self.frame_lock:
            if self.current_frame is None:
                return

            frame = self.current_frame.copy()

        # Detect pieces
        player1_mask, player2_mask = self.detect_pieces(frame)

        current_time = time.time()
        stable_p1, stable_p2, updated = self.update_stable_board(
            player1_mask, player2_mask, current_time
        )

        if updated:
            self.player1_bitboard, self.player2_bitboard = stable_p1, stable_p2
            # Print G values and Depth grids when detection stabilizes
            try:
                g_rows = [
                    " ".join(f"{int(self.last_g_values[r][c]):3d}" for c in range(7))
                    for r in range(6)
                ]
                d_rows = [
                    " ".join(
                        (
                            f"{self.last_depth_values[r][c]:.2f}" if self.last_depth_values[r][c] is not None else "  --  "
                        )
                        for c in range(7)
                    )
                    for r in range(6)
                ]
                print("G values (row 0=top):\n" + "\n".join(g_rows))
                print("Depth (m, row 0=top):\n" + "\n".join(d_rows))
            except Exception:
                pass

        # Apply image adjustments for display
        frame = self.adjust_image(frame)

        # Draw detected pieces on frame
        if self.calibration_data:
            # Corners are already in correct order: top_left, top_right, bottom_left, bottom_right
            corners = np.array(self.corners)

            dst_points = np.array(
                [
                    [0, 0],
                    [6 * self.h_spacing, 0],
                    [0, 5 * self.v_spacing],
                    [6 * self.h_spacing, 5 * self.v_spacing],
                ],
                dtype=np.float32,
            )

            src_points = corners.astype(np.float32)
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            for row in range(6):
                for col in range(7):
                    bit_pos = (5 - row) * 7 + col
                    grid_x = col * self.h_spacing
                    grid_y = row * self.v_spacing

                    grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(
                        grid_point.reshape(1, 1, 2), np.linalg.inv(M)
                    )

                    x, y = transformed[0, 0].astype(int)

                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        # Draw immediate per-frame classification to keep UI responsive
                        if player1_mask & (1 << bit_pos):
                            color = (0, 0, 0)  # black
                        elif player2_mask & (1 << bit_pos):
                            color = (0, 255, 0)  # green
                        else:
                            color = (0, 255, 255)  # empty/unknown this frame
                        cv2.circle(
                            frame,
                            (x, y),
                            self.hole_diameter // 2,
                            color,
                            2,
                        )
                        g_val = self.last_g_values[row][col]
                        cv2.putText(
                            frame,
                            f"G:{g_val:.0f}",
                            (x - 12, y + self.hole_diameter // 2 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                        d_val = self.last_depth_values[row][col]
                        if d_val is not None:
                            depth_txt = f"D:{d_val:.2f}m"
                        else:
                            depth_txt = "D:--"
                        cv2.putText(
                            frame,
                            depth_txt,
                            (x - 18, y + self.hole_diameter // 2 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (180, 180, 180),
                            1,
                        )

        # Convert to RGB for Dear PyGui and normalize to 0-1 range
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Update texture
        if self.texture_id:
            dpg.set_value(self.texture_id, frame_rgb.flatten())

        # Update bitboard display
        if self.bitboard_text_id:
            mask = self.player1_bitboard | self.player2_bitboard
            position = self.player1_bitboard  # Assuming Player 1 is current player
            mask_visual = self.format_bitboard(mask)
            position_visual = self.format_bitboard(position)
            dpg.set_value(
                self.bitboard_text_id,
                f"Mask: {mask}\n{mask_visual}\n\nPosition: {position}\n{position_visual}",
            )

        

    def create_gui(self):
        """Create the Dear PyGui interface"""
        dpg.create_context()
        # Adjust viewport size based on frame dimensions
        viewport_width = self.frame_width + 300  # Extra space for controls
        viewport_height = max(self.frame_height + 40, 600)  # At least 600 for controls
        dpg.create_viewport(title="Connect Four Detection", width=viewport_width, height=viewport_height)

        with dpg.window(label="Detection", width=viewport_width, height=viewport_height) as self.window_id:
            with dpg.group(horizontal=True):
                # Left side - Image display
                with dpg.child_window(width=self.frame_width, height=self.frame_height + 20):
                    dpg.add_text("RealSense Feed - Real-time Detection")
                    with dpg.texture_registry():
                        self.texture_id = dpg.add_raw_texture(
                            self.frame_width,
                            self.frame_height,
                            np.zeros((self.frame_width * self.frame_height * 3,), dtype=np.float32),
                            format=dpg.mvFormat_Float_rgb,
                        )
                    dpg.add_image(self.texture_id, width=self.frame_width, height=self.frame_height)

                # Right side - Bitboard display
                with dpg.child_window(width=280, height=600):
                    dpg.add_text("Detected Bitboards:")
                    self.bitboard_text_id = dpg.add_text(
                        "Player 1: 0000000000000000000000000000000000000000000000000\n"
                        "Player 2: 0000000000000000000000000000000000000000000000000",
                        wrap=270,
                    )
                    dpg.add_separator()
                    dpg.add_text("Detection Threshold")
                    dpg.add_slider_float(
                        label="Threshold",
                        default_value=self.detection_threshold,
                        min_value=5.0,
                        max_value=100.0,
                        callback=lambda s, a: setattr(self, "detection_threshold", a),
                    )
                    dpg.add_separator()
                    dpg.add_text("RGB Max Values (above = invalid)")
                    dpg.add_slider_float(
                        label="Max R",
                        default_value=self.max_r_value,
                        min_value=0.0,
                        max_value=255.0,
                        callback=lambda s, a: setattr(self, "max_r_value", a),
                    )
                    dpg.add_slider_float(
                        label="Max G",
                        default_value=self.max_g_value,
                        min_value=0.0,
                        max_value=255.0,
                        callback=lambda s, a: setattr(self, "max_g_value", a),
                    )
                    dpg.add_slider_float(
                        label="Max B",
                        default_value=self.max_b_value,
                        min_value=0.0,
                        max_value=255.0,
                        callback=lambda s, a: setattr(self, "max_b_value", a),
                    )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run_detection_gui(self):
        """Run detection with GUI"""
        if not self.load_calibration():
            return False

        if not self.start_webcam():
            return False

        if not self.start_socket_server():
            self.stop_webcam()
            return False

        self.create_gui()

        # Main loop
        while dpg.is_dearpygui_running():
            self.update_frame()
            dpg.render_dearpygui_frame()
            

        self.stop_webcam()
        self.stop_socket_server()
        dpg.destroy_context()
        return True

    def get_bitboards(self):
        """Get current bitboards without GUI"""
        if not self.load_calibration():
            return None, None

        if not self.start_webcam():
            return None, None

        # Wait for camera to stabilize and capture initial frame
        timeout = 5.0  # seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.frame_lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
                    break
            time.sleep(0.1)
        else:
            print("Timeout: No frame captured within 5 seconds")
            self.stop_webcam()
            return None, None

        player1_mask, player2_mask = self.detect_pieces(frame)
        self.stop_webcam()

        return player1_mask, player2_mask


def main():
    parser = argparse.ArgumentParser(description="Connect Four Board Detection")
    parser.add_argument(
        "--calibration",
        default="calibration.json",
        help="Path to calibration JSON file (default: calibration.json)",
    )

    args = parser.parse_args()

    detector = ConnectFourDetector(args.calibration)

    success = detector.run_detection_gui()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
