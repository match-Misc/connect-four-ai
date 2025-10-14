#!/usr/bin/env python3
"""
Connect Four Board Detection Script

This script performs computer vision detection of the current state
of a Connect Four board using a webcam and calibration data. It outputs
two bitboards representing the positions of Player 1 and Player 2 pieces.

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
import sys
import threading
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np


class ConnectFourDetector:
    # Tunable parameters for detection
    DETECTION_THRESHOLD = 55  # Color distance threshold for piece detection
    CONSISTENCY_WINDOW = 1.0  # Seconds for detection consistency check

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

        # Webcam and threading
        self.cap = None
        self.current_frame = None
        self.running = True
        self.frame_lock = threading.Lock()

        # GUI elements
        self.window_id = None
        self.texture_id = None
        self.bitboard_text_id = None

        # Detection results
        self.player1_bitboard = 0
        self.player2_bitboard = 0

        # Robustness improvements
        self.consistency_window = self.CONSISTENCY_WINDOW
        self.detection_history = []  # list of (timestamp, player1_mask, player2_mask)

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
            self.player1_color = np.array(self.calibration_data["player1_color"])
            self.player2_color = np.array(self.calibration_data["player2_color"])
            self.contrast = self.calibration_data.get("contrast", 100)
            self.saturation = self.calibration_data.get("saturation", 100)
            self.brightness = self.calibration_data.get("brightness", 0)

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
        """Start webcam capture in a separate thread"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False

        def capture_loop():
            frame_count = 0
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                else:
                    print("Failed to read frame")
                time.sleep(0.033)  # ~30 FPS

        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_webcam(self):
        """Stop webcam capture"""
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.cap:
            self.cap.release()

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

        player1_mask = 0
        player2_mask = 0

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
                    radius = max(3, self.hole_diameter // 4)
                    roi = adjusted_frame[
                        max(0, y - radius) : min(adjusted_frame.shape[0], y + radius),
                        max(0, x - radius) : min(adjusted_frame.shape[1], x + radius),
                    ]

                    if roi.size > 0:
                        avg_color = cv2.mean(roi)[:3]  # BGR values
                        avg_color = np.array(avg_color)

                        # Calculate color distances
                        dist_p1 = np.linalg.norm(avg_color - self.player1_color)
                        dist_p2 = np.linalg.norm(avg_color - self.player2_color)

                        # Classify piece
                        # Use a threshold to determine if it's a piece or empty
                        min_dist = min(dist_p1, dist_p2)
                        threshold = self.DETECTION_THRESHOLD

                        if min_dist < threshold:
                            # Calculate bit position
                            # Bitboard layout: bottom-left is bit 0, increases right then up
                            # row=0 is top, so bit_pos = (5 - row) * 7 + col
                            bit_pos = (5 - row) * 7 + col

                            if dist_p1 < dist_p2:
                                player1_mask |= 1 << bit_pos
                            else:
                                player2_mask |= 1 << bit_pos

        return player1_mask, player2_mask

    def enforce_gravity(self, player1_mask, player2_mask):
        """Enforce gravity: pieces can only be present if rows below are also present"""
        combined_mask = player1_mask | player2_mask
        enforced_p1 = 0
        enforced_p2 = 0

        for col in range(7):
            # Collect pieces in this column from bottom to top (row 0 is bottom)
            column_pieces = []
            for row in range(6):
                bit_pos = row * 7 + col
                if combined_mask & (1 << bit_pos):
                    column_pieces.append(bit_pos)

            # Sort by row (bottom to top)
            column_pieces.sort(key=lambda pos: pos // 7)

            # Enforce gravity: only keep pieces from the bottom up, no gaps
            valid_positions = []
            for i, pos in enumerate(column_pieces):
                expected_row = i  # 0 is bottom
                actual_row = pos // 7
                if actual_row == expected_row:
                    valid_positions.append(pos)
                else:
                    # Gap detected, stop here
                    break

            # Assign back to players
            for pos in valid_positions:
                if player1_mask & (1 << pos):
                    enforced_p1 |= 1 << pos
                elif player2_mask & (1 << pos):
                    enforced_p2 |= 1 << pos

        return enforced_p1, enforced_p2

    def check_consistency(self):
        """Check if recent detections within the time window are consistent"""
        if not self.detection_history:
            return False

        current_time = time.time()
        # Filter recent detections
        recent_detections = [
            (p1, p2)
            for ts, p1, p2 in self.detection_history
            if current_time - ts <= self.consistency_window
        ]

        if len(recent_detections) < 2:
            return False

        # Check if all recent detections match
        first_p1, first_p2 = recent_detections[0]
        for p1, p2 in recent_detections[1:]:
            if p1 != first_p1 or p2 != first_p2:
                return False

        return True

    def update_frame(self):
        """Update the displayed frame in the GUI"""
        with self.frame_lock:
            if self.current_frame is None:
                return

            frame = self.current_frame.copy()

        # Detect pieces
        raw_p1, raw_p2 = self.detect_pieces(frame)

        # Enforce gravity
        enforced_p1, enforced_p2 = self.enforce_gravity(raw_p1, raw_p2)

        # Add to history with timestamp
        current_time = time.time()
        self.detection_history.append((current_time, enforced_p1, enforced_p2))

        # Prune old entries outside the consistency window
        self.detection_history = [
            (ts, p1, p2)
            for ts, p1, p2 in self.detection_history
            if current_time - ts <= self.consistency_window
        ]

        # Check consistency and update bitboards only if consistent
        if self.check_consistency():
            self.player1_bitboard, self.player2_bitboard = enforced_p1, enforced_p2

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
                        if self.player1_bitboard & (1 << bit_pos):
                            cv2.circle(
                                frame,
                                (x, y),
                                self.hole_diameter // 2,
                                tuple(int(c) for c in self.player1_color),
                                2,
                            )  # Player 1 calibrated color
                        elif self.player2_bitboard & (1 << bit_pos):
                            cv2.circle(
                                frame,
                                (x, y),
                                self.hole_diameter // 2,
                                tuple(int(c) for c in self.player2_color),
                                2,
                            )  # Player 2 calibrated color
                        else:
                            cv2.circle(
                                frame, (x, y), self.hole_diameter // 2, (0, 255, 0), 1
                            )  # Green outline for empty

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
        dpg.create_viewport(title="Connect Four Detection", width=1000, height=700)

        with dpg.window(label="Detection", width=1000, height=700) as self.window_id:
            with dpg.group(horizontal=True):
                # Left side - Image display
                with dpg.child_window(width=700, height=600):
                    dpg.add_text("Webcam Feed - Real-time Detection")
                    with dpg.texture_registry():
                        self.texture_id = dpg.add_raw_texture(
                            640,
                            480,
                            np.zeros((640 * 480 * 3,), dtype=np.float32),
                            format=dpg.mvFormat_Float_rgb,
                        )
                    dpg.add_image(self.texture_id, width=640, height=480)

                # Right side - Bitboard display
                with dpg.child_window(width=280, height=600):
                    dpg.add_text("Detected Bitboards:")
                    self.bitboard_text_id = dpg.add_text(
                        "Player 1: 0000000000000000000000000000000000000000000000000\n"
                        "Player 2: 0000000000000000000000000000000000000000000000000",
                        wrap=270,
                    )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run_detection_gui(self):
        """Run detection with GUI"""
        if not self.load_calibration():
            return False

        if not self.start_webcam():
            return False

        self.create_gui()

        # Main loop
        while dpg.is_dearpygui_running():
            self.update_frame()
            dpg.render_dearpygui_frame()

        self.stop_webcam()
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
        # Enforce gravity for non-GUI mode as well
        player1_mask, player2_mask = self.enforce_gravity(player1_mask, player2_mask)
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
