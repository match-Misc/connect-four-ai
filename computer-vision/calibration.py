#!/usr/bin/env python3
"""
Connect Four Board Calibration Script

This script performs computer vision calibration for detecting the current state
of a Connect Four board using a webcam. It allows users to:
1. Define the board area using a rectangle (4 corners)
2. Align holes using sliders for diameter, horizontal and vertical spacing
3. Calibrate colors for both players by sampling from the first two columns
4. Export calibration data to JSON for use in detection scripts

Usage: python calibration.py
"""

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np


class ConnectFourCalibrator:
    def __init__(self):
        self.corners = []  # Will store 4 corner points
        self.hole_diameter = 30
        self.h_spacing = 60
        self.v_spacing = 60
        self.player1_color = None
        self.player2_color = None
        self.calibration_complete = False

        # Image adjustment parameters
        self.contrast = 100  # 0-200, 100=1.0
        self.saturation = 100  # 0-200, 100=1.0
        self.brightness = 0  # -100 to 100

        # Webcam and threading
        self.cap = None
        self.current_frame = None
        self.running = True
        self.frame_lock = threading.Lock()

        # GUI elements
        self.window_id = None
        self.texture_id = None
        self.status_text = ""

    def start_webcam(self):
        """Start webcam capture in a separate thread"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_text = "Error: Could not open webcam"
            return False

        def capture_loop():
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
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

    def mouse_callback(self, sender, app_data, user_data):
        """Handle mouse clicks for defining board corners"""
        # Get mouse position relative to the viewport
        mouse_x, mouse_y = dpg.get_mouse_pos()

        # Get the position of the image widget
        image_pos = dpg.get_item_pos(self.image_id)

        # Calculate relative position within the image
        rel_x = mouse_x - image_pos[0]
        rel_y = mouse_y - image_pos[1]

        # Debug: print the coordinates
        print(
            f"Mouse click at viewport: ({mouse_x}, {mouse_y}), image pos: ({image_pos[0]}, {image_pos[1]}), relative: ({rel_x}, {rel_y})"
        )

        # Check if click is within image bounds (640x480)
        if 0 <= rel_x < 640 and 0 <= rel_y < 480:
            if len(self.corners) < 4:
                # Scale coordinates to match actual image size if needed
                # For now, assume 1:1 mapping
                self.corners.append((int(rel_x), int(rel_y)))
                self.status_text = (
                    f"Corner {len(self.corners)} set at ({int(rel_x)}, {int(rel_y)})"
                )
                print(f"Added corner {len(self.corners)}: {self.corners[-1]}")
                if len(self.corners) == 4:
                    self.status_text = (
                        "All corners defined. Adjust hole parameters with sliders."
                    )

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

    def draw_corners(self, frame):
        """Draw the defined corners on the frame"""
        for i, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"{i + 1}",
                (corner[0] + 10, corner[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def draw_hole_grid(self, frame):
        """Draw the hole grid based on current parameters"""
        if len(self.corners) == 4:
            # Apply image adjustments for visual feedback
            adjusted_frame = self.adjust_image(frame)

            # Sort corners to get top-left, top-right, bottom-left, bottom-right
            corners = np.array(self.corners)

            # Sort corners by y-coordinate first (top vs bottom), then by x-coordinate (left vs right)
            corners = corners[np.lexsort((corners[:, 0], corners[:, 1]))]
            # corners[0] = top-left, corners[1] = top-right, corners[2] = bottom-left, corners[3] = bottom-right

            # Calculate perspective transform to map grid to corners
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

            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)

            # Draw grid of circles using perspective transform on adjusted frame
            for row in range(6):  # Connect Four has 6 rows
                for col in range(7):  # Connect Four has 7 columns
                    # Calculate grid position
                    grid_x = col * self.h_spacing
                    grid_y = row * self.v_spacing

                    # Transform to image coordinates
                    grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(
                        grid_point.reshape(1, 1, 2), np.linalg.inv(M)
                    )

                    x, y = transformed[0, 0].astype(int)

                    if (
                        0 <= x < adjusted_frame.shape[1]
                        and 0 <= y < adjusted_frame.shape[0]
                    ):
                        cv2.circle(
                            adjusted_frame,
                            (x, y),
                            self.hole_diameter // 2,
                            (255, 0, 0),
                            2,
                        )

            return adjusted_frame
        return frame

    def calibrate_colors(self):
        """Calibrate player colors from the first two columns"""
        if len(self.corners) != 4 or self.current_frame is None:
            return False

        # Apply image adjustments before calibration
        adjusted_frame = self.adjust_image(self.current_frame)

        # Sort corners properly
        corners = np.array(self.corners)
        corners = corners[np.lexsort((corners[:, 0], corners[:, 1]))]

        # Calculate perspective transform
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

        player1_samples = []
        player2_samples = []

        # Sample from first column (Player 1) and second column (Player 2)
        for col in range(2):  # First two columns
            for row in range(6):
                # Calculate grid position
                grid_x = col * self.h_spacing
                grid_y = row * self.v_spacing

                # Transform to image coordinates
                grid_point = np.array([[grid_x, grid_y]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(
                    grid_point.reshape(1, 1, 2), np.linalg.inv(M)
                )

                x, y = transformed[0, 0].astype(int)

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
                        if col == 0:
                            player1_samples.append(avg_color)
                        else:
                            player2_samples.append(avg_color)

        if player1_samples and player2_samples:
            # Average the samples
            self.player1_color = np.mean(player1_samples, axis=0).astype(int).tolist()
            self.player2_color = np.mean(player2_samples, axis=0).astype(int).tolist()
            self.status_text = f"Player 1 color: {self.player1_color}, Player 2 color: {self.player2_color}"
            return True

        return False

    def save_calibration(self, filename="calibration.json"):
        """Save calibration data to JSON file"""
        if not self.calibration_complete:
            self.status_text = (
                "Calibration not complete. Please calibrate colors first."
            )
            return False

        data = {
            "corners": self.corners,
            "hole_diameter": self.hole_diameter,
            "horizontal_spacing": self.h_spacing,
            "vertical_spacing": self.v_spacing,
            "player1_color": self.player1_color,
            "player2_color": self.player2_color,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "brightness": self.brightness,
        }

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            self.status_text = f"Calibration saved to {filename}"
            return True
        except Exception as e:
            self.status_text = f"Error saving calibration: {e}"
            return False

    def update_frame(self):
        """Update the displayed frame in the GUI"""
        with self.frame_lock:
            if self.current_frame is None:
                return

            frame = self.current_frame.copy()

        # Draw corners and grid
        self.draw_corners(frame)
        if len(self.corners) == 4:
            frame = self.draw_hole_grid(frame)

        # Convert to RGB for Dear PyGui and normalize to 0-1 range
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Update texture
        if self.texture_id:
            dpg.set_value(self.texture_id, frame_rgb.flatten())

    def create_gui(self):
        """Create the Dear PyGui interface"""
        dpg.create_context()
        dpg.create_viewport(title="Connect Four Calibration", width=1200, height=800)

        with dpg.window(label="Calibration", width=1200, height=800) as self.window_id:
            with dpg.group(horizontal=True):
                # Left side - Image display
                with dpg.child_window(width=800, height=600):
                    dpg.add_text("Webcam Feed - Click to define corners")
                    with dpg.texture_registry():
                        # Create a placeholder texture (will be updated with actual frame)
                        self.texture_id = dpg.add_raw_texture(
                            640,
                            480,
                            np.zeros((640 * 480 * 3,), dtype=np.float32),
                            format=dpg.mvFormat_Float_rgb,
                        )
                    self.image_id = dpg.add_image(
                        self.texture_id, width=640, height=480
                    )
                    # Use mouse handlers instead of item handlers for better coordinate handling
                    with dpg.handler_registry():
                        dpg.add_mouse_click_handler(
                            button=dpg.mvMouseButton_Left, callback=self.mouse_callback
                        )
                        dpg.add_mouse_click_handler(
                            button=dpg.mvMouseButton_Right,
                            callback=self.reset_corners_callback,
                        )

                # Right side - Controls
                with dpg.child_window(width=380, height=600):
                    dpg.add_text("Hole Parameters")
                    dpg.add_slider_int(
                        label="Diameter",
                        default_value=self.hole_diameter,
                        min_value=10,
                        max_value=100,
                        callback=lambda s, a: setattr(self, "hole_diameter", a),
                    )

                    dpg.add_separator()
                    dpg.add_text("Image Adjustments")
                    dpg.add_slider_int(
                        label="Contrast",
                        default_value=self.contrast,
                        min_value=0,
                        max_value=200,
                        callback=lambda s, a: setattr(self, "contrast", a),
                    )
                    dpg.add_slider_int(
                        label="Saturation",
                        default_value=self.saturation,
                        min_value=0,
                        max_value=200,
                        callback=lambda s, a: setattr(self, "saturation", a),
                    )
                    dpg.add_slider_int(
                        label="Brightness",
                        default_value=self.brightness + 100,
                        min_value=0,
                        max_value=200,
                        callback=lambda s, a: setattr(self, "brightness", a - 100),
                    )

                    dpg.add_separator()
                    dpg.add_button(
                        label="Calibrate Colors",
                        callback=self.calibrate_colors_callback,
                    )
                    dpg.add_button(
                        label="Save Calibration",
                        callback=self.save_calibration_callback,
                    )
                    dpg.add_button(
                        label="Reset Corners", callback=self.reset_corners_callback
                    )

                    dpg.add_separator()
                    dpg.add_text("Status:")
                    self.status_text_id = dpg.add_text(self.status_text, wrap=350)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def calibrate_colors_callback(self, sender, app_data, user_data):
        """Callback for calibrate colors button"""
        if len(self.corners) == 4:
            if self.calibrate_colors():
                self.calibration_complete = True
                self.status_text = "Color calibration complete!"
            else:
                self.status_text = "Color calibration failed. Check board setup."
        else:
            self.status_text = "Please define all 4 corners first."

    def save_calibration_callback(self, sender, app_data, user_data):
        """Callback for save calibration button"""
        self.save_calibration()

    def reset_corners_callback(self, sender, app_data, user_data):
        """Callback for reset corners button"""
        self.corners = []
        self.calibration_complete = False
        self.status_text = "Corners reset. Click to define 4 corners."

    def run_calibration(self):
        """Main calibration loop"""
        if not self.start_webcam():
            return False

        self.create_gui()

        # Main loop
        while dpg.is_dearpygui_running():
            self.update_frame()
            dpg.set_value(self.status_text_id, self.status_text)
            dpg.render_dearpygui_frame()

        self.stop_webcam()
        dpg.destroy_context()
        return self.calibration_complete


def main():
    calibrator = ConnectFourCalibrator()
    success = calibrator.run_calibration()

    if success:
        print("Calibration completed successfully!")
        return 0
    else:
        print("Calibration was not completed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
