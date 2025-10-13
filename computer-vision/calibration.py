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
from pathlib import Path

import cv2
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

        # Window names
        self.main_window = "Connect Four Calibration"
        self.trackbar_window = "Hole Alignment"

        # Mouse callback variables
        self.drawing = False
        self.current_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for defining board corners"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
                print(f"Corner {len(self.corners)} set at ({x}, {y})")
                if len(self.corners) == 4:
                    print("All corners defined. Adjust hole parameters with sliders.")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Reset corners
            self.corners = []
            print("Corners reset. Click to define 4 corners.")

    def create_trackbars(self):
        """Create OpenCV trackbars for hole alignment and image adjustments"""
        cv2.namedWindow(self.trackbar_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.trackbar_window, 600, 200)
        cv2.createTrackbar(
            "Diameter", self.trackbar_window, self.hole_diameter, 100, lambda x: None
        )
        cv2.createTrackbar(
            "H Spacing", self.trackbar_window, self.h_spacing, 150, lambda x: None
        )
        cv2.createTrackbar(
            "V Spacing", self.trackbar_window, self.v_spacing, 150, lambda x: None
        )
        cv2.createTrackbar(
            "Contrast", self.trackbar_window, self.contrast, 200, lambda x: None
        )
        cv2.createTrackbar(
            "Saturation", self.trackbar_window, self.saturation, 200, lambda x: None
        )
        cv2.createTrackbar(
            "Brightness",
            self.trackbar_window,
            self.brightness + 100,
            200,
            lambda x: None,
        )

    def get_trackbar_values(self):
        """Get current trackbar values"""
        try:
            self.hole_diameter = cv2.getTrackbarPos("Diameter", self.trackbar_window)
            self.h_spacing = cv2.getTrackbarPos("H Spacing", self.trackbar_window)
            self.v_spacing = cv2.getTrackbarPos("V Spacing", self.trackbar_window)
            self.contrast = cv2.getTrackbarPos("Contrast", self.trackbar_window)
            self.saturation = cv2.getTrackbarPos("Saturation", self.trackbar_window)
            self.brightness = (
                cv2.getTrackbarPos("Brightness", self.trackbar_window) - 100
            )
        except cv2.error:
            # Trackbars not created yet or window closed
            pass

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

    def calibrate_colors(self, frame):
        """Calibrate player colors from the first two columns"""
        if len(self.corners) != 4:
            return False

        # Apply image adjustments before calibration
        adjusted_frame = self.adjust_image(frame)

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
            print(f"Player 1 color calibrated: {self.player1_color}")
            print(f"Player 2 color calibrated: {self.player2_color}")
            return True

        return False

    def save_calibration(self, filename="calibration.json"):
        """Save calibration data to JSON file"""
        if not self.calibration_complete:
            print("Calibration not complete. Please calibrate colors first.")
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
            print(f"Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def run_calibration(self):
        """Main calibration loop"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False

        # Set up windows and callbacks
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.main_window, 800, 600)
        cv2.setMouseCallback(self.main_window, self.mouse_callback)
        self.create_trackbars()

        print("Connect Four Board Calibration")
        print("Instructions:")
        print("1. Click 4 corners of the board (in any order)")
        print(
            "2. Adjust hole diameter, spacing, contrast, saturation, and brightness with sliders in the 'Hole Alignment' window"
        )
        print("3. Press 'c' to calibrate colors from first two columns")
        print("4. Press 's' to save calibration to JSON")
        print("5. Press 'r' to reset corners")
        print("6. Press 'q' to quit")
        print()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.current_frame = frame.copy()

            # Draw corners and grid
            self.draw_corners(frame)
            if len(self.corners) == 4:
                self.get_trackbar_values()
                adjusted_frame = self.draw_hole_grid(frame)
                # Use adjusted frame for display if adjustments are applied
                if (
                    self.contrast != 100
                    or self.saturation != 100
                    or self.brightness != 0
                ):
                    frame = adjusted_frame

            # Display instructions
            cv2.putText(
                frame,
                "Corners: Click to define 4 corners",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if len(self.corners) == 4:
                cv2.putText(
                    frame,
                    "Press 'c' to calibrate colors",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 's' to save calibration",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(self.main_window, frame)

            key = cv2.waitKey(1) & 0xFF

            if key != 255:  # Only process if a key was actually pressed
                print(f"Key pressed: {key} (char: {chr(key) if key > 0 else 'none'})")

            if key == ord("q"):
                break
            elif key == ord("c") and len(self.corners) == 4:
                print("Attempting color calibration...")
                if self.calibrate_colors(self.current_frame):
                    self.calibration_complete = True
                    print("Color calibration complete!")
                    print(
                        f"Current settings - Contrast: {self.contrast}, Saturation: {self.saturation}, Brightness: {self.brightness}"
                    )
                else:
                    print("Color calibration failed. Check board setup.")
            elif key == ord("s") and self.calibration_complete:
                print("Saving calibration...")
                self.save_calibration()
            elif key == ord("r"):
                self.corners = []
                self.calibration_complete = False
                print("Corners reset. Click to define 4 corners.")

        cap.release()
        cv2.destroyAllWindows()
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
