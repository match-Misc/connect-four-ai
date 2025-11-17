import pyrealsense2 as rs
import numpy as np
import cv2


# Globals used by the mouse callback
latest_color_image = None  # updated each frame (BGR)
latest_depth_m = None      # updated each frame, meters, aligned to color
last_click_info = None     # (x, y, (R, G, B), depth_m or None)


def on_mouse(event, x, y, flags, param):
    global latest_color_image, latest_depth_m, last_click_info
    if event == cv2.EVENT_LBUTTONDOWN and latest_color_image is not None:
        h, w = latest_color_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            b, g, r = latest_color_image[y, x].tolist()
            rgb = (int(r), int(g), int(b))
            d_val = None
            if latest_depth_m is not None:
                d = float(latest_depth_m[y, x])
                # Some sensors may report 0 for invalid depth
                d_val = d if d > 0 else None
            if d_val is not None:
                print(f"Clicked at ({x},{y}) -> RGB={rgb} Depth={d_val:.3f}m")
            else:
                print(f"Clicked at ({x},{y}) -> RGB={rgb} Depth=--")
            last_click_info = (x, y, rgb, d_val)

pipeline = rs.pipeline()
config = rs.config()
# Enable aligned depth + color
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = float(depth_sensor.get_depth_scale())  # to meters
align = rs.align(rs.stream.color)

# Prepare window and mouse callback
cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Color", on_mouse)

try:
    while True:
        frames = pipeline.wait_for_frames()
        # Align depth to color
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16
        depth_m = depth_raw.astype(np.float32) * depth_scale
        # update global reference for mouse sampling
        latest_color_image = color_image
        latest_depth_m = depth_m

        # draw overlay if user clicked
        display = color_image.copy()
        if last_click_info is not None:
            cx, cy, rgb, d_val = last_click_info
            # small marker and text with RGB
            cv2.circle(display, (cx, cy), 4, (0, 255, 255), -1)
            txt = f"RGB {rgb}"
            # offset text to avoid covering the point
            tx, ty = cx + 8, max(15, cy - 8)
            cv2.putText(display, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # Depth line under RGB
            dtext = f"D: {d_val:.3f} m" if d_val is not None else "D: --"
            cv2.putText(display, dtext, (tx, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Color", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
