import pyrealsense2 as rs
import numpy as np
import cv2
import json


def save_settings_callback(exp, gain, laser, preset, min_d, max_d, emitter):
    """Callback to save settings when button is clicked"""
    settings = {
        "exposure": exp,
        "gain": gain,
        "laser_power": laser,
        "visual_preset": preset,
        "min_depth_mm": min_d,
        "max_depth_mm": max_d,
        "emitter_enabled": emitter,
        "auto_exposure": 0
    }
    with open("realsense_settings.json", "w") as f:
        json.dump(settings, f, indent=4)
    print("Settings saved to realsense_settings.json")
    print(json.dumps(settings, indent=2))


def adjust_realsense_parameters():
    """
    GUI to adjust RealSense camera parameters (exposure, gain, laser power)
    and visualize depth colormap in real-time.
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get the depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()
    
    # Get parameter ranges
    try:
        exposure_range = depth_sensor.get_option_range(rs.option.exposure)
        gain_range = depth_sensor.get_option_range(rs.option.gain)
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        
        exp_min, exp_max = int(exposure_range.min), int(exposure_range.max)
        gain_min, gain_max = int(gain_range.min), int(gain_range.max)
        laser_min, laser_max = int(laser_range.min), int(laser_range.max)
        
        # Get current values
        exp_current = int(depth_sensor.get_option(rs.option.exposure))
        gain_current = int(depth_sensor.get_option(rs.option.gain))
        laser_current = int(depth_sensor.get_option(rs.option.laser_power))
        
        print(f"Exposure range: {exp_min} - {exp_max}, current: {exp_current}")
        print(f"Gain range: {gain_min} - {gain_max}, current: {gain_current}")
        print(f"Laser Power range: {laser_min} - {laser_max}, current: {laser_current}")
    except Exception as e:
        print(f"Could not get parameter ranges: {e}")
        exp_min, exp_max, exp_current = 1, 10000, 5000
        gain_min, gain_max, gain_current = 16, 248, 128
        laser_min, laser_max, laser_current = 0, 360, 150
    
    # Get visual preset and depth units options
    try:
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        preset_min, preset_max = int(preset_range.min), int(preset_range.max)
        preset_current = int(depth_sensor.get_option(rs.option.visual_preset))
        print(f"Visual Preset range: {preset_min} - {preset_max}, current: {preset_current}")
        print("Presets: 0=Custom, 1=Default, 2=Hand, 3=High Accuracy, 4=High Density, 5=Medium Density")
    except Exception as e:
        print(f"Could not get visual preset: {e}")
        preset_min, preset_max, preset_current = 0, 5, 1
    
    # Get depth units
    try:
        depth_units = depth_sensor.get_option(rs.option.depth_units)
        print(f"Depth Units: {depth_units} (meters per unit)")
    except Exception as e:
        print(f"Could not get depth units: {e}")
    
    # Get min/max depth (for filtering)
    try:
        min_dist_range = depth_sensor.get_option_range(rs.option.min_distance)
        min_dist = int(min_dist_range.default)
        max_dist = 2000  # mm
        print(f"Min distance: {min_dist} mm")
    except Exception as e:
        print(f"Could not get min distance: {e}")
        min_dist = 0
        max_dist = 2000
    
    # Get emitter enabled option
    try:
        emitter_current = int(depth_sensor.get_option(rs.option.emitter_enabled))
        print(f"Emitter enabled: {emitter_current} (0=Off, 1=On, 2=Auto)")
    except Exception as e:
        print(f"Could not get emitter option: {e}")
        emitter_current = 1
    
    # Turn off auto exposure
    try:
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        print("Auto exposure disabled")
    except Exception as e:
        print(f"Could not disable auto exposure: {e}")
    
    # Create window and trackbars
    cv2.namedWindow("RealSense Depth", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Exposure", "RealSense Depth", exp_current, exp_max if exp_max > 0 else 10000, lambda x: None)
    cv2.createTrackbar("Gain", "RealSense Depth", gain_current, gain_max if gain_max > 0 else 248, lambda x: None)
    cv2.createTrackbar("Laser Power", "RealSense Depth", laser_current, laser_max if laser_max > 0 else 360, lambda x: None)
    cv2.createTrackbar("Visual Preset", "RealSense Depth", preset_current, preset_max if preset_max > 0 else 5, lambda x: None)
    cv2.createTrackbar("Min Depth (mm)", "RealSense Depth", min_dist, 5000, lambda x: None)
    cv2.createTrackbar("Max Depth (mm)", "RealSense Depth", max_dist, 10000, lambda x: None)
    cv2.createTrackbar("Emitter", "RealSense Depth", emitter_current, 2, lambda x: None)
    
    # Create a button for saving (using a trackbar as a button)
    cv2.createTrackbar("Save Settings", "RealSense Depth", 0, 1, lambda x: None)
    
    # Create colorizer for depth visualization
    colorizer = rs.colorizer()
    
    print("Press 'q' to quit, or click 'Save Settings' button")
    
    last_save_button_state = 0
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if not depth_frame:
                continue
            
            # Get slider values and update camera parameters
            try:
                exp_val = cv2.getTrackbarPos("Exposure", "RealSense Depth")
                gain_val = cv2.getTrackbarPos("Gain", "RealSense Depth")
                laser_val = cv2.getTrackbarPos("Laser Power", "RealSense Depth")
                preset_val = cv2.getTrackbarPos("Visual Preset", "RealSense Depth")
                min_depth = cv2.getTrackbarPos("Min Depth (mm)", "RealSense Depth")
                max_depth = cv2.getTrackbarPos("Max Depth (mm)", "RealSense Depth")
                emitter_val = cv2.getTrackbarPos("Emitter", "RealSense Depth")
                save_button = cv2.getTrackbarPos("Save Settings", "RealSense Depth")
                
                # Check if save button was clicked (transitioned from 0 to 1)
                if save_button == 1 and last_save_button_state == 0:
                    save_settings_callback(exp_val, gain_val, laser_val, preset_val, 
                                         min_depth, max_depth, emitter_val)
                    # Reset button
                    cv2.setTrackbarPos("Save Settings", "RealSense Depth", 0)
                last_save_button_state = save_button
                
                # Clamp values to valid ranges
                exp_val = max(exp_min, min(exp_max, exp_val))
                gain_val = max(gain_min, min(gain_max, gain_val))
                laser_val = max(laser_min, min(laser_max, laser_val))
                preset_val = max(preset_min, min(preset_max, preset_val))
                
                depth_sensor.set_option(rs.option.visual_preset, float(preset_val))
                depth_sensor.set_option(rs.option.exposure, float(exp_val))
                depth_sensor.set_option(rs.option.gain, float(gain_val))
                depth_sensor.set_option(rs.option.laser_power, float(laser_val))
                depth_sensor.set_option(rs.option.emitter_enabled, float(emitter_val))
            except Exception as e:
                print(f"Could not set parameter: {e}")
            
            # Apply depth filtering
            depth_frame = rs.threshold_filter(min_depth / 1000.0, max_depth / 1000.0).process(depth_frame)
            
            # Colorize depth frame
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # Display
            cv2.imshow("RealSense Depth", colorized_depth)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    adjust_realsense_parameters()
