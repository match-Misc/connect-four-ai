from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
import json
import pyrealsense2 as rs

app = Flask(__name__)

class RealSenseCalibrator:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.colorizer = rs.colorizer()
        self.exposure = int(self.depth_sensor.get_option(rs.option.exposure))
        self.gain = int(self.depth_sensor.get_option(rs.option.gain))
        self.laser_power = int(self.depth_sensor.get_option(rs.option.laser_power))
        self.visual_preset = int(self.depth_sensor.get_option(rs.option.visual_preset))
        self.min_depth = 0
        self.max_depth = 2000
        self.emitter = int(self.depth_sensor.get_option(rs.option.emitter_enabled))
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

    def get_depth_image(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        depth_frame = rs.threshold_filter(self.min_depth / 1000.0, self.max_depth / 1000.0).process(depth_frame)
        colorized = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return colorized

    def update_params(self, params):
        self.exposure = int(params.get('exposure', self.exposure))
        self.gain = int(params.get('gain', self.gain))
        self.laser_power = int(params.get('laser_power', self.laser_power))
        self.visual_preset = int(params.get('visual_preset', self.visual_preset))
        self.min_depth = int(params.get('min_depth', self.min_depth))
        self.max_depth = int(params.get('max_depth', self.max_depth))
        self.emitter = int(params.get('emitter', self.emitter))
        try:
            self.depth_sensor.set_option(rs.option.exposure, float(self.exposure))
            self.depth_sensor.set_option(rs.option.gain, float(self.gain))
            self.depth_sensor.set_option(rs.option.laser_power, float(self.laser_power))
            self.depth_sensor.set_option(rs.option.visual_preset, float(self.visual_preset))
            self.depth_sensor.set_option(rs.option.emitter_enabled, float(self.emitter))
        except Exception as e:
            print(f"Error setting params: {e}")

    def save_settings(self):
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
        print("Settings saved")

calibrator = RealSenseCalibrator()

@app.route('/')
def index():
    return render_template('realsense-calibration.html')

@app.route('/frame')
def frame():
    img = calibrator.get_depth_image()
    if img is None:
        # Return a placeholder image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "No depth feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/update_params', methods=['POST'])
def update_params():
    params = request.form
    calibrator.update_params(params)
    return jsonify({'status': 'Parameters updated'})

@app.route('/save', methods=['POST'])
def save():
    calibrator.save_settings()
    return jsonify({'status': 'Settings saved'})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)