import os
import time
import json
import threading
import subprocess
import cv2
import socket
from flask import Flask, render_template, Response, jsonify, request, send_file
from picamera2 import Picamera2
import psutil
import datetime

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

# --- Global State ---
# Base directory for config paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Manage state for two cameras independently
CAMERAS = {
    "cam0": {
        "device_id": 0,
        "config_path": os.path.join(base_dir, 'configs', 'config_cam0.json'),
        "mode": "webui",
        "picam2": None,
        "tcp_process": None,
        "lock": threading.Lock(),
        "preview_resize": True  # Default to True for bandwidth savings
    },
    "cam1": {
        "device_id": 1,
        "config_path": os.path.join(base_dir, 'configs', 'config_cam1.json'),
        "mode": "webui",
        "picam2": None,
        "tcp_process": None,
        "lock": threading.Lock(),
        "preview_resize": True
    }
}

# --- Helper Functions ---
def get_camera_settings(cam_id):
    """Load default dimensions and camera controls from the config file."""
    width, height = 2304, 1296
    controls = {}
    try:
        cfg_path = CAMERAS[cam_id]["config_path"]
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                config = json.load(f)
                cam_cfg = config.get("camera", {})
                width = cam_cfg.get("default_width", 2304)
                height = cam_cfg.get("default_height", 1296)
                controls = config.get("controls", {})
    except Exception as e:
        print(f"Warning: Could not read config file for {cam_id}: {e}")
    return width, height, controls

def start_picamera(cam_id):
    """Initialize and start picamera2 for WebUI streaming for a specific camera."""
    cam_data = CAMERAS[cam_id]
    print(f"INFO: Starting Picamera2 for WebUI ({cam_id})...")
    try:
        if cam_data["picam2"] is None:
            cam_data["picam2"] = Picamera2(camera_num=cam_data["device_id"])
            
        width, height, controls = get_camera_settings(cam_id)
        # Use configured dimensions for preview
        cam_config = cam_data["picam2"].create_preview_configuration(
            main={'format': 'RGB888', 'size': (width, height)}
        )
        cam_data["picam2"].configure(cam_config)
        cam_data["picam2"].start()
        
        # Apply Custom Camera Controls if available
        if controls:
            try:
                cam_data["picam2"].set_controls(controls)
                print(f"INFO: Applied camera controls: {controls}")
            except Exception as ce:
                print(f"ERROR: Failed to apply camera controls on {cam_id}: {ce}")
        
        # Extract Sensor Name
        raw_id = cam_data["picam2"].camera.id
        if '/' in raw_id and '@' in raw_id:
            # Typically: /base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a -> imx708
            cam_data["sensor_name"] = raw_id.split('/')[-1].split('@')[0].upper()
        else:
            cam_data["sensor_name"] = raw_id
            
        print(f"INFO: Picamera2 ({cam_id}) started successfully. Sensor: {cam_data.get('sensor_name')}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to start Picamera2 for {cam_id}: {e}")
        cam_data["picam2"] = None
        return False

def stop_picamera(cam_id):
    """Stop and release picamera2 for a specific camera."""
    cam_data = CAMERAS[cam_id]
    print(f"INFO: Stopping Picamera2 ({cam_id})...")
    if cam_data["picam2"] is not None:
        try:
            cam_data["picam2"].stop()
            cam_data["picam2"].close()
        except Exception as e:
            print(f"Warning: Error while stopping camera {cam_id}: {e}")
        finally:
            cam_data["picam2"] = None
            print(f"INFO: Picamera2 ({cam_id}) stopped and camera resource released.")

def start_tcp_sender(cam_id):
    """Start the main.py script as a subprocess or systemd service for a specific camera."""
    cam_data = CAMERAS[cam_id]
    cfg_path = cam_data["config_path"]
    cam_num = cam_id.replace('cam', '')
    
    # Try systemd first
    try:
        subprocess.run(['sudo', 'systemctl', 'start', f'camera-sender@{cam_num}.service'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"INFO: TCP Sender ({cam_id}) started via systemd.")
        return True
    except Exception as e:
        print(f"Warning: systemd start failed for {cam_id}. Falling back to subprocess.")
        
    if cam_data["tcp_process"] is None or cam_data["tcp_process"].poll() is not None:
        print(f"INFO: Starting TCP Sender Subprocess ({cam_id}): python3 main.py -c {cfg_path}")
        try:
            cam_data["tcp_process"] = subprocess.Popen(
                ['python3', 'main.py', '-c', cfg_path],
                cwd=base_dir
            )
            print(f"INFO: TCP Sender ({cam_id}) started with PID {cam_data['tcp_process'].pid}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to start TCP Sender for {cam_id}: {e}")
            return False
    return True

def stop_tcp_sender(cam_id):
    """Terminate the main.py systemd service or subprocess if it's running."""
    cam_data = CAMERAS[cam_id]
    cam_num = cam_id.replace('cam', '')
    
    print(f"INFO: Stopping TCP Sender ({cam_id})...")
    # Try systemd first
    try:
        subprocess.run(['sudo', 'systemctl', 'stop', f'camera-sender@{cam_num}.service'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"INFO: TCP Sender ({cam_id}) stopped via systemd. Waiting 2 seconds for V4L2 to fully unbind...")
        time.sleep(2)
    except Exception as e:
        print(f"Warning: systemd stop failed for {cam_id}. Attempting to stop Subprocess directly.")
        
    if cam_data["tcp_process"] is not None and cam_data["tcp_process"].poll() is None:
        try:
            cam_data["tcp_process"].terminate()
            try:
                cam_data["tcp_process"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Warning: Process {cam_id} did not terminate gracefully, forcing kill.")
                cam_data["tcp_process"].kill()
        except Exception as e:
            print(f"Warning: Error while killing TCP Sender {cam_id}: {e}")
        finally:
            print(f"INFO: TCP Sender Subprocess ({cam_id}) stopped.")
    cam_data["tcp_process"] = None

# --- Camera Generator ---
def generate_frames(cam_id):
    """Generator function that yields JPEG frames from Picamera2."""
    cam_data = CAMERAS[cam_id]
    while True:
        frame_bytes = None
        w, h = 0, 0
        
        with cam_data["lock"]:
            if cam_data["mode"] != 'webui' or cam_data["picam2"] is None:
                # If not in WebUI mode, yield nothing or sleep
                pass
            else:
                try:
                    # Capture frame from the camera
                    frame = cam_data["picam2"].capture_array()
                    
                    # Resize frame to save CPU and Bandwidth in WebUI preview mode
                    PREVIEW_MAX_WIDTH = 800
                    h, w = frame.shape[:2]
                    
                    if cam_data.get("preview_resize", True) and w > PREVIEW_MAX_WIDTH:
                        scale = PREVIEW_MAX_WIDTH / w
                        target_h = int(h * scale)
                        frame = cv2.resize(frame, (PREVIEW_MAX_WIDTH, target_h), interpolation=cv2.INTER_AREA)
                        h, w = frame.shape[:2]

                    # Add resolution label to the bottom right corner
                    text = f"{w}x{h}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5 if w < 1000 else 1.0
                    thickness = 1 if w < 1000 else 2
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_w, text_h = text_size
                    org = (w - text_w - 10, h - 10)
                    
                    # Draw black background rectangle for better visibility
                    cv2.rectangle(frame, (org[0] - 5, org[1] - text_h - 5), (w, h), (0, 0, 0), -1)
                    cv2.putText(frame, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    # Encode to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                except Exception as e:
                    print(f"ERROR: Frame capture error on {cam_id}: {e}")

        if frame_bytes is None:
             time.sleep(1)
             continue
             
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.05)

# --- API Routes for Config ---
@app.route('/api/config/<cam_id>', methods=['GET'])
def get_config(cam_id):
    """Returns the current config from disk."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    try:
        cfg_path = CAMERAS[cam_id]["config_path"]
        if os.path.exists(cfg_path):
            config = json.load(open(cfg_path, 'r'))
            # Add runtime preview_resize state to the config for UI
            config["preview_resize"] = CAMERAS[cam_id].get("preview_resize", True)
            return jsonify(config)
        else:
            return jsonify({"error": "Config file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/<cam_id>', methods=['POST'])
def save_config(cam_id):
    """Receives, validates, and saves new config to disk."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    try:
        new_config = request.json
        if not new_config:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        # Extract and update runtime preview_resize state if present
        if "preview_resize" in new_config:
            CAMERAS[cam_id]["preview_resize"] = bool(new_config.pop("preview_resize"))
            print(f"INFO: Updated preview_resize for {cam_id} to {CAMERAS[cam_id]['preview_resize']}")

        required_sections = ["tcp", "mqtt", "camera", "preprocessing", "sftp"]
        for section in required_sections:
            if section not in new_config:
                new_config[section] = {}

        cfg_path = CAMERAS[cam_id]["config_path"]
        with open(cfg_path, 'w') as f:
            json.dump(new_config, f, indent=4)
            
        print(f"INFO: Config file for {cam_id} updated via WebUI.")

        cam_data = CAMERAS[cam_id]
        with cam_data["lock"]:
            if cam_data["mode"] == 'tcp':
                print(f"INFO: Restarting TCP Sender {cam_id} to apply new configuration...")
                stop_tcp_sender(cam_id)
                time.sleep(1)
                start_tcp_sender(cam_id)
                
        return jsonify({"status": "success", "message": "Configuration saved successfully."})
        
    except Exception as e:
        print(f"ERROR saving config for {cam_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/<cam_id>/camera_controls', methods=['POST'])
def update_camera_controls(cam_id):
    """Receives camera control properties, saves them, and applies immediately if streaming."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    try:
        controls_update = request.json
        if not controls_update:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        cfg_path = CAMERAS[cam_id]["config_path"]
        
        # Load existing config
        config = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                config = json.load(f)
                
        # Update or create the controls namespace
        if "controls" not in config:
            config["controls"] = {}
            
        config["controls"].update(controls_update)
        
        # Save seamlessly without destroying stream mode
        with open(cfg_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"INFO: Camera controls for {cam_id} updated: {controls_update}")
        
        # Apply the new controls instantly if picam2 is actively streaming in WebUI
        cam_data = CAMERAS[cam_id]
        with cam_data["lock"]:
            if cam_data["mode"] == "webui" and cam_data["picam2"] is not None:
                try:
                    cam_data["picam2"].set_controls(config["controls"])
                    print(f"INFO: Applied dynamic controls to active stream.")
                except Exception as ce:
                    print(f"ERROR: Failed to apply dynamic controls on {cam_id}: {ce}")
                    
        return jsonify({"status": "success", "message": "Camera controls updated and saved."})
        
    except Exception as e:
        print(f"ERROR updating camera controls for {cam_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/calibrate/capture/<cam_id>', methods=['GET'])
def calibrate_capture(cam_id):
    """Capture a high-res frame for calibration."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    cam_data = CAMERAS[cam_id]
    save_path = os.path.join(base_dir, "logs", f"{cam_id}_calibration_target.jpg")
    
    with cam_data["lock"]:
        if cam_data["mode"] == 'webui':
            if cam_data["picam2"] is None:
                return jsonify({"error": "Camera not running"}), 500
                
            try:
                # Capture high-res frame
                frame = cam_data["picam2"].capture_array()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, frame)
                return send_file(save_path, mimetype='image/jpeg')
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            # TCP Mode - trigger via MQTT
            try:
                cfg_path = cam_data["config_path"]
                with open(cfg_path, 'r') as f:
                    config = json.load(f)
                
                broker = config.get("mqtt", {}).get("broker", "localhost")
                port = config.get("mqtt", {}).get("port", 1883)
                topic = config.get("mqtt", {}).get("topic_cmd", f"wf51/w/command/{cam_id}")
                user = config.get("mqtt", {}).get("username", "")
                password = config.get("mqtt", {}).get("password", "")
                
                import paho.mqtt.client as mqtt
                client = mqtt.Client()
                if user and password:
                    client.username_pw_set(user, password)
                client.connect(broker, port, 60)
                client.publish(topic, json.dumps({"action": "capture"}))
                client.disconnect()
                
                # Wait for file to update (max 3 seconds)
                start_mtime = os.path.getmtime(save_path) if os.path.exists(save_path) else 0
                for _ in range(30):
                    time.sleep(0.1)
                    if os.path.exists(save_path) and os.path.getmtime(save_path) > start_mtime:
                        return send_file(save_path, mimetype='image/jpeg')
                        
                return jsonify({"error": "Timeout waiting for TCP Sender to capture image"}), 504
            except Exception as e:
                return jsonify({"error": f"MQTT trigger failed: {e}"}), 500

@app.route('/api/calibrate/wait/<cam_id>', methods=['GET'])
def calibrate_wait(cam_id):
    """Wait for an external MQTT capture trigger."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    cam_data = CAMERAS[cam_id]
    save_path = os.path.join(base_dir, "logs", f"{cam_id}_calibration_target.jpg")
    
    try:
        cfg_path = cam_data["config_path"]
        with open(cfg_path, 'r') as f:
            config = json.load(f)
            
        broker = config.get("mqtt", {}).get("broker", "localhost")
        port = config.get("mqtt", {}).get("port", 1883)
        topic = config.get("mqtt", {}).get("topic_cmd", f"wf51/w/command/{cam_id}")
        user = config.get("mqtt", {}).get("username", "")
        password = config.get("mqtt", {}).get("password", "")
        
        trigger_received = [False]
        
        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                if payload.get("action") == "capture":
                    trigger_received[0] = True
            except:
                pass

        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        if user and password:
            client.username_pw_set(user, password)
        client.on_message = on_message
        
        client.connect(broker, port, 60)
        client.subscribe(topic)
        client.loop_start()
        
        # Wait up to 60 seconds for an external trigger
        for _ in range(600):
            time.sleep(0.1)
            if trigger_received[0]:
                break
                
        client.loop_stop()
        client.disconnect()
        
        if not trigger_received[0]:
            return jsonify({"error": "Timeout waiting for external MQTT trigger"}), 504
            
        # If trigger received, capture the frame!
        if cam_data["mode"] == 'webui':
            with cam_data["lock"]:
                if cam_data["picam2"] is None:
                    return jsonify({"error": "Camera not running"}), 500
                frame = cam_data["picam2"].capture_array()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, frame)
                return send_file(save_path, mimetype='image/jpeg')
        else:
            # If TCP mode, main.py should have captured it already since it also listens to MQTT.
            start_mtime = os.path.getmtime(save_path) if os.path.exists(save_path) else 0
            for _ in range(30):
                time.sleep(0.1)
                # Wait for file timestamp to update or file to be created
                if os.path.exists(save_path) and os.path.getmtime(save_path) > start_mtime:
                    return send_file(save_path, mimetype='image/jpeg')
            
            # Fallback if the file didn't update but we somehow got the trigger
            if os.path.exists(save_path):
                return send_file(save_path, mimetype='image/jpeg')
            return jsonify({"error": "TCP Sender received trigger but file wasn't updated"}), 500

    except Exception as e:
        return jsonify({"error": f"MQTT listener failed: {e}"}), 500

@app.route('/api/calibrate/save_alignment/<cam_id>', methods=['POST'])
def save_alignment(cam_id):
    """Save 4 marks and 4 corners and generate templates."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    try:
        data = request.json
        marks = data.get("marks", []) # [{"x": 10, "y": 20}, ...]
        corners = data.get("corners", []) # [{"x": 10, "y": 20}, ...]
        
        # Backward compatibility or if user skipped corners
        if cam_id != 'cam0' and len(corners) != 4:
            if len(marks) == 4:
                corners = marks
            elif len(marks) == 2:
                # If only 2 marks, create 4 corners from bounding box of the 2 marks
                x_coords = [m.get("x", 0) for m in marks]
                y_coords = [m.get("y", 0) for m in marks]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                # Expand slightly to encompass the marks width/height
                w0 = marks[0].get("width", 60)
                h0 = marks[0].get("height", 60)
                max_x += w0
                max_y += h0
                corners = [
                    {"x": min_x, "y": min_y},
                    {"x": max_x, "y": min_y},
                    {"x": max_x, "y": max_y},
                    {"x": min_x, "y": max_y}
                ]
            
        if cam_id == 'cam0':
            if len(marks) != 1:
                return jsonify({"error": "Exactly 1 marker point is required for Camera 0"}), 400
        else:
            if len(marks) != 2:
                return jsonify({"error": "Exactly 2 marker points are required for Camera 1"}), 400
            
        img_path = os.path.join(base_dir, "logs", f"{cam_id}_calibration_target.jpg")
        if not os.path.exists(img_path):
            return jsonify({"error": "Reference image not found. Please capture first."}), 404
            
        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"error": "Failed to read reference image"}), 500
            
        # Ensure templates directory exists
        templates_dir = os.path.join(base_dir, "configs", "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        h_img, w_img = img.shape[:2]
        
        for i, m in enumerate(marks):
            # m is now expected to be {x, y, width, height}
            x = int(m.get("x", 0))
            y = int(m.get("y", 0))
            w = int(m.get("width", 60))
            h = int(m.get("height", 60))
            
            # Boundary checks
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            
            crop = img[y1:y2, x1:x2]
            tmpl_name = f"{cam_id}_mark{i}.jpg"
            if crop.size > 0:
                cv2.imwrite(os.path.join(templates_dir, tmpl_name), crop)
            
            # The homography script expects the mark object to contain the center coordinate of the box
            m["center_x"] = x + (w / 2.0)
            m["center_y"] = y + (h / 2.0)
            m["template"] = tmpl_name

        calib_data = {
            "calibration_marks": marks,
            "calibration_corners": corners
        }
        
        # Add padding configuration specifically for cam0
        if cam_id == 'cam0':
            calib_data["padding"] = data.get("padding", 50)
        
        config_file = os.path.join(base_dir, "configs", f"{cam_id}_calibration_points.json")
        with open(config_file, 'w') as f:
            json.dump(calib_data, f, indent=4)
            
        return jsonify({"status": "success", "message": "Alignment templates saved"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/calibrate/save_crop/<cam_id>', methods=['POST'])
def save_crop(cam_id):
    """Save crop regions."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    try:
        data = request.json
        regions = data.get("regions", [])
        ref_size = data.get("reference_image_size", {"width": 2304, "height": 1296})
        
        crop_data = {
            "reference_image_size": ref_size,
            "mask_regions": regions
        }
        
        config_file = os.path.join(base_dir, "configs", f"{cam_id}_crop_regions.json")
        with open(config_file, 'w') as f:
            json.dump(crop_data, f, indent=4)
            
        return jsonify({"status": "success", "message": "Crop regions saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Routes ---
@app.route('/api/system_stats', methods=['GET'])
def system_stats():
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_c = float(f.read().strip()) / 1000.0
        except Exception:
            temp_c = 0.0

        return jsonify({
            "status": "success",
            "cpu": cpu,
            "ram": ram.percent,
            "temp": round(temp_c, 1),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "disk_total_gb": round(disk.total / (1024**3), 1)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    """Render the main WebUI."""
    hostname = socket.gethostname()
    # We pass the dictionary of modes to the template although JS will fetch it anyway
    modes = {k: v["mode"] for k, v in CAMERAS.items()}
    return render_template('index.html', modes=modes, hostname=hostname)

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    """Video streaming route."""
    if cam_id not in CAMERAS:
        return "Camera ID not found", 404
        
    if CAMERAS[cam_id]["mode"] == 'webui':
        return Response(generate_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Camera is currently allocated to TCP Sender.", status=409)

def get_camera_display_name(cam_id):
    """Load the custom display name from the config file, fallback to default."""
    display_name = f"Camera {CAMERAS[cam_id]['device_id']}"
    try:
        cfg_path = CAMERAS[cam_id]["config_path"]
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                config = json.load(f)
                custom_name = config.get("camera", {}).get("name")
                if custom_name:
                    display_name = custom_name
    except:
        pass
    return display_name

@app.route('/debug/<cam_id>')
def debug_view(cam_id):
    """Serve the debug log viewer page."""
    if cam_id not in CAMERAS:
        return "Camera ID not found", 404
    hostname = socket.gethostname()
    display_name = get_camera_display_name(cam_id)
    server_start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('debug.html', cam_id=cam_id, hostname=hostname, display_name=display_name, start_time=server_start_time)

@app.route('/api/logs/<cam_id>')
def api_logs(cam_id):
    """Fetch recent systemd logs for the specific camera based on its current mode."""
    if cam_id not in CAMERAS:
        return jsonify({"error": "Invalid camera ID"}), 400
        
    mode = CAMERAS[cam_id]["mode"]
    cam_num = cam_id.replace('cam', '')
    since = request.args.get('since')
    
    try:
        if mode == 'tcp':
            service_name = f"camera-sender@{cam_num}.service"
        else:
            service_name = "camera-app.service"
            
        cmd = ['sudo', 'journalctl', '-u', service_name, '--no-pager', '--output=short-iso', '-n', '1000']
        if since:
            cmd.extend(['--since', since])
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Filter out noisy sudo and journalctl logs
        filtered_lines = []
        for line in result.stdout.splitlines():
            if 'COMMAND=/usr/bin/journalctl' in line or 'pam_unix(sudo:session)' in line:
                continue
            filtered_lines.append(line)
        
        filtered_logs = '\n'.join(filtered_lines) + '\n' if filtered_lines else ""

        return jsonify({
            "status": "success",
            "mode": mode,
            "service": service_name,
            "logs": filtered_logs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Return the current system status for all cameras."""
    res = {}
    for cid, cam in CAMERAS.items():
        res[cid] = {
            "mode": cam["mode"],
            "tcp_pid": cam["tcp_process"].pid if cam["tcp_process"] and cam["tcp_process"].poll() is None else None,
            "sensor_name": cam.get("sensor_name", "Unknown"),
            "display_name": get_camera_display_name(cid)
        }
    return jsonify(res)

@app.route('/switch_mode', methods=['POST'])
def switch_mode():
    """API endpoint to switch modes per camera."""
    data = request.json
    target_mode = data.get('mode')
    cam_id = data.get('cam_id')
    
    if target_mode not in ['webui', 'tcp'] or cam_id not in CAMERAS:
        return jsonify({"error": "Invalid mode or camera ID"}), 400
        
    cam_data = CAMERAS[cam_id]
    
    with cam_data["lock"]:
        if target_mode == cam_data["mode"]:
            return jsonify({"status": "Mode already active", "mode": cam_data["mode"]})
            
        print(f"\n=========================================")
        print(f"[{cam_id}] SWITCHING MODE: {cam_data['mode']} -> {target_mode}")
        print(f"=========================================\n")
        
        if target_mode == 'tcp':
            stop_picamera(cam_id)
            success = start_tcp_sender(cam_id)
            if success:
                cam_data["mode"] = 'tcp'
            else:
                start_picamera(cam_id)
                
        elif target_mode == 'webui':
            stop_tcp_sender(cam_id)
            time.sleep(2)
            success = start_picamera(cam_id)
            if success:
                cam_data["mode"] = 'webui'
            else:
                time.sleep(3)
                start_picamera(cam_id)
                cam_data["mode"] = 'webui' 
                
    return jsonify({"status": "success", "mode": cam_data["mode"]})

if __name__ == '__main__':
    # Initialize initial state for all cameras
    for cid, cam in CAMERAS.items():
        with cam["lock"]:
            if cam["mode"] == 'webui':
                stop_tcp_sender(cid)
                time.sleep(1)
                start_picamera(cid)
            elif cam["mode"] == 'tcp':
                stop_picamera(cid)
                start_tcp_sender(cid)
            
    # Run the Flask app on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
