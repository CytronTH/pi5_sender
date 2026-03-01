import socket
import struct
import time
import json
import os
import threading
import argparse
import paho.mqtt.client as mqtt
import cv2
import numpy as np
from picamera2 import Picamera2
from queue import Queue
import glob

import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))

from src.align_wall_boxes import calculate_canonical_targets, find_mark
from src.batch_shadow_removal import remove_shadows_divisive
from src.batch_process_grayscale import cv2 as dummy_cv2
import src.crop_wall_boxes as cwb

# --- Configuration Loader ---
parser = argparse.ArgumentParser(description="Camera Sender Script")
parser.add_argument('-c', '--config', type=str, default=os.path.join(base_dir, 'configs', 'config.json'), help='Path to config file')
parser.add_argument('--mock_dir', type=str, default=None, help='Directory containing mock images for offline testing')
parser.add_argument('--debug_align', action='store_true', help='Save visualization of the alignment process to disk')
parser.add_argument('--disable_clahe', action='store_true', help='Disable CLAHE enhancement and use raw grayscale')
args = parser.parse_args()

try:
    with open(args.config, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"CRITICAL: {args.config} not found. Using barebone defaults!")
    config = {}

TCP_IP = config.get("tcp", {}).get("ip", "10.10.10.199")
TCP_PORT = config.get("tcp", {}).get("port", 8080)
MQTT_BROKER = config.get("mqtt", {}).get("broker", "10.10.10.199")
MQTT_PORT = config.get("mqtt", {}).get("port", 1883)
MQTT_TOPIC_CMD = config.get("mqtt", {}).get("topic_cmd", "camera/command")
MQTT_TOPIC_STATUS = config.get("mqtt", {}).get("topic_status", "camera/status")
MQTT_USERNAME = config.get("mqtt", {}).get("username", None)
MQTT_PASSWORD = config.get("mqtt", {}).get("password", None)

CAMERA_ID = config.get("camera", {}).get("id", 0)
JPEG_QUALITY = config.get("camera", {}).get("jpeg_quality", 90)
CONTINUOUS_STREAM = config.get("camera", {}).get("continuous_stream", True)
STREAM_INTERVAL = config.get("camera", {}).get("stream_interval", 1.0)
LOOP_DELAY = config.get("camera", {}).get("loop_delay", 0.05)

# Default camera config
current_width = config.get("camera", {}).get("default_width", 2304)
current_height = config.get("camera", {}).get("default_height", 1296)

# Global state
picam2 = None
tcp_socket = None
capture_triggered = False
capture_lock = threading.Lock()
image_queue = Queue(maxsize=7) # Limit queue size to avoid OOM
last_mock_image_name = None # Track the current mock image name

# Image Processing config
preproc_config = None
preproc_templates = None
mask_config = None
target_marks = None
output_size = None
ref_mark_points = None
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

class MockCamera:
    def __init__(self, image_dir):
        self.images = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.images.sort() # Ensure consistent order
        self.idx = 0
        if not self.images:
            raise ValueError(f"No mock images found in {image_dir}")
        print(f"INFO: Initialized MockCamera with {len(self.images)} images from {image_dir}")

    def capture_array(self):
        global last_mock_image_name
        img_path = self.images[self.idx]
        last_mock_image_name = os.path.basename(img_path)
        frame = cv2.imread(img_path)
        if frame is None:
             raise RuntimeError(f"Failed to read mock image: {img_path}")
        self.idx = (self.idx + 1) % len(self.images)
        return frame
        
    def start(self): pass
    def stop(self): pass
    def configure(self, config): pass
    def create_preview_configuration(self, main): return {}
    def set_controls(self, controls): pass

def init_preprocessing():
    global preproc_config, preproc_templates, mask_config
    global target_marks, output_size, ref_mark_points
    
    prep_config = config.get("preprocessing", {})
    enable_align = prep_config.get("enable_alignment", True)
    enable_crop = prep_config.get("enable_box_cropping", True)
    
    # 1. Load Alignment Config
    if enable_align:
        print("INFO: Loading Calibration Config...")
        try:
            preproc_config, preproc_templates = cwb.load_calibration()
            target_marks, output_size = calculate_canonical_targets(preproc_config)
            ref_mark_points = np.array([[m["x"], m["y"]] for m in preproc_config["calibration_marks"]], dtype=np.float32)
            print(f"INFO: Alignment Targets calculated. Output size: {output_size}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: Failed to load calibration config: {e}. Program will terminate.")
        
    # 2. Load Mask Config
    if enable_crop:
        print("INFO: Loading Mask Config...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mask_config_path = os.path.join(base_dir, "configs", "masks.json")
        
        try:
            with open(mask_config_path, "r") as f:
                mask_config = json.load(f)
            print(f"INFO: Loaded mask config with {len(mask_config.get('mask_regions', []))} regions.")
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: Failed to load mask config: {e}. Program will terminate.")

def get_cpu_temperature():
    """Reads CPU temperature from system files."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000.0
        return temp
    except Exception:
        return 0.0

def save_config():
    """Saves the current config dictionary back to the JSON file."""
    try:
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"INFO: Saved updated configuration to {args.config}")
    except Exception as e:
        print(f"ERROR: Failed to save config to {args.config}: {e}")

last_cpu_idle = 0
last_cpu_total = 0

def get_cpu_usage():
    """Calculates CPU usage percentage from /proc/stat."""
    global last_cpu_idle, last_cpu_total
    try:
        with open('/proc/stat', 'r') as f:
            line = f.readline()
        
        if not line.startswith('cpu '):
            return 0.0
            
        parts = [float(p) for p in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        non_idle = parts[0] + parts[1] + parts[2] + (sum(parts[5:8]) if len(parts) > 7 else 0)
        total = idle + non_idle
        
        total_diff = total - last_cpu_total
        idle_diff = idle - last_cpu_idle
        
        last_cpu_total = total
        last_cpu_idle = idle
        
        # Return 0.0 on the very first call since delta is not meaningful yet
        if total == total_diff: 
            return 0.0
            
        if total_diff > 0:
            return (total_diff - idle_diff) / total_diff * 100.0
        return 0.0
    except Exception:
        return 0.0

def get_ram_usage():
    """Calculates RAM usage percentage from /proc/meminfo."""
    try:
        with open('/proc/meminfo', 'r') as mem:
            mem_info = mem.readlines()
        
        mem_total = 0
        mem_free = 0
        mem_buffers = 0
        mem_cached = 0
        
        for line in mem_info:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1])
            elif line.startswith('MemFree:'):
                mem_free = int(line.split()[1])
            elif line.startswith('Buffers:'):
                mem_buffers = int(line.split()[1])
            elif line.startswith('Cached:'):
                mem_cached = int(line.split()[1])
                
        used_memory = mem_total - mem_free - mem_buffers - mem_cached
        if mem_total > 0:
            return (used_memory / mem_total) * 100.0
        return 0.0
    except Exception:
        return 0.0

def connect_tcp():
    global tcp_socket
    if tcp_socket:
        try:
            tcp_socket.close()
        except:
            pass
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(5.0) # Handle timeouts gracefully
        tcp_socket.connect((TCP_IP, TCP_PORT))
        print(f"INFO: Connected to TCP server at {TCP_IP}:{TCP_PORT}")
        return True
    except socket.timeout:
        print("ERROR: TCP Connection timed out.")
        tcp_socket = None
        return False
    except ConnectionRefusedError:
        print(f"ERROR: TCP Connection refused by {TCP_IP}:{TCP_PORT}.")
        tcp_socket = None
        return False
    except Exception as e:
        print(f"ERROR: TCP Connection failed: {e}")
        tcp_socket = None
        return False

def send_image(frame, image_id="raw_image"):
    global tcp_socket
    if tcp_socket is None:
        if not connect_tcp():
            return

    try:
        # 1. Encode to JPEG
        # Using a balanced quality to save memory on Zero 2W and reduce transmission time
        result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not result:
            print("ERROR: Failed to encode image")
            return

        data = encoded_frame.tobytes()
        img_size = len(data)
        
        # 2. Create JSON Metadata Header
        metadata = {
            "id": image_id,
            "size": img_size
        }
        metadata_json = json.dumps(metadata).encode('utf-8')
        meta_size = len(metadata_json)
        
        # 3. Protocol: 
        # [4-byte Metadata Size] + [JSON Metadata] + [JPEG Data]
        # Receiver reads 4 bytes -> gets meta_size -> reads meta_size bytes for JSON -> parses JSON to get img_size -> reads img_size bytes for Image.
        header = struct.pack(">L", meta_size)
        
        # Send all parts as a single continuous stream
        tcp_socket.sendall(header + metadata_json + data)
        print(f"INFO: Sent {image_id}: {current_width}x{current_height} ({img_size} bytes)")
        
    except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
        print(f"ERROR: TCP Send Error: {e}")
        tcp_socket.close()
        tcp_socket = None
    except Exception as e:
        print(f"ERROR: Unexpected error during send: {e}")

def pre_process_worker():
    """ Worker thread to process images from the queue and send them. """
    global target_marks, output_size, ref_mark_points, preproc_templates, mask_config, clahe_obj
    
    print("INFO: Pre-processing worker thread started.")
    while True:
        try:
            frame = image_queue.get()
            
            prep_config = config.get("preprocessing", {})
            enable_align = prep_config.get("enable_alignment", True)
            enable_shadow = prep_config.get("enable_shadow_removal", True)
            enable_gray = prep_config.get("enable_grayscale", True)
            enable_clahe = prep_config.get("enable_clahe", True)
            enable_crop = prep_config.get("enable_box_cropping", True)
            
            # If no configs loaded but we need them, fallback to raw
            if (enable_align and not preproc_config) or (enable_crop and not mask_config):
                send_image(frame, image_id="raw_image")
                image_queue.task_done()
                continue
                
            # --- 1. Alignment ---
            if enable_align:
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found_marks = []
                
                # Create a debug image if requested
                debug_img = frame.copy() if args.debug_align else None
                
                tmpl1 = cv2.cvtColor(preproc_templates[0], cv2.COLOR_BGR2GRAY)
                loc, score = find_mark(img_gray, tmpl1)
                
                if score < 0.5:
                    if debug_img is not None:
                        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"debug_align_fail_m1_{int(time.time()*100)}.jpg")
                        cv2.imwrite(save_path, debug_img)
                    raise ValueError("Mark 1 not found. Alignment failed.")
                    
                th, tw = tmpl1.shape
                m1_cx, m1_cy = loc[0] + tw//2, loc[1] + th//2
                found_marks.append([m1_cx, m1_cy])
                
                if debug_img is not None:
                    cv2.rectangle(debug_img, loc, (loc[0] + tw, loc[1] + th), (0, 255, 0), 2)
                    cv2.circle(debug_img, (int(m1_cx), int(m1_cy)), 5, (0, 0, 255), -1)
                    cv2.putText(debug_img, "M1", (loc[0], loc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                ref_m1 = ref_mark_points[0]
                valid = True
                
                for i in range(1, 4):
                    tmpl = cv2.cvtColor(preproc_templates[i], cv2.COLOR_BGR2GRAY)
                    ref_m = ref_mark_points[i]
                    dx, dy = ref_m[0] - ref_m1[0], ref_m[1] - ref_m1[1]
                    exp_cx, exp_cy = m1_cx + dx, m1_cy + dy
                    
                    th_i, tw_i = tmpl.shape
                    search_pad_x, search_pad_y = 200, 200 # พิกเซลที่บวกเพิ่มด้านละ
                    w_box = tw_i + (search_pad_x * 2)
                    h_box = th_i + (search_pad_y * 2)
                    
                    roi_x, roi_y = int(exp_cx - w_box/2), int(exp_cy - h_box/2)
                    roi_rect = (roi_x, roi_y, w_box, h_box)
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + w_box, roi_y + h_box), (255, 0, 0), 2)
                        cv2.putText(debug_img, f"ROI M{i+1}", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
                    loc_i, score_i = find_mark(img_gray, tmpl, roi_rect)
                    
                    if score_i < 0.4:
                        if debug_img is not None:
                            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"debug_align_fail_m{i+1}_{int(time.time()*100)}.jpg")
                            cv2.imwrite(save_path, debug_img)
                        raise ValueError(f"Mark {i+1} not found. Alignment failed.")
                        
                    found_cx = loc_i[0] + tw_i//2
                    found_cy = loc_i[1] + th_i//2
                    found_marks.append([found_cx, found_cy])
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, loc_i, (loc_i[0] + tw_i, loc_i[1] + th_i), (0, 255, 0), 2)
                        cv2.circle(debug_img, (int(found_cx), int(found_cy)), 5, (0, 0, 255), -1)
                        cv2.putText(debug_img, f"M{i+1}", (loc_i[0], loc_i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                if debug_img is not None:
                    base_name = last_mock_image_name if (last_mock_image_name and args.mock_dir) else f"{int(time.time()*100)}"
                    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"debug_align_success_{base_name}.jpg")
                    cv2.imwrite(save_path, debug_img)
                    
                input_marks = np.array(found_marks, dtype=np.float32)
                H, _ = cv2.findHomography(input_marks, target_marks, cv2.RANSAC, 5.0)
                
                if H is None:
                    raise ValueError("Homography Matrix calculation failed.")
                    
                aligned_img = cv2.warpPerspective(frame, H, output_size)
            else:
                aligned_img = frame.copy()
            
            # --- 2. Shadow Removal ---
            if enable_shadow:
                shadow_removed = remove_shadows_divisive(aligned_img, sigma=50)
                if shadow_removed is None:
                    shadow_removed = aligned_img
            else:
                shadow_removed = aligned_img
                
            # --- 3. Grayscale & CLAHE ---
            if enable_gray:
                gray_img = cv2.cvtColor(shadow_removed, cv2.COLOR_BGR2GRAY)
                
                if enable_clahe and not args.disable_clahe:
                    processed_gray = clahe_obj.apply(gray_img)
                else:
                    processed_gray = gray_img
                    
                # TCP expects 3 channels for current send_image encoding / viewing, convert back
                final_surface = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
            else:
                final_surface = shadow_removed
            
            # --- 4. Process Masks & Crops ---
            images_to_send = []
            masked_surface = final_surface.copy()
            
            if enable_crop and mask_config:
                h_out, w_out = masked_surface.shape[:2]
                
                # Map coordinates from original JSON size to current output size
                # Since size changes in this pipeline are due to calibration padding, 
                # the size difference is an exact translation (offset) rather than a linear scale.
                ref_size = mask_config.get("reference_image_size", {})
                ref_w = ref_size.get("width", w_out)
                ref_h = ref_size.get("height", h_out)
                
                offset_x = (w_out - ref_w) // 2
                offset_y = (h_out - ref_h) // 2
                
                mask_regions = mask_config.get("mask_regions", [])
                for region in mask_regions:
                    # Apply absolute offset correctly based on padding differences
                    rx = int(region["x"] + offset_x)
                    ry = int(region["y"] + offset_y)
                    rw = int(region["w"])
                    rh = int(region["h"])
                    
                    # Boundary checks
                    rx = max(0, rx); ry = max(0, ry)
                    rw = min(rw, w_out - rx)
                    rh = min(rh, h_out - ry)
                    
                    if rw > 0 and rh > 0:
                        mark_crop = final_surface[ry:ry+rh, rx:rx+rw]
                        images_to_send.append(mark_crop)
                        
                    # Mask Surface
                    cv2.rectangle(masked_surface, (rx, ry), (rx+rw, ry+rh), (0, 0, 0), -1)
                
            # --- 5. Send ALL Images Sequential Stream Protocol ---
            if args.debug_align:
                # สร้างโฟลเดอร์สำหรับเก็บภาพทดสอบชั่วคราว
                debug_out_dir = os.path.join(base_dir, "logs")
                os.makedirs(debug_out_dir, exist_ok=True)
                base_name = last_mock_image_name if (last_mock_image_name and args.mock_dir) else f"{int(time.time()*100)}"
                cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_masked_surface.jpg"), masked_surface)
            
            # Send the main masked surface first
            total_to_send = 1 + len(images_to_send)
            print(f"INFO: Pre-processing complete. {total_to_send} images prepared for sending (1 surface + {len(images_to_send)} crops).")
            send_image(masked_surface, image_id="masked_surface")
            
            if enable_crop:
                # Followed by all the crops (Should be 6 crops based on user config)
                # The receiver will get id="crop_0", "crop_1", etc.
                for idx, crop in enumerate(images_to_send):
                    crop_id = f"crop_{idx + 1}"
                    if args.debug_align:
                        cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_{crop_id}.jpg"), crop)
                    send_image(crop, image_id=crop_id)
                
            image_queue.task_done()
            
        except ValueError as e:
            if args.mock_dir:
                img_name = last_mock_image_name if last_mock_image_name else "Unknown"
                print(f"⚠️ MOCK WARNING: Skipping image '{img_name}': {e}")
                image_queue.task_done()
            else:
                print(f"CRITICAL ERROR: Pre-processing worker failed: {e}")
                os._exit(1)
        except Exception as e:
            print(f"CRITICAL ERROR: Unexpected Pre-processing worker failure: {e}")
            os._exit(1) # Force exit immediately if there is a fatal error in the thread

def on_mqtt_connect(client, userdata, flags, rc):
    print(f"INFO: Connected to MQTT broker with result code {rc}")
    client.subscribe(MQTT_TOPIC_CMD)
    print(f"INFO: Subscribed to MQTT topic: {MQTT_TOPIC_CMD}")

def on_mqtt_message(client, userdata, msg):
    global capture_triggered, current_width, current_height, picam2, config
    try:
        payload = json.loads(msg.payload.decode())
        print(f"MQTT CMD Received: {payload}")

        if not picam2:
            return

        controls = {}
        config_updated = False
        if "camera_params" not in config:
            config["camera_params"] = {}
            
        # Map JSON payload to Picamera2 controls
        if 'ExposureTime' in payload:
            controls['ExposureTime'] = int(payload['ExposureTime'])
            config["camera_params"]['ExposureTime'] = controls['ExposureTime']
            config_updated = True
        if 'AnalogueGain' in payload:
            controls['AnalogueGain'] = float(payload['AnalogueGain'])
            config["camera_params"]['AnalogueGain'] = controls['AnalogueGain']
            config_updated = True
        if 'ColourGains' in payload:
            # Format expected: [red_gain, blue_gain]
            gains = payload['ColourGains']
            if isinstance(gains, list) and len(gains) == 2:
                controls['ColourGains'] = (float(gains[0]), float(gains[1]))
                config["camera_params"]['ColourGains'] = gains
                config_updated = True
        if 'LensPosition' in payload:
            controls['LensPosition'] = float(payload['LensPosition'])
            config["camera_params"]['LensPosition'] = controls['LensPosition']
            controls['AfMode'] = 0 # 0 sets AfModeEnum.Manual usually required for LensPosition
            config_updated = True
            
        # Additional settings like AfMode
        if 'AfMode' in payload:
            controls['AfMode'] = int(payload['AfMode'])
            config["camera_params"]['AfMode'] = controls['AfMode']
            config_updated = True
            
        if controls:
            picam2.set_controls(controls)
            print(f"INFO: Applied camera controls: {controls}")
            
        if config_updated:
            # Move save_config to a background thread to prevent blocking MQTT loop
            threading.Thread(target=save_config, daemon=True).start()

        # Handle resolution changes
        if 'resolution' in payload:
            res = payload['resolution']
            if isinstance(res, list) and len(res) == 2:
                new_width, new_height = int(res[0]), int(res[1])
                if new_width != current_width or new_height != current_height:
                    with capture_lock:
                        print(f"INFO: Reconfiguring resolution to {new_width}x{new_height}...")
                        picam2.stop()
                        current_width, current_height = new_width, new_height
                        config = picam2.create_preview_configuration(
                            main={'format': 'RGB888', 'size': (current_width, current_height)}
                        )
                        picam2.configure(config)
                        picam2.start()
                        print("INFO: Camera resolution updated.")

        # Capture signal
        if payload.get('action') == 'capture':
            capture_triggered = True


        # System management
        if payload.get('system') == 'restart':
            print("WARNING: Restarting system via MQTT command...")
            os.system("sudo reboot")
            
        if payload.get('system') == 'shutdown':
            print("WARNING: Shutting down system via MQTT command...")
            os.system("sudo halt")

    except json.JSONDecodeError:
        print("ERROR: Invalid JSON received via MQTT")
    except Exception as e:
        print(f"ERROR: Error handling MQTT message: {e}")

def main():
    global picam2, capture_triggered

    # 1. Initialize Camera
    try:
        if args.mock_dir:
            print(f"INFO: Starting in MOCK mode using directory: {args.mock_dir}")
            picam2 = MockCamera(args.mock_dir)
        else:
            picam2 = Picamera2(camera_num=CAMERA_ID)
            cam_config = picam2.create_preview_configuration(
                main={'format': 'RGB888', 'size': (current_width, current_height)}
            )
            picam2.configure(cam_config)
            picam2.start()
            print(f"INFO: Camera started successfully at {current_width}x{current_height}")
            
            # Apply saved parameters
            saved_params = config.get("camera_params", {})
            if saved_params:
                initial_controls = {}
                if 'ExposureTime' in saved_params:
                    initial_controls['ExposureTime'] = int(saved_params['ExposureTime'])
                if 'AnalogueGain' in saved_params:
                    initial_controls['AnalogueGain'] = float(saved_params['AnalogueGain'])
                if 'ColourGains' in saved_params:
                    gains = saved_params['ColourGains']
                    if isinstance(gains, list) and len(gains) == 2:
                         initial_controls['ColourGains'] = (float(gains[0]), float(gains[1]))
                if 'LensPosition' in saved_params:
                    initial_controls['LensPosition'] = float(saved_params['LensPosition'])
                    initial_controls['AfMode'] = 0
                if 'AfMode' in saved_params:
                    initial_controls['AfMode'] = int(saved_params['AfMode'])
                    
                if initial_controls:
                    picam2.set_controls(initial_controls)
                    print(f"INFO: Applied saved camera parameters from config: {initial_controls}")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize camera: {e}")
        os._exit(1) # Exit forcefully with an error code to trigger Systemd Restart

    # 1.5 Initialize Pre-processing (Will stop program if fails)
    try:
        init_preprocessing()
    except Exception as e:
        print(e)
        os._exit(1)
    
    # Start worker thread
    worker = threading.Thread(target=pre_process_worker, daemon=True)
    worker.start()

    # 2. Initialize MQTT
    mqtt_client = mqtt.Client()
    
    if MQTT_USERNAME and MQTT_PASSWORD:
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        print("INFO: MQTT Authentication configured.")
        
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # Use non-blocking loop to ensure main thread remains responsive
        mqtt_client.loop_start() 
    except Exception as e:
        print(f"ERROR: Failed to connect to MQTT broker {MQTT_BROKER}: {e}")
        # Not exiting here; might still want camera running or try connecting later
        # A more robust script could implement MQTT reconnect logic

    print("INFO: Sender Service is running. Waiting for commands...")
    
    # 3. Main Loop
    last_status_time = 0
    last_capture_time = 0
    try:
        while True:
            current_time = time.time()
            if current_time - last_status_time >= 5.0:
                status = {
                    'cpu_temp': round(get_cpu_temperature(), 2),
                    'ram_usage_percent': round(get_ram_usage(), 2),
                    'cpu_usage_percent': round(get_cpu_usage(), 2),
                    'resolution': [current_width, current_height],
                }
                mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status))
                last_status_time = current_time

            # Trigger a capture either if forced via MQTT or via continuous stream interval
            should_stream = CONTINUOUS_STREAM and (current_time - last_capture_time >= STREAM_INTERVAL)

            if capture_triggered or should_stream:
                with capture_lock:
                    if capture_triggered:
                        print("INFO: Manual capture triggered.")
                        capture_triggered = False
                        
                    try:
                        frame = picam2.capture_array()
                        # Drop old frame if queue is full to maintain real-time
                        if image_queue.full():
                            try:
                                image_queue.get_nowait()
                                image_queue.task_done()
                                print("WARNING: Dropped frame due to full queue.")
                            except: pass
                        image_queue.put(frame)
                        last_capture_time = time.time()
                    except Exception as e:
                        print(f"ERROR: Capture failed: {e}")
            
            # Sleep briefly to yield CPU and avoid 100% usage loop
            time.sleep(LOOP_DELAY)
            
    except KeyboardInterrupt:
        print("INFO: Stopping script (KeyboardInterrupt)...")
    finally:
        print("INFO: Cleaning up resources...")
        mqtt_client.loop_stop()
        if picam2:
            picam2.stop()
        if tcp_socket:
            tcp_socket.close()

if __name__ == "__main__":
    main()