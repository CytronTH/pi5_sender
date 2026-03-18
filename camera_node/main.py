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

from src.image_alignment import calculate_canonical_targets, find_mark
from src.shadow_removal import remove_shadows_divisive
from src.grayscale_filter import cv2 as dummy_cv2
import src.image_cropping as cwb
from src.sftp_handler import SFTPHandler

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
    print(f"CRITICAL: {args.config} not found. Generating fallback configuration for initial calibration!")
    config = {
        "tcp": {
            "ip": "10.10.10.199",
            "port": 8080 if "cam0" in args.config else 8081
        },
        "mqtt": {
            "broker": "wfmain.local",
            "port": 1883,
            "topic_cmd": f"{socket.gethostname()}/w/command",
            "topic_status": f"{socket.gethostname()}/w/status"
        },
        "camera": {
            "id": 0 if "cam0" in args.config else 1,
            "default_width": 2304,
            "default_height": 1296,
            "jpeg_quality": 90,
            "continuous_stream": False,
            "stream_interval": 1.0,
            "loop_delay": 0.05
        },
        "preprocessing": {
            "enable_alignment": False,
            "enable_shadow_removal": False,
            "enable_pre_crop": False,
            "enable_grayscale": False,
            "enable_clahe": False,
            "enable_box_cropping": False
        }
    }
    # Auto-save the fallback config so it exists for next time
    os.makedirs(os.path.dirname(os.path.abspath(args.config)), exist_ok=True)
    try:
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"INFO: Saved fallback configuration to {args.config}")
    except Exception as e:
        print(f"ERROR: Failed to save fallback config: {e}")

TCP_IP = config.get("tcp", {}).get("ip", "10.10.10.199")
TCP_PORT = config.get("tcp", {}).get("port", 8080)
MQTT_BROKER = config.get("mqtt", {}).get("broker", "10.10.10.199")
MQTT_PORT = config.get("mqtt", {}).get("port", 1883)
MQTT_TOPIC_CMD = config.get("mqtt", {}).get("topic_cmd", "camera/command")
MQTT_TOPIC_STATUS = config.get("mqtt", {}).get("topic_status", "camera/status")

hostname = socket.gethostname()
# Replace hardcoded wf52 prefix with actual hostname to prevent conflicts on new boards
if MQTT_TOPIC_CMD.startswith("wf52/"):
    MQTT_TOPIC_CMD = f"{hostname}/" + MQTT_TOPIC_CMD.split("/", 1)[1]
elif "{hostname}" in MQTT_TOPIC_CMD:
    MQTT_TOPIC_CMD = MQTT_TOPIC_CMD.replace("{hostname}", hostname)

# Force status topic to be <hostname>/status as requested
MQTT_TOPIC_STATUS = f"{hostname}/status"

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
pending_transfers = []

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
            preproc_config, preproc_templates = cwb.load_calibration(f"cam{CAMERA_ID}")
            
            if len(preproc_config.get("calibration_marks", [])) > 1:
                target_marks, output_size = calculate_canonical_targets(preproc_config)
            else:
                target_marks = None
                output_size = None
            
            ref_mark_points = np.array([
                [m.get("center_x", m["x"]), m.get("center_y", m["y"])] 
                for m in preproc_config["calibration_marks"]
            ], dtype=np.float32)
            
            print(f"INFO: Alignment Targets calculated. Output size: {output_size}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: Failed to load calibration config: {e}. Program will terminate.")
        
    # 2. Load Mask Config
    if enable_crop:
        print("INFO: Loading Mask Config...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mask_config_path = os.path.join(base_dir, "configs", f"cam{CAMERA_ID}_crop_regions.json")
        
        try:
            with open(mask_config_path, "r") as f:
                mask_config = json.load(f)
            print(f"INFO: Loaded mask config with {len(mask_config.get('mask_regions', []))} regions.")
        except FileNotFoundError:
            print(f"WARNING: Mask config not found at {mask_config_path}. Box cropping will be skipped.")
            mask_config = None
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
        print(f"INFO: Connected to TCP server at {TCP_IP}:{TCP_PORT}", flush=True)
        return True
    except socket.timeout:
        print("ERROR: TCP Connection timed out.", flush=True)
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
        print(f"INFO: Sent {image_id}: {current_width}x{current_height} ({img_size} bytes)", flush=True)
        
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
            
            # Check if we should fallback to just sending/saving raw images.
            # This happens if:
            # 1. We want alignment/cropping but the configs were not found
            # 1. We want alignment but the calibration config is missing
            # 2. We want cropping but the mask config is missing, AND alignment is disabled or missing
            # 3. Both alignment and cropping are explicitly disabled (e.g. from Fallback Config)
            missing_align = enable_align and not preproc_config
            missing_crop = enable_crop and not mask_config
            
            # If alignment is enabled and configured, we can still align even if crop is missing.
            # But if BOTH the configured features are missing, or we are in calibration mode, send raw.
            calibration_mode_active = (not enable_align) and (not enable_crop)
            fallback_to_raw = calibration_mode_active or (missing_align and missing_crop)
            
            # If alignment is missing, we cannot do ANYTHING (alignment is prereq for crop)
            if missing_align and enable_align:
                fallback_to_raw = True
            
            if fallback_to_raw:
                if not calibration_mode_active:
                    print("INFO: Sending raw image for calibration (Missing JSON configs)")
                else:
                    print("INFO: Sending raw image for calibration (Calibration Mode Active)")
                    
                send_image(frame, image_id="raw_image")
                
                # Save a high-quality copy locally for offline calibration via SSH
                calib_out_dir = os.path.join(base_dir, "logs")
                os.makedirs(calib_out_dir, exist_ok=True)
                cam_id_str = f"cam{CAMERA_ID}"
                save_path = os.path.join(calib_out_dir, f"{cam_id_str}_calibration_target.jpg")
                cv2.imwrite(save_path, frame)
                print(f"INFO: Saved local calibration image to {save_path}")
                
                image_queue.task_done()
                continue
                
        # --- 1. Alignment & Cropping ---
            if enable_align:
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found_marks = []
                
                # Create a debug image if requested
                debug_img = frame.copy() if args.debug_align else None
                
                tmpl1 = cv2.cvtColor(preproc_templates[0], cv2.COLOR_BGR2GRAY)
                loc, score = find_mark(img_gray, tmpl1)
                
                if score < 0.2:
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
                
                if is_single_mark:
                    # === SINGLE MARK CENTER CROP (Camera 0) ===
                    padding = preproc_config.get("padding", 50)
                    mark_w = preproc_config["calibration_marks"][0].get("width", tw)
                    mark_h = preproc_config["calibration_marks"][0].get("height", th)
                    
                    crop_w = int(mark_w + (padding * 2))
                    crop_h = int(mark_h + (padding * 2))
                    
                    start_x = int(m1_cx - (crop_w / 2))
                    start_y = int(m1_cy - (crop_h / 2))
                    
                    h_img, w_img = frame.shape[:2]
                    start_x = max(0, start_x)
                    start_y = max(0, start_y)
                    end_x = min(w_img, start_x + crop_w)
                    end_y = min(h_img, start_y + crop_h)
                    
                    aligned_img = frame[start_y:end_y, start_x:end_x]
                    
                    # Update output_size dynamically for downstream shadow/mask coordinates mapping
                    output_size = (end_x - start_x, end_y - start_y)
                    
                    if debug_img is not None:
                        cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
                        base_name = last_mock_image_name if (last_mock_image_name and args.mock_dir) else f"{int(time.time()*100)}"
                        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"debug_align_success_{base_name}.jpg")
                        cv2.imwrite(save_path, debug_img)
                        
                else:
                    # === MULTI-MARK ALIGNMENT (Camera 1) ===
                    ref_m1 = ref_mark_points[0]
                    valid = True
                    
                    for i in range(1, len(ref_mark_points)):
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
                    
                    if len(input_marks) == 4:
                        H, _ = cv2.findHomography(input_marks, target_marks, cv2.RANSAC, 5.0)
                        if H is None:
                            raise ValueError("Homography Matrix calculation failed.")
                        aligned_img = cv2.warpPerspective(frame, H, output_size)
                    elif len(input_marks) == 2:
                        M, _ = cv2.estimateAffinePartial2D(input_marks, target_marks)
                        if M is None:
                            raise ValueError("Affine Matrix calculation failed.")
                        aligned_img = cv2.warpAffine(frame, M, output_size)
                    else:
                        raise ValueError(f"Unsupported number of marks for alignment: {len(input_marks)}")
            else:
                aligned_img = frame.copy()
            
            # --- 2. Shadow Removal ---
            if enable_shadow:
                shadow_removed = remove_shadows_divisive(aligned_img, sigma=50)
                if shadow_removed is None:
                    shadow_removed = aligned_img
            else:
                shadow_removed = aligned_img
                
            # --- 2.5 Pre-Crop ---
            c_top = c_bottom = c_left = c_right = 0
            enable_pre_crop = prep_config.get("enable_pre_crop", False)
            if enable_pre_crop:
                crop_cfg = prep_config.get("pre_crop", {})
                c_top = int(crop_cfg.get("top", 0))
                c_bottom = int(crop_cfg.get("bottom", 0))
                c_left = int(crop_cfg.get("left", 0))
                c_right = int(crop_cfg.get("right", 0))
                
                h, w = shadow_removed.shape[:2]
                
                c_top = max(0, min(c_top, h - 1))
                c_bottom = max(0, min(c_bottom, h - 1 - c_top))
                c_left = max(0, min(c_left, w - 1))
                c_right = max(0, min(c_right, w - 1 - c_left))
                
                pre_cropped = shadow_removed[c_top:h-c_bottom, c_left:w-c_right]
            else:
                pre_cropped = shadow_removed

            # --- 3. Grayscale & CLAHE ---
            if enable_gray:
                gray_img = cv2.cvtColor(pre_cropped, cv2.COLOR_BGR2GRAY)
                
                if enable_clahe and not args.disable_clahe:
                    processed_gray = clahe_obj.apply(gray_img)
                else:
                    processed_gray = gray_img
                    
                # TCP expects 3 channels for current send_image encoding / viewing, convert back
                final_surface = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
            else:
                final_surface = pre_cropped
            
            # --- 4. Process Masks & Crops ---
            images_to_send = []
            masked_surface = final_surface.copy()
            
            if enable_crop and mask_config:
                h_out, w_out = masked_surface.shape[:2]
                
                # Map coordinates from original JSON size to current output size
                # Reference size here should be compared directly to the current image size (w_out, h_out)
                # without caring if it was pre-cropped or not. Users want coordinates to remain fixed to the frame.
                ref_size = mask_config.get("reference_image_size", {})
                ref_w = ref_size.get("width", w_out)
                ref_h = ref_size.get("height", h_out)
                
                offset_x = (w_out - ref_w) // 2
                offset_y = (h_out - ref_h) // 2
                
                mask_regions = mask_config.get("mask_regions", [])
                for region in mask_regions:
                    # Apply absolute offset correctly based on padding differences 
                    # from the mask config reference, ignoring any pre_crop shift.
                    rx = int(region["x"] + offset_x)
                    ry = int(region["y"] + offset_y)
                    rw = int(region.get("width", region.get("w", 0)))
                    rh = int(region.get("height", region.get("h", 0)))
                    
                    # Boundary checks
                    # Map rx, ry to inside the image safely
                    crop_x1 = max(0, rx)
                    crop_y1 = max(0, ry)
                    crop_x2 = min(w_out, rx + rw)
                    crop_y2 = min(h_out, ry + rh)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        mark_crop = final_surface[crop_y1:crop_y2, crop_x1:crop_x2]
                        # Support sending with the name specified in the config if available
                        raw_id = region.get("id", f"mask_{len(images_to_send) + 1}")
                        crop_id = raw_id.replace("mask_", "crop_") if raw_id.startswith("mask_") else raw_id
                        images_to_send.append((crop_id, mark_crop))
                        
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
            if enable_pre_crop:
                total_to_send += 1
                
            print(f"INFO: Pre-processing complete. {total_to_send} images prepared for sending.", flush=True)
            send_image(masked_surface, image_id="raw_image")
            
            if enable_pre_crop:
                # Send the pre-cropped version before the small crops
                if args.debug_align:
                    cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_pre_crop.jpg"), pre_cropped)
                # Currently pre_cropped is generated in Step 2.5
                send_image(pre_cropped, image_id="pre_crop")
            
            if enable_crop:
                # Followed by all the crops (Should be 6 crops based on user config)
                # The receiver will get id="mask_1", "mask_2", etc as defined in masks.json
                for crop_id, crop in images_to_send:
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
                print(f"⚠️ ALIGNMENT WARNING: Pre-processing skipped this frame: {e}")
                image_queue.task_done()
        except Exception as e:
            print(f"CRITICAL ERROR: Unexpected Pre-processing worker failure: {e}")
            os._exit(1) # Force exit immediately if there is a fatal error in the thread

def run_sftp_transfer(files_to_upload, sftp_config):
    def _transfer():
        print(f"INFO: Starting SFTP background transfer for {len(files_to_upload)} files...")
        handler = SFTPHandler(sftp_config)
        handler.upload_files(files_to_upload)
    threading.Thread(target=_transfer, daemon=True).start()

def handle_sftp_capture(frame):
    global pending_transfers
    sftp_config = config.get("sftp", {})
    batch_size = int(sftp_config.get("batch_size", 10))
    save_dir = os.path.join(base_dir, "logs", "captures")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Add a fractional part to avoid overwriting if triggered multiple times a second
    frac = int((time.time() % 1) * 1000)
    filename = f"cam{CAMERA_ID}_{timestamp}_{frac:03d}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Save the frame
    cv2.imwrite(filepath, frame)
    print(f"INFO: Saved capture for SFTP: {filepath}")
    
    pending_transfers.append(filepath)
    print(f"INFO: Pending transfers: {len(pending_transfers)}/{batch_size}")
    
    if len(pending_transfers) >= batch_size:
        print(f"INFO: SFTP Batch size ({batch_size}) reached. Initiating upload...")
        files_to_upload = list(pending_transfers)
        pending_transfers.clear()
        run_sftp_transfer(files_to_upload, sftp_config)

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
            
            # Publish updated parameters immediately
            updated_params = {
                'camera_params': config.get("camera_params", {})
            }
            client.publish(MQTT_TOPIC_STATUS, json.dumps(updated_params))

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
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n=======================================================")
            print(f"[{current_time}] 🎯 MQTT TRIGGER RECEIVED: 'capture'")
            print(f"=======================================================\n")


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
            
            # Apply saved generic parameters
            controls = config.get("controls", {})
            if controls:
                try:
                    picam2.set_controls(controls)
                    print(f"INFO: Applied camera controls from config: {controls}")
                except Exception as ce:
                    print(f"WARNING: Failed to apply some camera controls: {ce}")
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
                    'camera_id': CAMERA_ID,
                    'cpu_temp': round(get_cpu_temperature(), 2),
                    'ram_usage_percent': round(get_ram_usage(), 2),
                    'cpu_usage_percent': round(get_cpu_usage(), 2),
                    'resolution': [current_width, current_height],
                    'camera_params': config.get("camera_params", {})
                }
                mqtt_client.publish(MQTT_TOPIC_STATUS, json.dumps(status))
                last_status_time = current_time

            sftp_enabled = config.get("sftp", {}).get("enabled", False)
            should_stream = CONTINUOUS_STREAM and (current_time - last_capture_time >= STREAM_INTERVAL)
            
            # If SFTP is enabled, we only care about manual triggers (not continuous TCP streaming)
            if sftp_enabled:
                should_stream = False

            if capture_triggered or should_stream:
                with capture_lock:
                    is_manual = False
                    if capture_triggered:
                        print("INFO: Manual capture triggered.")
                        is_manual = True
                        capture_triggered = False
                        
                    try:
                        frame = picam2.capture_array()
                        
                        if sftp_enabled and is_manual:
                            handle_sftp_capture(frame)
                        elif not sftp_enabled:
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