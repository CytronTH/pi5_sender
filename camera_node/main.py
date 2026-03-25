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
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'src'))

from src.image_alignment import calculate_canonical_targets, find_mark
from src.shadow_removal import remove_shadows_divisive
from src.grayscale_filter import cv2 as dummy_cv2
import src.image_cropping as cwb
from src.sftp_handler import SFTPHandler
from src.config_manager import ConfigManager
from src.tcp_sender import TCPSender
from src.mqtt_handler import MQTTHandler

# --- Configuration Loader ---
parser = argparse.ArgumentParser(description="Camera Sender Script")
parser.add_argument('-c', '--config', type=str, default=os.path.join(base_dir, 'configs', 'config.json'), help='Path to config file')
parser.add_argument('--mock_dir', type=str, default=None, help='Directory containing mock images for offline testing')
parser.add_argument('--debug_align', action='store_true', help='Save visualization of the alignment process to disk')
parser.add_argument('--disable_clahe', action='store_true', help='Disable CLAHE enhancement and use raw grayscale')
args = parser.parse_args()

config_mgr = ConfigManager(args.config)
config = config_mgr.get_all()

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
        logger.info(f"Initialized MockCamera with {len(self.images)} images from {image_dir}")

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

# init_preprocessing has been replaced by ImagePipeline

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
        config_mgr.save_all(config)
        logger.info(f"Saved updated configuration to {args.config}")
    except Exception as e:
        logger.error(f"Failed to save config to {args.config}: {e}")

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

def pre_process_worker():
    """ Worker thread to process images from the queue and send them. """
    import os, time, json
    from src.image_pipeline import ImagePipeline
    
    logger.info(f"Pre-processing worker thread started.")
    
    # Initialize the new pipeline
    pipeline = ImagePipeline(CAMERA_ID, base_dir)
    
    try:
        prep_config = config.get("preprocessing", {})
        enable_align = prep_config.get("enable_alignment", True)
        enable_crop = prep_config.get("enable_box_cropping", True)
        pipeline.load_configs(enable_align, enable_crop)
        tcp_ip = config.get("tcp", {}).get("ip", "10.10.10.199")
        tcp_port = config.get("tcp", {}).get("port", 8080)
        tcp_sender = TCPSender(tcp_ip, tcp_port)
        
        # Clear any stale monitor data from previous runs to prevent ghosting
        stale_monitor = f"/dev/shm/tcp_monitor_cam{CAMERA_ID}.json"
        if os.path.exists(stale_monitor):
            os.remove(stale_monitor)
            logger.info(f"Cleared stale monitor data: {stale_monitor}")
    except Exception as e:
        print(f"CRITICAL ERROR during pipeline initialization: {e}")
        sys.exit(1)
        
    while True:
        try:
            frame = image_queue.get()
            prep_config = config.get("preprocessing", {})
            disable_clahe = args.disable_clahe
            
            # Use mock name for debug writes if available
            m_name = last_mock_image_name if args.mock_dir else None
            
            # Process the frame through the encapsulated pipeline
            images_to_send = pipeline.process_frame(
                frame=frame,
                prep_config=prep_config,
                debug=args.debug_align,
                mock_name=m_name,
                disable_clahe=disable_clahe
            )
            
            # Send the images sequentially
            logger.info(f"Pre-processing complete. {len(images_to_send)} images prepared for sending.")
            
            monitor_images = []
            for img_id, img_data in images_to_send:
                h_img, w_img = img_data.shape[:2]
                tcp_sender.send_image(img_data, image_id=img_id, jpeg_quality=JPEG_QUALITY, width=w_img, height=h_img)
                
                # Encode for WebUI Monitor (lightweight)
                try:
                    ret, buffer = cv2.imencode('.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    if ret:
                        import base64
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        monitor_images.append({
                            "id": img_id,
                            "data": b64_str
                        })
                except Exception as ignore:
                    pass
                    
            if monitor_images:
                try:
                    import os, time, json
                    monitor_data = {
                        "timestamp": time.time(),
                        "images": monitor_images
                    }
                    tmp_path = f"/dev/shm/tcp_monitor_cam{CAMERA_ID}.json.tmp"
                    final_path = f"/dev/shm/tcp_monitor_cam{CAMERA_ID}.json"
                    with open(tmp_path, "w") as f:
                        json.dump(monitor_data, f)
                    os.rename(tmp_path, final_path)
                except Exception as ignore:
                    pass
                
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
            sys.exit(1)


def run_sftp_transfer(files_to_upload, sftp_config):
    def _transfer():
        logger.info(f"Starting SFTP background transfer for {len(files_to_upload)} files...")
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
    logger.info(f"Saved capture for SFTP: {filepath}")
    
    pending_transfers.append(filepath)
    logger.info(f"Pending transfers: {len(pending_transfers)}/{batch_size}")
    
    if len(pending_transfers) >= batch_size:
        logger.info(f"SFTP Batch size ({batch_size}) reached. Initiating upload...")
        files_to_upload = list(pending_transfers)
        pending_transfers.clear()
        run_sftp_transfer(files_to_upload, sftp_config)

def main():
    global picam2, capture_triggered, current_width, current_height, config

    # 1. Initialize Camera
    try:
        if args.mock_dir:
            logger.info(f"Starting in MOCK mode using directory: {args.mock_dir}")
            picam2 = MockCamera(args.mock_dir)
        else:
            logger.debug(f"Initializing Picamera2 with CAMERA_ID={CAMERA_ID}")
            picam2 = Picamera2(camera_num=CAMERA_ID)
            cam_config = picam2.create_preview_configuration(
                main={'format': 'RGB888', 'size': (current_width, current_height)}
            )
            picam2.configure(cam_config)
            picam2.start()
            logger.info(f"Camera started successfully at {current_width}x{current_height}")
            
            # Apply saved generic parameters
            controls = config.get("controls", {})
            if controls:
                try:
                    picam2.set_controls(controls)
                    logger.info(f"Applied camera controls from config: {controls}")
                except Exception as ce:
                    logger.warning(f"Failed to apply some camera controls: {ce}")
    except Exception as e:
        logger.critical(f"Failed to initialize camera: {e}")
        sys.exit(1) # Exit forcefully with an error code to trigger Systemd Restart

    # 1.5 Initialize Pre-processing (Will stop program if fails)
    pass # init_preprocessing() handled by worker
    
    # Start worker thread
    worker = threading.Thread(target=pre_process_worker, daemon=True)
    worker.start()

    # 2. Initialize MQTT
    def apply_controls(controls):
        if picam2:
            try:
                picam2.set_controls(controls)
                logger.info(f"Applied camera controls from MQTT: {controls}")
            except Exception as ce:
                logger.warning(f"Failed to apply controls: {ce}")

    def change_resolution(new_width, new_height):
        global current_width, current_height, config
        if new_width != current_width or new_height != current_height:
            with capture_lock:
                logger.info(f"Reconfiguring resolution to {new_width}x{new_height}...")
                if picam2:
                    picam2.stop()
                current_width, current_height = new_width, new_height
                try:
                    c = picam2.create_preview_configuration(
                        main={'format': 'RGB888', 'size': (current_width, current_height)}
                    )
                    picam2.configure(c)
                    picam2.start()
                    logger.info(f"Camera resolution updated.")
                except Exception as ce:
                    logger.error(f"Failed switching resolution: {ce}")

    def trigger_capture():
        global capture_triggered
        capture_triggered = True
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\\n=======================================================")
        print(f"[{current_time}] 🎯 MQTT TRIGGER RECEIVED: 'capture'")
        print(f"=======================================================\\n")

    callbacks = {
        'apply_controls': apply_controls,
        'change_resolution': change_resolution,
        'trigger_capture': trigger_capture
    }

    mqtt_handler = MQTTHandler(config_mgr, callbacks)
    mqtt_handler.setup()
    logger.info(f"Sender Service is running. Waiting for commands...")
    
    # 3. Main Loop
    last_status_time = 0
    last_capture_time = 0
    try:
        while True:
            current_time = time.time()
            if current_time - last_status_time >= 5.0:
                try:
                    status = {
                        'camera_id': CAMERA_ID,
                        'cpu_temp': round(get_cpu_temperature(), 2),
                        'ram_usage_percent': round(get_ram_usage(), 2),
                        'cpu_usage_percent': round(get_cpu_usage(), 2),
                        'resolution': [current_width, current_height],
                        'camera_params': config.get("camera_params", {})
                    }
                    mqtt_handler.publish_status(status)
                except Exception as e:
                    logger.error(f"Failed to publish status: {e}")
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
                        logger.info(f"Manual capture triggered.")
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
                                    logger.warning(f"Dropped frame due to full queue.")
                                except: pass
                            image_queue.put(frame)
                            
                        last_capture_time = time.time()
                    except Exception as e:
                        logger.error(f"Capture failed: {e}")
            
            # Sleep briefly to yield CPU and avoid 100% usage loop
            time.sleep(LOOP_DELAY)
            
    except KeyboardInterrupt:
        logger.info(f"Stopping script (KeyboardInterrupt)...")
    finally:
        logger.info(f"Cleaning up resources...")
        mqtt_handler.stop()
        if picam2:
            picam2.stop()
        if tcp_socket:
            tcp_socket.close()

if __name__ == "__main__":
    main()