#!/usr/bin/env python3
import os
import json
import socket
import time
import argparse
import paho.mqtt.client as mqtt
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



base_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="MQTT System Controller")
parser.add_argument('-c', '--config', type=str, default=None, help='Path to config file')
args = parser.parse_args()

if not args.config:
    if os.path.exists(os.path.join(base_dir, 'configs', 'config_cam0.json')):
        args.config = os.path.join(base_dir, 'configs', 'config_cam0.json')
    elif os.path.exists(os.path.join(base_dir, 'configs', 'config_cam1.json')):
        args.config = os.path.join(base_dir, 'configs', 'config_cam1.json')
    else:
        args.config = os.path.join(base_dir, 'configs', 'config.json')

try:
    with open(args.config, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error(f"{args.config} not found. Cannot load MQTT configuration.")
    sys.exit(1)

MQTT_BROKER = config.get("mqtt", {}).get("broker", "localhost")
MQTT_PORT = config.get("mqtt", {}).get("port", 1883)
MQTT_USERNAME = config.get("mqtt", {}).get("username", None)
MQTT_PASSWORD = config.get("mqtt", {}).get("password", None)

MQTT_TOPIC_CMD = config.get("mqtt", {}).get("topic_cmd", "camera/command")

hostname = socket.gethostname()
# Unconditionally use the hostname as the base topic
# Overrides whatever is in the config to ensure script portability
MQTT_TOPIC_CMD = f"{hostname}/c/command"
MQTT_TOPIC_SYS = f"{hostname}/sys/command"
MQTT_TOPIC_SYS_STATUS = f"{hostname}/sys/status"

import threading

def mqtt_status_publisher():
    while True:
        try:
            if client.is_connected():
                status_payload = {
                    "system": "online",
                    "hostname": hostname
                }
                client.publish(MQTT_TOPIC_SYS_STATUS, json.dumps(status_payload), retain=True)
                
        except Exception as e:
            logger.error(f"Failed to publish status: {e}")
        time.sleep(10)

def on_client_connect(client, userdata, flags, reason_code, properties):
    logger.info(f"System Controller connected to MQTT broker with result code {reason_code}")
    client.subscribe(MQTT_TOPIC_CMD)
    client.subscribe(MQTT_TOPIC_SYS)
    logger.info(f"Subscribed to camera topic: {MQTT_TOPIC_CMD}")
    logger.info(f"Subscribed to system topic: {MQTT_TOPIC_SYS}")
    
    # Send an initial connect status immediately
    status_payload = {"system": "online", "hostname": hostname}
    client.publish(MQTT_TOPIC_SYS_STATUS, json.dumps(status_payload), retain=True)

def on_client_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        # Only process system commands
        if 'system' in payload:
            print(f"System CMD Received: {payload} on topic {msg.topic}")
            command = payload.get('system')
            
            if command == 'restart':
                logger.warning(f"Restarting system via MQTT command...")
                
                # Send closing status
                closing_payload = {"system": "restarting", "hostname": hostname}
                client.publish(MQTT_TOPIC_SYS_STATUS, json.dumps(closing_payload), retain=True)
                time.sleep(0.5)
                
                os.system("sudo reboot")
            elif command == 'shutdown':
                logger.warning(f"Shutting down system via MQTT command...")
                
                # Send closing status
                closing_payload = {"system": "offline", "hostname": hostname}
                client.publish(MQTT_TOPIC_SYS_STATUS, json.dumps(closing_payload), retain=True)
                time.sleep(0.5)
                
                os.system("sudo halt")
                
    except json.JSONDecodeError:
        pass # Ignore non-JSON messages quietly
    except Exception as e:
        logger.error(f"Error handling MQTT message: {e}")

import signal
import sys

def graceful_exit(signum, frame):
    logger.info(f"Received signal {signum}, disconnecting MQTT gracefully...")
    try:
        closing_payload = {"system": "offline", "hostname": hostname}
        client.publish(MQTT_TOPIC_SYS_STATUS, json.dumps(closing_payload), retain=True)
        time.sleep(0.5)
        client.disconnect()
    except:
        pass
    sys.exit(0)

if __name__ == "__main__":
    logger.info(f"Starting MQTT System Controller for {hostname}")
    # Use a static client ID so reconnects cancel pending LWTs
    client_id = f"{hostname}_sys_ctrl"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    
    # Configure Last Will and Testament (LWT) for unexpected disconnects
    lwt_payload = json.dumps({"system": "offline", "hostname": hostname})
    client.will_set(MQTT_TOPIC_SYS_STATUS, lwt_payload, retain=True)
    
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
    client.on_connect = on_client_connect
    client.on_message = on_client_message
    
    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGINT, graceful_exit)
    
    # Start status publisher thread
    threading.Thread(target=mqtt_status_publisher, daemon=True).start()
    
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            logger.warning(f"MQTT connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    client.loop_forever()
