import paho.mqtt.client as mqtt
import json
import os
import threading
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class MQTTHandler:
    """Handles MQTT communication for pi5_sender. Translates remote commands to local callbacks."""
    def __init__(self, config_mgr, callbacks):
        self.config_mgr = config_mgr
        self.callbacks = callbacks
        self.client = mqtt.Client()
        self.topic_cmd = None
        self.topic_status = None

    def setup(self):
        config = self.config_mgr.get_all()
        mqtt_cfg = config.get("mqtt", {})
        
        broker = mqtt_cfg.get("broker", "localhost")
        port = mqtt_cfg.get("port", 1883)
        self.topic_cmd = mqtt_cfg.get("topic_cmd", "camera/command")
        self.topic_status = mqtt_cfg.get("topic_status", "camera/status")
        
        import socket
        hostname = socket.gethostname()
        if self.topic_cmd.startswith("wf52/"):
            self.topic_cmd = f"{hostname}/" + self.topic_cmd.split("/", 1)[1]
        elif "{hostname}" in self.topic_cmd:
            self.topic_cmd = self.topic_cmd.replace("{hostname}", hostname)
            
        self.topic_status = f"{hostname}/status"
        
        username = mqtt_cfg.get("username")
        password = mqtt_cfg.get("password")
        if username and password:
            self.client.username_pw_set(username, password)
            
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        try:
            self.client.connect(broker, port, 60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT Broker at {broker}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker {broker}: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.topic_cmd)
            logger.info(f"Subscribed to MQTT topic: {self.topic_cmd}")
        else:
            logger.error(f"MQTT Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            print(f"MQTT CMD Received: {payload}")
            
            config = self.config_mgr.get_all()
            config_updated = False
            controls = {}
            if "camera_params" not in config:
                config["camera_params"] = {}
                
            if 'ExposureTime' in payload:
                controls['ExposureTime'] = int(payload['ExposureTime'])
                config["camera_params"]['ExposureTime'] = controls['ExposureTime']
                config_updated = True
            if 'AnalogueGain' in payload:
                controls['AnalogueGain'] = float(payload['AnalogueGain'])
                config["camera_params"]['AnalogueGain'] = controls['AnalogueGain']
                config_updated = True
            if 'ColourGains' in payload:
                gains = payload['ColourGains']
                if isinstance(gains, list) and len(gains) == 2:
                    controls['ColourGains'] = (float(gains[0]), float(gains[1]))
                    config["camera_params"]['ColourGains'] = gains
                    config_updated = True
            if 'LensPosition' in payload:
                controls['LensPosition'] = float(payload['LensPosition'])
                config["camera_params"]['LensPosition'] = controls['LensPosition']
                controls['AfMode'] = 0 
                config_updated = True
            if 'AfMode' in payload:
                controls['AfMode'] = int(payload['AfMode'])
                config["camera_params"]['AfMode'] = controls['AfMode']
                config_updated = True
                
            if controls and 'apply_controls' in self.callbacks:
                self.callbacks['apply_controls'](controls)
                
            if config_updated:
                threading.Thread(target=self.config_mgr.save_all, args=(config,), daemon=True).start()
                self.publish_status({'camera_params': config.get("camera_params", {})})

            if 'resolution' in payload and 'change_resolution' in self.callbacks:
                res = payload['resolution']
                if isinstance(res, list) and len(res) == 2:
                    self.callbacks['change_resolution'](int(res[0]), int(res[1]))

            if payload.get('action') == 'capture' and 'trigger_capture' in self.callbacks:
                self.callbacks['trigger_capture']()

            if payload.get('system') == 'restart':
                logger.warning(f"Restarting system via MQTT command...")
                os.system("sudo reboot")
            if payload.get('system') == 'shutdown':
                logger.warning(f"Shutting down system via MQTT command...")
                os.system("sudo halt")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received via MQTT")
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")

    def publish_status(self, status_dict):
        """Publishes a dictionary to the status topic as JSON."""
        try:
            self.client.publish(self.topic_status, json.dumps(status_dict))
        except Exception as e:
            pass

    def stop(self):
        self.client.loop_stop()
