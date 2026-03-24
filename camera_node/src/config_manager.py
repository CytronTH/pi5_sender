import json
import os
import threading
import socket
from src.constants import *
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class ConfigManager:
    """Thread-safe manager for reading and writing JSON configuration files."""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, config_path):
        """Implement Singleton pattern per config path to ensure locks are shared across threads."""
        abs_path = os.path.abspath(config_path)
        with cls._lock:
            if abs_path not in cls._instances:
                instance = super(ConfigManager, cls).__new__(cls)
                instance._init(abs_path)
                cls._instances[abs_path] = instance
        return cls._instances[abs_path]
    
    def _init(self, config_path):
        self.config_path = config_path
        self.file_lock = threading.RLock()
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        self.config = self._load()

    def _get_fallback_config(self):
        """Generates a fallback configuration if none exists."""
        cam_id = 0 if "cam0" in self.config_path else 1
        hostname = socket.gethostname()
        
        return {
            "tcp": {
                "ip": DEFAULT_TCP_IP,
                "port": DEFAULT_TCP_PORT_CAM0 if cam_id == 0 else DEFAULT_TCP_PORT_CAM1
            },
            "mqtt": {
                "broker": DEFAULT_MQTT_BROKER,
                "port": DEFAULT_MQTT_PORT,
                "topic_cmd": f"{hostname}/w/command",
                "topic_status": f"{hostname}/status"
            },
            "camera": {
                "id": cam_id,
                "default_width": DEFAULT_CAMERA_WIDTH,
                "default_height": DEFAULT_CAMERA_HEIGHT,
                "jpeg_quality": DEFAULT_JPEG_QUALITY,
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

    def _load(self):
        """Loads config from disk, or generates fallback if not found."""
        with self.file_lock:
            if not os.path.exists(self.config_path):
                logger.critical(f"{self.config_path} not found. Generating fallback configuration!")
                config = self._get_fallback_config()
                self._save(config)
                return config
                
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {self.config_path}: {e}")
                return self._get_fallback_config()

    def _save(self, config_dict):
        """Saves a dictionary to disk."""
        with self.file_lock:
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                self.config = config_dict
                return True
            except Exception as e:
                logger.error(f"Failed to save config to {self.config_path}: {e}")
                return False

    def get_all(self):
        """Returns a copy of the entire current configuration."""
        with self.file_lock:
            # Re-read from disk to ensure we have the latest if modified externally (e.g. by WebUI via another process)
            return self._load()

    def get(self, section, key=None, default=None):
        """Gets a configuration section or key. Highly recommended for thread-safety."""
        config = self.get_all()
        sec = config.get(section, {})
        if key is None:
            return sec if sec is not None else default
        return sec.get(key, default)

    def set(self, section, key, value):
        """Sets a configuration key and saves to disk immediately."""
        with self.file_lock:
            config = self._load()
            if section not in config:
                config[section] = {}
            config[section][key] = value
            return self._save(config)
            
    def update_section(self, section, update_dict):
        """Updates multiple keys within a section."""
        with self.file_lock:
            config = self._load()
            if section not in config:
                config[section] = {}
            config[section].update(update_dict)
            return self._save(config)
            
    def save_all(self, new_config):
        """Completely overwrites the configuration dictionary."""
        with self.file_lock:
            return self._save(new_config)
