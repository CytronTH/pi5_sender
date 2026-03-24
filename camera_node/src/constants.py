# Constants and static configurations for pi5_sender

# Default IP and Ports
DEFAULT_TCP_IP = "10.10.10.199"
DEFAULT_TCP_PORT_CAM0 = 8080
DEFAULT_TCP_PORT_CAM1 = 8081

DEFAULT_MQTT_BROKER = "wfmain.local"
DEFAULT_MQTT_PORT = 1883

# Default Camera resolutions
DEFAULT_CAMERA_WIDTH = 2304
DEFAULT_CAMERA_HEIGHT = 1296
DEFAULT_JPEG_QUALITY = 90

# Pre-processing padding default (for single mark center crop)
DEFAULT_ALIGNMENT_PADDING = 50

# Sensor Information
SENSOR_INFO = {
    "IMX708": {"name": "Camera Module 3", "max_res": "4608x2592", "modes": ["4608x2592", "2304x1296", "1536x864"]},
    "IMX708_WIDE": {"name": "Camera Module 3 Wide", "max_res": "4608x2592", "modes": ["4608x2592", "2304x1296", "1536x864"]},
    "IMX477": {"name": "HQ Camera", "max_res": "4056x3040", "modes": ["4056x3040", "2028x1520", "2028x1080", "1332x990"]},
    "IMX219": {"name": "Camera Module 2", "max_res": "3280x2464", "modes": ["3280x2464", "1920x1080", "1640x1232"]},
    "OV5647": {"name": "Camera Module 1", "max_res": "2592x1944", "modes": ["2592x1944", "1920x1080"]}
}
