import socket
import struct
import json
import cv2
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



class TCPSender:
    """Handles TCP Socket connection and transmission of JSON metadata + JPEG image data."""
    
    def __init__(self, ip, port, timeout=5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.tcp_socket = None

    def connect(self):
        """Attempts to establish connection to the TCP receiver."""
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except:
                pass
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(self.timeout)
            self.tcp_socket.connect((self.ip, self.port))
            logger.info(f"Connected to TCP server at {self.ip}:{self.port}")
            return True
        except socket.timeout:
            print("ERROR: TCP Connection timed out.")
            self.tcp_socket = None
            return False
        except ConnectionRefusedError:
            logger.error(f"TCP Connection refused by {self.ip}:{self.port}.")
            self.tcp_socket = None
            return False
        except Exception as e:
            logger.error(f"TCP Connection failed: {e}")
            self.tcp_socket = None
            return False
            
    def disconnect(self):
        """Standard disconnect method."""
        if self.tcp_socket:
            try:
                self.tcp_socket.close()
            except:
                pass
            self.tcp_socket = None

    def send_image(self, frame, image_id="raw_image", jpeg_quality=90, width=0, height=0):
        """Encodes frame and sends it over TCP with metadata header."""
        if self.tcp_socket is None:
            if not self.connect():
                return False

        try:
            # 1. Encode to JPEG
            result, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not result:
                logger.error(f"Failed to encode image")
                return False

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
            header = struct.pack(">L", meta_size)
            
            # Send all parts as a single continuous stream
            self.tcp_socket.sendall(header + metadata_json + data)
            
            if width and height:
                logger.info(f"Sent {image_id}: {width}x{height} ({img_size} bytes)")
            else:
                logger.info(f"Sent {image_id}: ({img_size} bytes)")
                
            return True
            
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            logger.error(f"TCP Send Error: {e}")
            self.disconnect()
            return False
        except Exception as e:
            logger.error(f"Unexpected error during send: {e}")
            return False
