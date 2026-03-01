import socket
import struct
import json
import cv2
import numpy as np

# Configuration
TCP_IP = '0.0.0.0'  # Listen on all available network interfaces
TCP_PORT = 8080     # Must match the sender's port

received_frames = {} # Store the latest frames for UI Dashboard

def resize_with_pad(image, target_w, target_h, pad_color=(0, 0, 0)):
    """
    ย่อ/ขยายภาพโดยคงอัตราส่วนเดิมไว้ (Aspect Ratio) 
    และถมสีดำ(หรือสีที่กำหนด)ในพื้นที่ที่เหลือ (Letterboxing)
    """
    h, w = image.shape[:2]
    # หาอัตราส่วนว่าควรใช้แกนกว้างหรือแกนสูงเป็นหลักในการย่อ
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # ย่อภาพก่อน
    resized = cv2.resize(image, (new_w, new_h))
    
    # สร้างผืนผ้าใบเปล่าๆ ขนาดเป้าหมาย
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    
    # คำนวณจุดกึ่งกลางที่จะใช้วางภาพ
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # วางภาพทับลงไป
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

def main():
    # 1. Start the TCP Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allows immediate port reuse
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(1) # We only expect one sender to connect at a time
    
    print(f"📡 Receiver started. Listening on {TCP_IP}:{TCP_PORT}...")
    
    while True:
        try:
            print("⏳ Waiting for sender to connect...")
            conn, addr = server_socket.accept()
            print(f"✅ Sender connected from: {addr}")
            
            # Start receiving loop for this connection
            receive_images(conn)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down receiver...")
            break
        except Exception as e:
            print(f"⚠️ Error accepting connection: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

def recv_exactly(sock, num_bytes):
    """
    Helper function to ensure we read exactly 'num_bytes' from the socket.
    TCP is a stream protocol; it doesn't guarantee that one send() equals one recv().
    Data might be fragmented, so we must loop until we get exactly what we need.
    """
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None # Connection closed by the sender
        data.extend(packet)
    return data

def receive_images(conn):
    """
    Handles the incoming byte stream from the sender using the new Protocol:
    [4-byte JSON Size] + [JSON Metadata] + [JPEG Data]
    """
    while True:
        try:
            # --- 1. Read JSON Size (4 bytes) ---
            # ">L" means Big-endian, Unsigned Long (4 bytes)
            size_data = recv_exactly(conn, 4)
            if not size_data:
                print("🔌 Sender disconnected.")
                break # Exit the receiving loop
                
            json_size = struct.unpack(">L", size_data)[0]
            
            # --- 2. Read JSON Metadata ---
            json_data_bytes = recv_exactly(conn, json_size)
            if not json_data_bytes:
                break
                
            metadata = json.loads(json_data_bytes.decode('utf-8'))
            image_id = metadata.get("id", "unknown")
            image_size = metadata.get("size", 0)
            
            # --- 3. Read JPEG Image Data ---
            image_data = recv_exactly(conn, image_size)
            if not image_data:
                break
                
            # --- 4. Decode JPEG to OpenCV Image (Numpy Array) ---
            # np.frombuffer converts bytes to a 1D numpy array without copying memory
            nparr = np.frombuffer(image_data, np.uint8)
            # cv2.imdecode decodes that array back into a 3-channel (BGR) OpenCV format
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                print(f"🖼️  Received '{image_id}' | Size: {image_size} bytes | Resolution: {img.shape[1]}x{img.shape[0]}")
                
                # --- 👉 จุดสำหรับวาง AI Inference 👈 ---
                # ที่ตรงนี้คือจุดที่คุณสามารถส่งตัวแปร 'img' เข้าไปรันสคริปต์ AI ของคุณได้เลย
                # เช่น:
                # if image_id == "masked_surface":
                #     results = my_ai_model.predict_surface(img)
                # elif image_id.startswith("crop_"):
                #     results = my_ai_model.predict_crop(img)
                # ----------------------------------------
                
                # --- UI Dashboard ---
                global received_frames
                received_frames[image_id] = img
                
                if "masked_surface" in received_frames:
                    main_img = received_frames["masked_surface"]
                    
                    # ขนาดของภาพหลัก (16:9)
                    target_main_w, target_main_h = 1024, 576 
                    main_resized = resize_with_pad(main_img, target_main_w, target_main_h)
                    
                    # ขนาดของภาพครอปแต่ละรูป (Grid 2 คอลัมน์ x 3 แถว)
                    crop_w, crop_h = 256, 192 
                    grid_w = crop_w * 2
                    
                    # สร้างผืนผ้าใบรวม
                    dashboard = np.zeros((target_main_h, target_main_w + grid_w, 3), dtype=np.uint8)
                    
                    # วางภาพหลักทางซ้าย
                    dashboard[0:target_main_h, 0:target_main_w] = main_resized
                    cv2.putText(dashboard, "Masked Surface", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # วางภาพครอป 6 ทางขวา
                    for i in range(1, 7):
                        c_id = f"crop_{i}"
                        if c_id in received_frames:
                            crop_img = received_frames[c_id]
                            # แถวและคอลัมน์ (0, 1, 2)
                            idx = i - 1
                            r = idx // 2
                            c = idx % 2
                            y = r * crop_h
                            x = target_main_w + (c * crop_w)
                            
                            c_resized = resize_with_pad(crop_img, crop_w, crop_h, pad_color=(50, 50, 50))
                            dashboard[y:y+crop_h, x:x+crop_w] = c_resized
                            
                            # ตีเส้นขอบแยกภาพ และเขียนชื่อ
                            cv2.rectangle(dashboard, (x, y), (x+crop_w, y+crop_h), (255, 255, 255), 1)
                            cv2.putText(dashboard, f"Crop {i}", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                    cv2.imshow("Receiver 7-Image Dashboard", dashboard)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 User quit.")
                    import os
                    os._exit(0)
            else:
                print(f"⚠️ Failed to decode image '{image_id}'")

        except Exception as e:
            print(f"❌ Receiver streaming error: {e}")
            break # Break out to parent loop and wait for a fresh connection

if __name__ == "__main__":
    main()
