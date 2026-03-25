# WF51 Camera Sender Service

ระบบรวบรวมและส่งภาพจากกล้อง Raspberry Pi สำหรับวิเคราะห์คุณภาพงานผลิต (Vision Inspection System) รองรับการทำงาน 2 กล้องพร้อมกัน (Dual Camera) ระบบจะถ่ายภาพ, ทำ Pre-processing อัตโนมัติ และสาดภาพผ่าน TCP ไปยัง AI Inference Server

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
/home/pi/pi5_sender/
├── README.md                       # คู่มือฉบับนี้
├── mock/                           # โฟลเดอร์เก็บภาพจำลองสำหรับทดสอบออฟไลน์
│
└── camera_node/
    ├── main.py                     # สคริปต์หลัก (Main Camera Node)
    ├── setup.sh                    # สคริปต์ติดตั้ง Systemd Service  
    ├── camera-sender@.service      # Systemd Template Service (Dual Camera)
    ├── requirements.txt            # Python Dependencies
    │
    ├── src/                        # โมดูลประมวลผลภาพเบื้องหลัง
    │   ├── image_alignment.py      # Alignment: ค้นหามาร์กและคำนวณ Homography
    │   ├── shadow_removal.py       # Shadow Removal: ลบเงาด้วย Divisive Filtering
    │   ├── grayscale_filter.py
    │   └── image_cropping.py       # Crop: โหลด Calibration และหั่นภาพย่อย
    │
    ├── configs/                    # ไฟล์ตั้งค่าของระบบ
    │   ├── config_cam0.json        # ตั้งค่ากล้อง 0 (TCP Port 8080)
    │   ├── config_cam1.json        # ตั้งค่ากล้อง 1 (TCP Port 8081)
    │   ├── calibration_points.json # พิกัดมาร์กอ้างอิง 4 จุด
    │   ├── crop_regions.json       # พิกัดการตัดภาพ (Crop/Mask Regions)
    │   └── templates/              # ภาพ Template ของมาร์ก 4 จุดสำหรับ Template Matching
    │
    └── logs/                       # รูปผลลัพธ์จากโหมด --debug_align (ออโต้สร้าง)
```

---

## ⚙️ 1. การตั้งค่าระบบ (Configuration)

ไฟล์ Config แต่ละกล้อง (`configs/config_cam0.json`, `configs/config_cam1.json`) แบ่งออกเป็น 3 หมวดหลัก:

### `tcp` – ปลายทางรับภาพ (AI Inference Server)
| Key | คำอธิบาย |
|---|---|
| `ip` | IP ของ Receiver (เช่น `10.10.10.199` หรือ `wfmain.local`) |
| `port` | `8080` สำหรับ cam0, `8081` สำหรับ cam1 (แยก Port กันเพื่อป้องกันข้อมูลชน) |

### `mqtt` – การตั้งค่า MQTT Broker สำหรับควบคุมและส่งสถานะ
| Key | คำอธิบาย |
|---|---|
| `broker` | IP หรือ Host ของ MQTT Broker (เช่น `10.10.10.199`) |
| `port` | Port ของ MQTT Broker (ค่าเริ่มต้น `1883`) |
| `username` | ชื่อผู้ใช้สำหรับเชื่อมต่อ MQTT (ถ้ามี) |
| `password` | รหัสผ่านสำหรับเชื่อมต่อ MQTT (ถ้ามี) |
| `topic_cmd` | Topic สำหรับรับคำสั่งควบคุมกล้อง (เช่น `camera/command`) |
| `topic_status` | Topic สำหรับส่งสถานะของตัวเครื่อง (เช่น `camera0/status` หรือ `camera1/status`) |

### `camera` – พฤติกรรมการถ่ายภาพและการตั้งค่ากล้อง
| Key | คำอธิบาย |
|---|---|
| `id` | หมายเลข Hardware Camera (`0` หรือ `1`) |
| `default_width` | ความกว้างเริ่มต้นของภาพ (เช่น `2304`) |
| `default_height`| ความสูงเริ่มต้นของภาพ (เช่น `1296`) |
| `jpeg_quality` | คุณภาพการบีบอัด JPEG `1-100` (แนะนำ `90`) |
| `continuous_stream` | `false` = รอคำสั่ง MQTT, `true` = ส่งอัตโนมัติตาม `stream_interval` |
| `stream_interval` | ความเร็วในการส่งภาพ (วินาที) ใช้เมื่อ `continuous_stream: true` |
| `loop_delay` | ระยะเวลาหน่วงใน Main Loop เพื่อเซฟการทำงาน CPU (วินาที) |

### `preprocessing` – ควบคุม Pipeline (สวิตช์แต่ละขั้นตอน)
| Key | คำอธิบาย |
|---|---|
| `enable_alignment` | ค้นหามาร์กและจัดภาพ (Homography Warp) |
| `enable_shadow_removal` | ลบเงาและปรับ Contrast (Divisive Filtering) |
| `enable_grayscale` | แปลงภาพเป็นขาว-ดำ |
| `enable_clahe` | เร่งความคมชัด (ใช้ได้เมื่อ `enable_grayscale: true`) |
| `enable_box_cropping` | ตัดภาพย่อยตาม `crop_regions.json` และส่งพร้อม masked_surface |

---

## 🚀 2. การวิ่งระบบในโหมดจริง (Production / Dual Camera)

ระบบใช้ **Systemd Template Service (`camera-sender@.service`)** ซึ่งให้แต่ละกล้องรันเป็น Process แยกกันอย่างอิสระ (Process Isolation) หากกล้องตัวใดตัวหนึ่งพัง Process ของอีกตัวจะไม่ได้รับผลกระทบเลย

### ติดตั้ง Service ครั้งแรก:
```bash
cd ~/pi5_sender/camera_node
chmod +x setup.sh
./setup.sh
```

### เปิดใช้งาน 2 กล้องให้รันตั้งแต่เปิดบอร์ด:
```bash
sudo systemctl enable camera-sender@0
sudo systemctl enable camera-sender@1
```

### คำสั่งจัดการ Service (เติม `@0` หรือ `@1` ท้ายตามต้องการ):
| คำสั่ง | ผล |
|---|---|
| `sudo systemctl start camera-sender@0` | เริ่มทำงาน |
| `sudo systemctl stop camera-sender@0` | หยุดการทำงาน |
| `sudo systemctl status camera-sender@0` | ดูสถานะ |
| `journalctl -u camera-sender@0 -f` | ดู Log แบบ Real-time |

---

## 🔔 3. การควบคุมและส่งสถานะผ่าน MQTT (IoT Telemetry)

### การรับคำสั่ง (MQTT Command Topic)
การลั่นชัตเตอร์แบบ Manual เป็นวิธีที่แนะนำหากต้องการรับรองว่า **ภาพทั้ง 2 กล้องมาจากชิ้นงานเดียวกันเสมอ**

ระบบจะฟังคำสั่งจาก Topic ที่ตั้งไว้ใน `topic_cmd` (เช่น `camera/command`)
เมื่อส่งคำสั่งนี้เข้าไป กล้องจะได้รับคำสั่งและทำงานทันที:

```json
{ "action": "capture" }
```

คำสั่งสั่งการผ่าน MQTT ที่รองรับในปัจจุบัน:

| Payload | ผล |
|---|---|
| `{"action": "capture"}` | ลั่นชัตเตอร์ถ่ายรูปและส่ง 1 ชุด |
| `{"ExposureTime": 10000}` | ปรับ Exposure Time (µs) |
| `{"AnalogueGain": 2.0}` | ปรับ Gain |
| `{"resolution": [1920, 1080]}` | ปรับเปลี่ยนความละเอียดกล้องแบบ Real-time |
| `{"system": "restart"}` | สั่ง Reboot บอร์ด |
| `{"system": "shutdown"}`| สั่ง Shutdown บอร์ด |

### การส่งสถานะการทำงาน (MQTT Status Topic)
โปรแกรมถูกพัฒนาให้รายงานสถานะตัวเครื่องตลอดการทำงาน ทุกๆ 5 วินาที โดยจะส่งไปยัง Topic ที่ระบุใน `topic_status` (ตัวอย่างเช่น สำหรับ cam0 ตามข้อกำหนดจะเป็น Topic: `camera0/status` และสำหรับ cam1 เป็น `camera1/status`) ข้อมูลเป็นดังนี้:

```json
{
  "cpu_temp": 48.5,
  "ram_usage_percent": 15.6,
  "cpu_usage_percent": 8.0,
  "resolution": [2304, 1296]
}
```

---

## 🖥️ 4. เครื่องมือจัดการภาพบนหน้าจอ (WebUI Tools)

ระบบมีเครื่องมืออำนวยความสะดวกใน WebUI เพื่อช่วยตั้งค่าและทดสอบภาพได้อย่างแม่นยำ:

### 📡 Live Output Monitor
เมื่อกล้องรันอยู่ในโหมด `TCP/SFTP Background Node` คุณสามารถกดปุ่ม **📡 Live Monitor** เพื่อเปิดหน้าต่างดูภาพสดที่ระบบส่งออกไปจริงๆ
- ดึงภาพผ่าน RAM Disk `/dev/shm` ออกมาโชว์บนจอโดยไม่รบกวนความเร็วของ Background Process
- แสดงภาพแยกตาม ID เช่น `[raw_image]`, `[pre_crop]`, หรือ `[crop_1]` ทำให้รู้ว่า Pipeline หั่นภาพออกมาได้ถูกต้องตามที่วาดไว้หรือไม่ 

### 🎯 Calibration Studio (การตั้งค่ามาร์กเกอร์และจุดตัด)
หน้าต่างตั้งค่าวาดกรอบภาพ (Crop Regions) และกำหนดพิกัด 4 จุด หรือจุดเดียว (Single-Mark):
- **Upload Target Image**: สามารถอัปโหลดภาพเครื่องจักร/ชิ้นงานจากคอมพิวเตอร์เข้าไปเป็นภาพอ้างอิงได้โดยตรง
- **Grid Overlay & Crosshair**: หน้าจอวาดกรอบมีตาราง Grid (ทุก 50px) และสเกลระบุระยะพร้อมเป้ากลางจอ เพื่อให้กะระยะวาดกล่องง่ายขึ้น
- **Live Pre-process Preview**: ทันทีที่วาดกรอบเสร็จ สามารถกดปุ่ม `👁️ Preview Pre-process` เพื่อดูล่วงหน้าได้เลยว่า ภาพเข้าสู่กระบวนการลบเงา ตัดขอบ และแปลงสีแล้วจะออกมาหน้าตาเป็นอย่างไร โดยไม่จำเป็นต้องเซฟแล้วมานั่งสุ่มรันในโหมด TCP 

> **📌 Note สำหรับ Single-Mark (cam0):** ในโหมด 1 Tracking Mark ระบบจะตัดภาพให้มีขนาด **เป๊ะ 1:1 เท่ากับกรอบสีเหลืองที่คุณวาดไว้พอดี (ไม่แถม Padding)** เพื่อรีดประสิทธิภาพการ Crop วัตถุให้ใช้แบนด์วิดท์น้อยที่สุด

---

## 🛠️ 5. การทดสอบออฟไลน์ (Mock Mode)

ทดสอบ Pipeline ได้โดยไม่ต้องต่อกล้องจริง โดยใช้ภาพใน `mock/`:

### รัน Mock พื้นฐาน:
```bash
python3 ~/pi5_sender/camera_node/main.py -c configs/config_cam0.json --mock_dir ~/pi5_sender/mock
```

### รัน Mock พร้อมบันทึกภาพทุกขั้น (--debug_align):
```bash
python3 ~/pi5_sender/camera_node/main.py -c configs/config_cam0.json --mock_dir ~/pi5_sender/mock --debug_align
```
รูปผลลัพธ์ (`masked_surface` และ `crop_X`) จะถูกบันทึกใน `logs/` โดยอัตโนมัติ

### ปิด CLAHE (ภาพขาวดำไม่เร่ง Contrast):
```bash
python3 ... --disable_clahe
```

---

## 📡 6. โครงสร้างโปรโตคอลการส่งข้อมูล (TCP Stream Protocol)

การลั่นชัตเตอร์ 1 ครั้ง จะส่งภาพหลายรูปต่อเนื่องกันในสาย TCP เดียวกัน:
`masked_surface` → `crop_0` → `crop_1` → … → `crop_5`

แต่ละรูปจะถูกห่อด้วยโปรโตคอล 3 ส่วน:

| ลำดับ | ขนาด | รูปแบบ | คำอธิบาย |
|---|---|---|---|
| 1 | 4 bytes | Big-endian `>L` | ขนาดของ JSON Metadata ที่จะตามมา |
| 2 | N bytes (จากข้อ 1) | UTF-8 JSON String | `{"id": "crop_0", "size": 15420}` |
| 3 | M bytes (จาก `size`) | Raw JPEG | ไฟล์ภาพ JPEG พร้อมใช้ |

> 📎 โปรโตคอลนี้ออกแบบมาให้ Receiver สามารถรับภาพและแยกส่วนต่างๆ ได้อย่างแม่นยำและรวดเร็ว โดยอาศัย Metadata นำทางข้อมูลภาพเสมอ
