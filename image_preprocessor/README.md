# Image Preprocessor

Standalone image pre-processing tool extracted from the `pi5_sender` camera node.
Runs the **exact same pipeline** used before TCP transmission — without requiring a camera, MQTT, or any Pi-specific hardware.

Available in two modes:
- 🖥️ **GUI App** (`gui_app.py`) — Windows desktop app with point-and-click interface
- 💻 **CLI Tool** (`run_preprocess.py`) — command-line batch processing

---

## 🖥️ GUI App (Recommended for Windows)

### Option A — Compile to standalone `.exe` (no Python needed on target PC)

Run on a **Windows machine** with Python installed:
```bat
build_exe.bat
```
This installs dependencies, compiles, and outputs `dist\ImagePreprocessor.exe`.
Copy the `dist\` folder to any Windows PC — no Python or installation required.

### Option B — Run directly with Python

If Python is already installed, just double-click:
```bat
run.bat
```
The script auto-installs `opencv-python` and `numpy` on first run.

### GUI Workflow

1. **📦 Browse for Calibration ZIP** — downloaded from WebUI → 🎯 Calibrate → 📦 Download Calib Files
   - The app auto-detects `cam0` or `cam1` from the ZIP contents
   - Config files are extracted automatically to the app's `configs/` folder
2. **📂 Select Input Folder** — folder containing your `.jpg`/`.png` images
3. **⚙️ Configure Options** — toggle processing steps and JPEG quality
4. **▶ Run Pre-processing** — progress and logs shown in real time
5. Results saved to `<input_folder>/preprocessed/` by default

---


## Folder Structure

```
image_preprocessor/
├── run_preprocess.py          # Main CLI entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── image_pipeline.py      # Core pipeline class (ImagePipeline)
│   ├── image_alignment.py     # Template matching & homography/affine warp
│   └── shadow_removal.py      # Divisive normalization shadow removal
└── configs/
    ├── cam0_calibration_points.json   # ← Copy from camera_node/configs/
    ├── cam1_calibration_points.json   # ← Copy from camera_node/configs/
    ├── cam0_crop_regions.json         # ← Copy from camera_node/configs/
    ├── cam1_crop_regions.json         # ← Copy from camera_node/configs/
    └── templates/
        ├── cam0_mark0.jpg             # ← Copy from camera_node/configs/templates/
        └── cam1_mark0.jpg             # ← Copy from camera_node/configs/templates/
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Copy calibration files from the Pi

You **must** copy the calibration and template files from your Pi's camera node.
From the Pi, run:

```bash
# Adjust paths and destination IP as needed
scp -r pi@<pi-ip>:/home/wf51/pi5_sender/camera_node/configs/cam0_calibration_points.json ./configs/
scp -r pi@<pi-ip>:/home/wf51/pi5_sender/camera_node/configs/cam0_crop_regions.json       ./configs/
scp -r pi@<pi-ip>:/home/wf51/pi5_sender/camera_node/configs/templates/                   ./configs/templates/
```

Or copy manually to `image_preprocessor/configs/`.

---

## Pipeline Steps

Each step can be **individually enabled/disabled** via flags or a config JSON:

| Step | Config Key | Default |
|------|-----------|---------|
| 1. Alignment (template matching + warp) | `enable_alignment` | `true` |
| 2. Shadow Removal (divisive normalization) | `enable_shadow_removal` | `true` |
| 2.5 Pre-Crop (trim N pixels from edges) | `enable_pre_crop` | `false` |
| 3. Grayscale + CLAHE | `enable_grayscale` / `enable_clahe` | `true` |
| 4. Mask-based Cropping (extract ROIs) | `enable_box_cropping` | `true` |

---

## Usage

### Basic — process a folder of images

```bash
python run_preprocess.py --input ./images --output ./output --cam cam0
```

### Single image

```bash
python run_preprocess.py --input ./photo.jpg --output ./output --cam cam0
```

### Load preprocessing config from a JSON file

The JSON can be the full camera config file (with a `"preprocessing"` key)
or just the preprocessing section directly:

```bash
python run_preprocess.py --input ./images --output ./output --cam cam0 \
    --config ./configs/config_cam0.json
```

### Override individual steps via flags

```bash
# Disable alignment and shadow removal
python run_preprocess.py --input ./images --output ./output --cam cam0 \
    --no-align --no-shadow

# Enable pre-crop (trim 10px top, 10px bottom, 0 left, 0 right)
python run_preprocess.py --input ./images --output ./output --cam cam0 \
    --pre-crop 10 10 0 0

# Disable CLAHE, save debug alignment overlays
python run_preprocess.py --input ./images --output ./output --cam cam0 \
    --disable-clahe --debug
```

---

## Output Files

For each input image, one or more output files are created in `--output`:

| File suffix | Content |
|-------------|---------|
| `__raw.jpg` | Original unprocessed frame |
| `__raw_image.jpg` | Final processed surface (mask regions blacked out) |
| `__pre_crop.jpg` | After CLAHE, before masking (only if `enable_pre_crop`) |
| `__crop_X.jpg` | Individual cropped region for each mask (e.g. `crop_1`) |
| `__debug_align.jpg` | Alignment overlay (only if `--debug`) |

---

## Preprocessing Config JSON Format

```json
{
    "enable_alignment": true,
    "enable_shadow_removal": true,
    "enable_pre_crop": false,
    "pre_crop": {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0
    },
    "enable_grayscale": true,
    "enable_clahe": true,
    "enable_box_cropping": true
}
```

Save this as e.g. `my_config.json` and pass it with `--config my_config.json`.

---

## Programmatic Usage

```python
import cv2
from src.image_pipeline import ImagePipeline

BASE_DIR = "/path/to/image_preprocessor"

pipeline = ImagePipeline(camera_id=0, base_dir=BASE_DIR)
pipeline.load_configs(enable_align=True, enable_crop=True)

prep_config = {
    "enable_alignment": True,
    "enable_shadow_removal": True,
    "enable_pre_crop": False,
    "enable_grayscale": True,
    "enable_clahe": True,
    "enable_box_cropping": True,
}

frame = cv2.imread("input.jpg")
results = pipeline.process_frame(frame, prep_config)

for img_id, img_data in results:
    if img_id != "debug_align":  # skip debug overlay
        cv2.imwrite(f"output_{img_id}.jpg", img_data)
```
