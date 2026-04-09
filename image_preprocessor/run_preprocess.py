#!/usr/bin/env python3
"""
Image Preprocessor - Standalone CLI Tool
=========================================
Processes a folder of images (or a single image) through the same
pre-processing pipeline used by pi5_sender before TCP transmission.

Pipeline steps:
  1. Alignment      - Template-matching based image registration
  2. Shadow Removal - Divisive normalization (LAB space)
  2.5 Pre-Crop      - Trim edges (top/bottom/left/right pixels)
  3. Grayscale + CLAHE - Contrast enhancement
  4. Mask Cropping  - Extract regions of interest

Usage examples:
  # Process a folder of images for cam0 with default config
  python run_preprocess.py --input ./images --output ./output --cam cam0

  # Process a single image
  python run_preprocess.py --input ./photo.jpg --output ./output --cam cam1

  # Use a custom preprocessing config JSON
  python run_preprocess.py --input ./images --output ./output --cam cam0 --config ./my_config.json

  # Disable CLAHE and save debug alignment overlays
  python run_preprocess.py --input ./images --output ./output --cam cam0 --disable-clahe --debug

  # Override which pipeline steps run
  python run_preprocess.py --input ./images --output ./output --cam cam0 \\
      --no-align --no-shadow --no-crop
"""

import argparse
import json
import os
import sys
import time
import glob
import cv2

# Allow running from inside the image_preprocessor folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.image_pipeline import ImagePipeline


# ---------------------------------------------------------------------------
# Default preprocessing config (mirrors config_cam0.json "preprocessing" section)
# ---------------------------------------------------------------------------
DEFAULT_PREP_CONFIG = {
    "enable_alignment": True,
    "enable_shadow_removal": True,
    "enable_pre_crop": False,
    "pre_crop": {"top": 0, "bottom": 0, "left": 0, "right": 0},
    "enable_grayscale": True,
    "enable_clahe": True,
    "enable_box_cropping": True,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_image_paths(input_path: str):
    """Return a list of image file paths from a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(paths))
    else:
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)


def save_outputs(results, output_dir: str, base_name: str, jpeg_quality: int = 90):
    """Save all pipeline output images to disk."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for img_id, img_data in results:
        if img_data is None or img_data.size == 0:
            continue
        out_name = f"{base_name}__{img_id}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        saved.append(out_path)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the pi5_sender image pre-processing pipeline on a batch of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- I/O ---
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input image file or directory of images."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Directory to save processed output images."
    )
    parser.add_argument(
        "--cam", default="cam0", choices=["cam0", "cam1"],
        help="Camera ID to use for calibration config lookup. (default: cam0)"
    )

    # --- Config override ---
    parser.add_argument(
        "--config",
        help="Path to a JSON file containing a 'preprocessing' section. "
             "Overrides the built-in defaults."
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=90, metavar="Q",
        help="JPEG output quality 0-100. (default: 90)"
    )

    # --- Pipeline step toggles ---
    parser.add_argument("--no-align",  action="store_true", help="Disable alignment step.")
    parser.add_argument("--no-shadow", action="store_true", help="Disable shadow removal.")
    parser.add_argument("--no-gray",   action="store_true", help="Disable grayscale conversion.")
    parser.add_argument("--no-clahe",  action="store_true", help="Disable CLAHE enhancement.")
    parser.add_argument("--no-crop",   action="store_true", help="Disable mask-based cropping.")
    parser.add_argument(
        "--pre-crop", nargs=4, type=int, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Enable pre-crop and trim N pixels from each edge. E.g. --pre-crop 10 10 0 0"
    )
    parser.add_argument(
        "--disable-clahe", action="store_true",
        help="Force-disable CLAHE even if enabled in config (same as --no-clahe)."
    )

    # --- Debug ---
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate debug images to logs/ subfolder."
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Do not save the 'raw' (unprocessed) output."
    )
    parser.add_argument(
        "--skip-debug-align", action="store_true", default=True,
        help="Do not save the debug_align overlay image. (default: True)"
    )

    args = parser.parse_args()

    # --- Build prep_config ---
    prep_config = dict(DEFAULT_PREP_CONFIG)

    # Load from file if provided
    if args.config:
        try:
            with open(args.config) as f:
                cfg = json.load(f)
            # Accept full config file (with "preprocessing" key) or bare preprocessing dict
            if "preprocessing" in cfg:
                prep_config.update(cfg["preprocessing"])
            else:
                prep_config.update(cfg)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"ERROR: Failed to load config file: {e}")
            sys.exit(1)

    # CLI flag overrides
    if args.no_align:
        prep_config["enable_alignment"] = False
    if args.no_shadow:
        prep_config["enable_shadow_removal"] = False
    if args.no_gray:
        prep_config["enable_grayscale"] = False
    if args.no_clahe or args.disable_clahe:
        prep_config["enable_clahe"] = False
    if args.no_crop:
        prep_config["enable_box_cropping"] = False
    if args.pre_crop:
        prep_config["enable_pre_crop"] = True
        prep_config["pre_crop"] = {
            "top": args.pre_crop[0],
            "bottom": args.pre_crop[1],
            "left": args.pre_crop[2],
            "right": args.pre_crop[3],
        }

    disable_clahe = args.disable_clahe or args.no_clahe

    cam_num = int(args.cam.replace("cam", ""))

    # --- Initialize Pipeline ---
    print(f"\n{'='*55}")
    print(f"  Image Preprocessor")
    print(f"{'='*55}")
    print(f"  Camera:  {args.cam}")
    print(f"  Base:    {BASE_DIR}")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  Steps enabled:")
    print(f"    Alignment:      {prep_config['enable_alignment']}")
    print(f"    Shadow removal: {prep_config['enable_shadow_removal']}")
    print(f"    Pre-crop:       {prep_config['enable_pre_crop']}")
    print(f"    Grayscale:      {prep_config['enable_grayscale']}")
    print(f"    CLAHE:          {prep_config['enable_clahe'] and not disable_clahe}")
    print(f"    Box cropping:   {prep_config['enable_box_cropping']}")
    print(f"{'='*55}\n")

    pipeline = ImagePipeline(camera_id=cam_num, base_dir=BASE_DIR)
    try:
        pipeline.load_configs(
            enable_align=prep_config["enable_alignment"],
            enable_crop=prep_config["enable_box_cropping"],
        )
        print("Pipeline configs loaded successfully.\n")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Collect images ---
    image_paths = collect_image_paths(args.input)
    if not image_paths:
        print(f"No images found in: {args.input}")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s) to process.\n")

    # --- Process ---
    success_count = 0
    skip_count = 0
    error_count = 0
    t_start = time.time()

    for i, img_path in enumerate(image_paths, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i:>4}/{len(image_paths)}] {base_name}", end=" ... ", flush=True)

        frame = cv2.imread(img_path)
        if frame is None:
            print("SKIP (could not read image)")
            skip_count += 1
            continue

        try:
            t0 = time.time()
            results = pipeline.process_frame(
                frame=frame,
                prep_config=prep_config,
                debug=args.debug,
                mock_name=base_name,
                disable_clahe=disable_clahe,
            )
            elapsed = time.time() - t0

            # Filter outputs
            filtered = []
            for img_id, img_data in results:
                if args.skip_raw and img_id == "raw":
                    continue
                if args.skip_debug_align and img_id == "debug_align":
                    continue
                filtered.append((img_id, img_data))

            saved = save_outputs(filtered, args.output, base_name, args.jpeg_quality)
            ids = [os.path.basename(p) for p in saved]
            print(f"OK ({elapsed:.2f}s) -> {len(saved)} file(s)")
            success_count += 1

        except ValueError as e:
            print(f"SKIP ({e})")
            skip_count += 1
        except Exception as e:
            print(f"ERROR ({e})")
            error_count += 1

    total_time = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Success: {success_count} | Skipped: {skip_count} | Errors: {error_count}")
    print(f"  Outputs saved to: {args.output}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
