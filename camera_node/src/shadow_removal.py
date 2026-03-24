"""
Batch Shadow Removal (Divisive Normalization)

This script processes a directory of images to remove shadows using background estimation.
It outputs side-by-side comparisons: [Original | Shadow Removed]

Usage:
    python batch_shadow_removal.py --input <input_dir> --output <output_dir>
    
Parameters:
    --sigma: Gaussian Blur kernel size for background estimation (default: 50)
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def remove_shadows_divisive(image, sigma=30):
    """
    Remove shadows using background division.
    """
    if image is None: return None
    
    # Work in LAB space to separate Luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Smooth L channel to get background light map
    bg_l = cv2.GaussianBlur(l, (0, 0), sigma)
    
    # Avoid division by zero
    bg_l = np.maximum(bg_l, 1)
    
    # Divide: Ref = Img / Bg
    mean_l = np.mean(l)
    result_l = (l.astype(np.float32) / bg_l.astype(np.float32)) * mean_l
    
    # Clip and convert back
    result_l = np.clip(result_l, 0, 255).astype(np.uint8)
    
    # Merge
    result_lab = cv2.merge((result_l, a, b))
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

def enhance_black_marks(image):
    """
    Enhance black marks by stretching contrast in dark regions.
    Equation: Output = (Input / 255)^Gamma * 255
    Gamma > 1 makes darks darker.
    Also can use Adaptive Threshold to isolate.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gamma Correction (Gamma=2.0) to make darks darker
    inv_gamma = 1.0 / 2.0
    table = np.array([((i / 255.0) ** 2.0) * 255 for i in np.arange(0, 256)]).astype("uint8")
    darker = cv2.LUT(gray, table)
    
    # 3. Apply CLAHE to boost local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(darker)
    
    # 4. Optional: Adaptive Threshold to binaries? 
    # User said "make background lighter", so maybe just strong contrast.
    # Let's try to bias towards white for non-black.
    # Normalize so that mean is high?
    
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def enhance_black_marks_invert(image):
    """
    Step 4: Invert the Enhanced Mark image.
    Black marks become White. Background becomes Dark.
    Usually easier for detection.
    """
    # 1. Invert
    inverted = cv2.bitwise_not(image)
    
    # 2. Optional: Threshold to remove noise?
    # gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_TOZERO)
    # return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    return inverted

def add_label(image, text):
    h, w = image.shape[:2]
    # Add black bar at top
    img_label = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    cv2.putText(img_label, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img_label

def process_directory(input_dir, output_dir, sigma=50, recursive=True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} not found.")
        return
    
    # Find images
    if recursive:
        image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png"))
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    processed_count = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # 1. Shadow Removal (RESTORED)
            shadow_removed = remove_shadows_divisive(img, sigma=sigma)
            
            # 2. Black Mark Enhancement (Optional, not used in output but good to have)
            # enhanced_mark = enhance_black_marks(shadow_removed)
            
            # 3. Invert (New!)
            # inverted_mark = enhance_black_marks_invert(enhanced_mark)
            
            # --- OUTPUT MODE ---
            # User requested ONLY shadow removed image (No labels, No side-by-side)
            final_output = shadow_removed
            
            # If user wants Enhanced/Inverted instead, un-comment below:
            # final_output = enhanced_mark
            
            # Output Path
            rel_path = img_path.relative_to(input_path)
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            cv2.imwrite(str(out_file), final_output)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print(f"\nCompleted! Processed {processed_count} images.")
    print(f"Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch Shadow Removal with Side-by-Side Output")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sigma", type=int, default=50, help="Gaussian Blur Sigma (Background Smoothness)")
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, sigma=args.sigma)

if __name__ == "__main__":
    main()
