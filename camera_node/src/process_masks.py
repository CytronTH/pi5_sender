import cv2
import numpy as np
import json
import os
import argparse
import glob
from .align_wall_boxes import load_calibration, calculate_canonical_targets, find_mark

def process_dataset_masks(input_dir, output_root="dataset_v2", mask_config_path="configs/masks.json", skip_align=False):
    # 1. Load Alignment Config (Only needed if aligning)
    if not skip_align:
        align_config, templates = load_calibration()
        if align_config is None:
            print("Error: Alignment calibration not found.")
            return

    # 2. Load Mask Config
    if not os.path.exists(mask_config_path):
        print(f"Error: Mask config '{mask_config_path}' not found. Please run select_mask_regions.py first.")
        return
        
    with open(mask_config_path, "r") as f:
        mask_config = json.load(f)
        
    mask_regions = mask_config["mask_regions"] # List of {x, y, w, h}
    ref_size = mask_config.get("reference_image_size", None)
    
    # 3. Setup Directories
    dir_surface = os.path.join(output_root, "surface_masked")
    os.makedirs(dir_surface, exist_ok=True)
    
    dir_marks_root = os.path.join(output_root, "marks_crop")
    os.makedirs(dir_marks_root, exist_ok=True)
    
    # Create subdirs for each mark ID
    for region in mask_regions:
        os.makedirs(os.path.join(dir_marks_root, region["id"]), exist_ok=True)
        
    # 4. Process Loop
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    print(f"Processing {len(image_files)} images (Skip Align: {skip_align})...")
    
    # Pre-calc alignment targets (if aligning)
    target_marks = None
    output_size = None
    
    if not skip_align:
        target_marks, output_size = calculate_canonical_targets(align_config)
        ref_mark_points = np.array([[m["x"], m["y"]] for m in align_config["calibration_marks"]], dtype=np.float32)
        ref_m1 = ref_mark_points[0]
    else:
        # If skipping alignment, rely on reference size from mask config
        if ref_size:
            output_size = (ref_size["width"], ref_size["height"])
        else:
            print("Warning: No reference size in mask config. Assuming inputs match mask coordinates.")
    
    count = 0
    for img_path in image_files:
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        aligned_img = None
        
        if skip_align:
            # Mode B: Input is ALREADY aligned
            # Check for "Post-Alignment Padding" (Cropping)
            # If user wants to "Crop In" further on an aligned image
            
            # Load config manually to get padding
            align_config_temp = None
            try:
                with open("configs/crop_4point.json", "r") as f:
                    align_config_temp = json.load(f)
            except: pass
            
            pad_x = 0; pad_y = 0
            if align_config_temp:
                padding = align_config_temp.get("padding", 0)
                pad_x = align_config_temp.get("padding_x", padding)
                pad_y = align_config_temp.get("padding_y", padding)
            
            # If negative padding (cropping), apply it
            # Start with full image
            h, w = img.shape[:2]
            start_x, start_y = 0, 0
            end_x, end_y = w, h
            
            if pad_x < 0:
                crop = abs(pad_x)
                start_x += crop
                end_x -= crop
            if pad_y < 0:
                crop = abs(pad_y)
                start_y += crop
                end_y -= crop
                
            # Validate
            if start_x < end_x and start_y < end_y:
                aligned_img = img[start_y:end_y, start_x:end_x]
            else:
                aligned_img = img # Fallback
                
        else:
            # Mode A: Input is RAW -> Need Alignment
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found_marks = []
            
            # M1
            tmpl1 = cv2.cvtColor(templates[0], cv2.COLOR_BGR2GRAY)
            loc, score = find_mark(img_gray, tmpl1)
            if score < 0.5: continue
            th, tw = tmpl1.shape
            m1_cx, m1_cy = loc[0] + tw//2, loc[1] + th//2
            found_marks.append([m1_cx, m1_cy])
            
            # M2-M4
            valid = True
            for i in range(1, 4):
                tmpl = cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY)
                ref_m = ref_mark_points[i]
                dx, dy = ref_m[0] - ref_m1[0], ref_m[1] - ref_m1[1]
                exp_cx, exp_cy = m1_cx + dx, m1_cy + dy
                # Roi search
                w_box, h_box = 300, 300
                roi_x, roi_y = int(exp_cx - w_box/2), int(exp_cy - h_box/2)
                roi_rect = (roi_x, roi_y, w_box, h_box)
                loc_i, score_i = find_mark(img_gray, tmpl, roi_rect)
                
                if score_i < 0.4: 
                    valid = False; break
                th_i, tw_i = tmpl.shape
                found_marks.append([loc_i[0] + tw_i//2, loc_i[1] + th_i//2])
                
            if not valid:
                print(f"Skipping {basename}: Marks not found (Use --skip-align if image is already aligned)")
                continue
                
            # Warp
            input_marks = np.array(found_marks, dtype=np.float32)
            H, _ = cv2.findHomography(input_marks, target_marks, cv2.RANSAC, 5.0)
            aligned_img = cv2.warpPerspective(img, H, output_size)
        
        # --- CROP & MASK STEP ---
        # Now we have 'aligned_img', whether it was warped or loaded directly
        masked_surface = aligned_img.copy()
        
        # Get output size for clamp
        h_out, w_out = aligned_img.shape[:2]
        
        for region in mask_regions:
            rid = region["id"]
            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
            
            # --- START_X / START_Y Adjustment ---
            # If we cropped the image (skip_align=True + Negative Padding),
            # we must shift the mask coordinates to match the new image origin.
            if skip_align:
                # start_x, start_y were calculated above
                rx -= start_x
                ry -= start_y
            
            # 1. Crop Mark
            # Boundary checks
            rx = max(0, rx); ry = max(0, ry)
            rw = min(rw, w_out - rx) # clamp width
            rh = min(rh, h_out - ry) # clamp height
            
            if rw > 0 and rh > 0:
                mark_crop = aligned_img[ry:ry+rh, rx:rx+rw]
                
                # Check for sub-crops
                if "sub_crops" in region and region["sub_crops"]:
                    for sub in region["sub_crops"]:
                        sid = sub["id"]
                        sx, sy, sw, sh = sub["x"], sub["y"], sub["w"], sub["h"]
                        
                        # Sub-crop coordinates are relative to the MAIN crop (mark_crop)
                        # Boundary checks
                        sx = max(0, sx); sy = max(0, sy)
                        sw = min(sw, rw - sx)
                        sh = min(sh, rh - sy)
                        
                        if sw > 0 and sh > 0:
                            sub_img = mark_crop[sy:sy+sh, sx:sx+sw]
                            sub_path = os.path.join(dir_marks_root, rid, f"{rid}_{sid}_{basename}")
                            cv2.imwrite(sub_path, sub_img)
                else:
                    # Default: Save the main crop
                    crop_path = os.path.join(dir_marks_root, rid, f"{rid}_{basename}")
                    cv2.imwrite(crop_path, mark_crop)
            
            # 2. Mask Surface (Draw Black Box)
            cv2.rectangle(masked_surface, (rx, ry), (rx+rw, ry+rh), (0, 0, 0), -1)
            
        # Save Masked Surface
        surf_path = os.path.join(dir_surface, f"masked_{basename}")
        cv2.imwrite(surf_path, masked_surface)
        
        count += 1
        # print(f"Processed {basename}")
        
    print(f"Done. Processed {count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="images", help="Input directory")
    parser.add_argument("--output", default="dataset_split", help="Output root directory")
    parser.add_argument("--skip-align", action="store_true", help="Set this if input images are ALREADY aligned/cropped. Skips homography.")
    args = parser.parse_args()
    
    process_dataset_masks(args.input, args.output, skip_align=args.skip_align)
