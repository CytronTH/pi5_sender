import cv2
import numpy as np
import json
import os
import argparse
from .image_cropping import load_calibration, find_mark

def calculate_canonical_targets(config):
    """
    Calculates the 'Ideal' positions for the Marks in the final output image.
    1. Determines output size from Calibration Corners.
    2. Maps Calibration Marks to this output space.
    """
    calib_marks = config["calibration_marks"]
    calib_corners = config["calibration_corners"]
    
    # 1. Convert to arrays
    pts_marks = np.array([
        [m.get("center_x", m["x"]), m.get("center_y", m["y"])] 
        for m in calib_marks
    ], dtype=np.float32)
    pts_corners = np.array([[c["x"], c["y"]] for c in calib_corners], dtype=np.float32)
    
    # 2. Determine Output Size (Width/Height)
    # Using the same logic: straighten the calibration crop
    # Top/Bottom width avg
    w_top = np.linalg.norm(pts_corners[0] - pts_corners[1])
    w_bot = np.linalg.norm(pts_corners[3] - pts_corners[2])
    out_w = int((w_top + w_bot) / 2)

    # Left/Right height avg
    h_left = np.linalg.norm(pts_corners[0] - pts_corners[3])
    h_right = np.linalg.norm(pts_corners[1] - pts_corners[2])
    out_h = int((h_left + h_right) / 2)
    
    # --- PADDING / MARGIN LOGIC ---
    # Supports separate X and Y padding
    padding = config.get("padding", 0) # Global padding
    pad_x = config.get("padding_x", padding) # Override for X
    pad_y = config.get("padding_y", padding) # Override for Y
    
    # Update Output Size
    out_w += pad_x * 2
    out_h += pad_y * 2
    
    # Safety check
    if out_w <= 0 or out_h <= 0:
        print(f"Warning: Padding X={pad_x}, Y={pad_y} invalid. Resetting.")
        out_w -= pad_x * 2
        out_h -= pad_y * 2
        pad_x = 0; pad_y = 0
    
    # 3. Define Target Corners (0,0) -> (W,H)
    target_corners = np.array([
        [pad_x, pad_y],
        [out_w - pad_x - 1, pad_y],
        [out_w - pad_x - 1, out_h - pad_y - 1],
        [pad_x, out_h - pad_y - 1]
    ], dtype=np.float32)
    
    # 4. Find Transform: Calibration Image -> Output Image
    # This maps the "Crop Region" in Calibration Image to the "Pad-shifted Region" in Output Image
    M_calib_to_out = cv2.getPerspectiveTransform(pts_corners, target_corners)
    
    # 5. Map Calibration Marks to Output Space
    # Shape needs to be (N, 1, 2) for perspectiveTransform
    pts_marks_reshaped = pts_marks.reshape(-1, 1, 2)
    target_marks = cv2.perspectiveTransform(pts_marks_reshaped, M_calib_to_out)
    target_marks = target_marks.reshape(-1, 2)
    
    return target_marks, (out_w, out_h)

def main():
    parser = argparse.ArgumentParser(description="Align wall boxes using Mark-Based Registration.")
    parser.add_argument("--input", default="images", help="Input directory")
    parser.add_argument("--output", default="dataset/aligned", help="Output directory")
    args = parser.parse_args()
    
    # 1. Load Calibration
    config, templates = load_calibration()
    if config is None:
        print("Error: Calibration not found.")
        return
        
    # 2. Calculate Targets (Where the marks SHOULD be in the output)
    target_marks, output_size = calculate_canonical_targets(config)
    print(f"Target Output Size: {output_size}")
    print("Target Mark Positions:")
    for i, tm in enumerate(target_marks):
        print(f"  M{i+1}: {tm}")
        
    os.makedirs(args.output, exist_ok=True)
    
    # 3. Process Images
    if not os.path.exists(args.input):
        print("Input directory not found.")
        return
        
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images.")
    
    # Pre-calculate relative positions for optimized search
    ref_mark_points = np.array([[m["x"], m["y"]] for m in config["calibration_marks"]], dtype=np.float32)
    ref_m1 = ref_mark_points[0]
    
    count = 0
    for img_file in image_files:
        path = os.path.join(args.input, img_file)
        img = cv2.imread(path)
        if img is None: continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- Detection Logic (Same as crop_wall_boxes) ---
        found_marks = []
        
        # Find Mark 1
        tmpl1 = cv2.cvtColor(templates[0], cv2.COLOR_BGR2GRAY)
        loc, score = find_mark(img_gray, tmpl1) # Full search
        if score < 0.5:
            print(f"Skipping {img_file}: Mark 1 not found.")
            continue
            
        th, tw = tmpl1.shape
        m1_cx = loc[0] + tw // 2
        m1_cy = loc[1] + th // 2
        found_marks.append([m1_cx, m1_cy])
        
        # Find Marks 2-N
        valid = True
        for i in range(1, len(ref_mark_points)):
            tmpl = cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY)
            ref_m = ref_mark_points[i]
            dx = ref_m[0] - ref_m1[0]
            dy = ref_m[1] - ref_m1[1]
            exp_cx, exp_cy = m1_cx + dx, m1_cy + dy
            
            search_pad = 150
            th_i, tw_i = tmpl.shape
            exp_tl_x = exp_cx - tw_i // 2
            exp_tl_y = exp_cy - th_i // 2
            roi_rect = (int(exp_tl_x - search_pad), int(exp_tl_y - search_pad), search_pad*2 + tw_i, search_pad*2 + th_i)
            
            loc, score = find_mark(img_gray, tmpl, roi_rect)
            if score < 0.5:
                # print(f"Skipping {img_file}: Mark {i+1} not found.")
                valid = False
                break
            found_marks.append([loc[0] + tw_i//2, loc[1] + th_i//2])
            
        if not valid: continue
        
        input_marks = np.array(found_marks, dtype=np.float32)
        
        # --- Crucial Difference: Registration ---
        # We want to warp Input Image such that Input Marks -> Target Marks
        if len(input_marks) == 4:
            H, _ = cv2.findHomography(input_marks, target_marks, cv2.RANSAC, 5.0)
            if H is None:
                print(f"Homography failed for {img_file}")
                continue
            aligned_img = cv2.warpPerspective(img, H, output_size)
        elif len(input_marks) == 2:
            M, _ = cv2.estimateAffinePartial2D(input_marks, target_marks)
            if M is None:
                print(f"Affine Transform failed for {img_file}")
                continue
            aligned_img = cv2.warpAffine(img, M, output_size)
        else:
            print(f"Unsupported number of marks ({len(input_marks)}) for {img_file}")
            continue
        
        out_path = os.path.join(args.output, f"{os.path.splitext(img_file)[0]}_aligned.jpg")
        cv2.imwrite(out_path, aligned_img)
        count += 1
        print(f"Saved {out_path}")

    print(f"Processed {count} images.")

if __name__ == "__main__":
    main()
