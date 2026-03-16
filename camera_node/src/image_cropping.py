import json
import os
import cv2
import numpy as np
import argparse

def load_calibration(cam_id="cam0"):
    # Resolve the directory of THIS script (src)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir) # sender_installer
    
    # Check possible places for crop_4point.json
    config_paths = [
        os.path.join(parent_dir, "configs", f"{cam_id}_calibration_points.json"),
        os.path.join(base_dir, f"{cam_id}_calibration_points.json"),
        f"configs/{cam_id}_calibration_points.json",
        f"{cam_id}_calibration_points.json"
    ]
    
    config_path = None
    for p in config_paths:
        if os.path.exists(p):
            config_path = p
            break
            
    if not config_path:
        raise ValueError(f"CRITICAL ERROR: configs/{cam_id}_calibration_points.json not found.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # We should search for templates in the same directory as the config file
    config_dir = os.path.dirname(config_path)
    
    templates = []
    
    # Check parent directory (e.g. sender_installer -> wf51)
    parent_dir = os.path.dirname(os.path.dirname(base_dir))

    for mark in config["calibration_marks"]:
        path = os.path.basename(mark["template"]) # Strip any old paths from json just in case
        
        possible_template_paths = [
            os.path.join(config_dir, "templates", path),
            os.path.join(parent_dir, "configs", "templates", path),
            os.path.join(config_dir, path),
            path
        ]
        
        template_file = None
        for pt in possible_template_paths:
            if os.path.exists(pt):
                template_file = pt
                break
                
        if template_file:
            templates.append(cv2.imread(template_file))
        else:
            raise ValueError(f"CRITICAL ERROR: Template image '{path}' not found! Please ensure all template images exist before running the application.")
            
    return config, templates

def find_mark(img_gray, template, search_roi=None):
    """
    Finds a template in an image.
    search_roi: (x, y, w, h) to limit search. If None, searches full image.
    Returns: (max_loc, max_val) -> ((x,y), score)
    """
    print(f"DEBUG FIND_MARK: img={img_gray.shape}, tmpl={template.shape}", flush=True)
    if search_roi:
        x, y, w, h = search_roi
        # Ensure ROI is within bounds
        h_img, w_img = img_gray.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w_img - x, w)
        h = min(h_img - y, h)
        
        roi = img_gray[y:y+h, x:x+w]
        if roi.size == 0 or roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
             return None, 0.0
             
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Adjust local coordinates to global
        global_x = max_loc[0] + x
        global_y = max_loc[1] + y
        return (global_x, global_y), max_val
    else:
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_loc, max_val

def main():
    parser = argparse.ArgumentParser(description="Crop wall boxes using feature-based homography.")
    parser.add_argument("--input", default="images", help="Input directory containing images")
    parser.add_argument("--output", default="dataset/cropy11", help="Output directory for cropped images")
    args = parser.parse_args()

    # Load Calibration
    config, templates = load_calibration()
    if config is None:
        print("Error: Calibration files not found! Run 'calibrate_offsets.py' first.")
        return

    calib_marks = config["calibration_marks"]
    calib_corners = config.get("calibration_corners", [])
    padding = config.get("padding", 50)
    
    # Extract reference points (centers of marks in calibration image)
    ref_mark_points = np.array([
        [m.get("center_x", m["x"]), m.get("center_y", m["y"])] 
        for m in calib_marks
    ], dtype=np.float32)

    # Output directory for crops
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Get all jpg images in the 'images' directory
    image_dir = args.input
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

    if not image_files:
        print(f"No .jpg files found in '{image_dir}'.")
        return

    is_single_mark = len(calib_marks) == 1
    
    if is_single_mark:
        print(f"Found {len(image_files)} images. Starting 1-Mark Center Cropping (Padding: {padding}px)...")
    else:
        print(f"Found {len(image_files)} images. Starting 4-Mark Homography Processing...")
        # Extract reference corners (points to crop in calibration image). Corners are still strict {x,y} points.
        ref_corner_points = np.array([[c["x"], c["y"]] for c in calib_corners], dtype=np.float32)

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"Processing {img_file}...")
        
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Find Mark 1 (Primary) - Full Search
        tmpl1 = cv2.cvtColor(templates[0], cv2.COLOR_BGR2GRAY)
        
        # We can implement a ROI search for the first mark as well in the future to speed things up,
        # but for now we search the full image just to be safe.
        loc, score = find_mark(img_gray, tmpl1)
        
        if score < 0.5:
            print(f"  Result: Mark 1 not found (score {score:.2f}). Skipping.")
            continue
            
        # Get center of found mark 1
        h, w = tmpl1.shape
        m1_cx = loc[0] + w // 2
        m1_cy = loc[1] + h // 2
        
        if is_single_mark:
            # === SINGLE MARK CENTER CROP (Camera 0) ===
            # The crop box should be: center of mark +/- (half mark width + padding)
            
            # Using the exact same dimensions as the template Mark width for crop sizing base
            mark_w = calib_marks[0].get("width", w)
            mark_h = calib_marks[0].get("height", h)
            
            crop_w = int(mark_w + (padding * 2))
            crop_h = int(mark_h + (padding * 2))
            
            # Calculate top-left based on center
            start_x = int(m1_cx - (crop_w / 2))
            start_y = int(m1_cy - (crop_h / 2))
            
            # Enforce bounds
            h_img, w_img = original_img.shape[:2]
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(w_img, start_x + crop_w)
            end_y = min(h_img, start_y + crop_h)
            
            final_crop = original_img[start_y:end_y, start_x:end_x]
            
            # Save
            base_name = os.path.splitext(img_file)[0]
            crop_filename = f"{base_name}_crop.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, final_crop)
            print(f"  Saved crop: {crop_path}")
            
        else:
            # === 4-MARK HOMOGRAPHY (Camera 1) ===
            found_marks = [[m1_cx, m1_cy]]
            ref_m1 = ref_mark_points[0]
            
            for i in range(1, 4):
                tmpl = cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY)
                ref_m = ref_mark_points[i]
                
                dx = ref_m[0] - ref_m1[0]
                dy = ref_m[1] - ref_m1[1]
                
                exp_cx = m1_cx + dx
                exp_cy = m1_cy + dy
                
                search_pad = 150
                th, tw = tmpl.shape
                exp_tl_x = exp_cx - tw // 2
                exp_tl_y = exp_cy - th // 2
                roi_rect = (int(exp_tl_x - search_pad), int(exp_tl_y - search_pad), search_pad*2 + tw, search_pad*2 + th)
                
                loc_i, score_i = find_mark(img_gray, tmpl, roi_rect)
                
                if score_i < 0.5:
                    print(f"  Result: Mark {i+1} not found (score {score_i:.2f}). Skipping.")
                    found_marks = None
                    break
                
                cx = loc_i[0] + tw // 2
                cy = loc_i[1] + th // 2
                found_marks.append([cx, cy])
                
            if found_marks is None:
                continue
                
            current_mark_points = np.array(found_marks, dtype=np.float32)
            
            # Compute Homography
            H, _ = cv2.findHomography(ref_mark_points, current_mark_points, cv2.RANSAC, 5.0)
            if H is None:
                print("  Result: Homography failed.")
                continue
                
            # Map Calibration Corners
            ref_corners_reshaped = ref_corner_points.reshape(-1, 1, 2)
            current_corners = cv2.perspectiveTransform(ref_corners_reshaped, H)
            src_crop_points = current_corners.reshape(4, 2).astype(np.float32)
            
            # Final Warp
            w_top = np.linalg.norm(src_crop_points[0] - src_crop_points[1])
            w_bot = np.linalg.norm(src_crop_points[3] - src_crop_points[2])
            out_w = int((w_top + w_bot) / 2)
            
            h_left = np.linalg.norm(src_crop_points[0] - src_crop_points[3])
            h_right = np.linalg.norm(src_crop_points[1] - src_crop_points[2])
            out_h = int((h_left + h_right) / 2)
            
            dst_rect = np.array([
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1]
            ], dtype=np.float32)
            
            M_final = cv2.getPerspectiveTransform(src_crop_points, dst_rect)
            final_crop = cv2.warpPerspective(original_img, M_final, (out_w, out_h))
            
            base_name = os.path.splitext(img_file)[0]
            crop_filename = f"{base_name}_homography.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, final_crop)
            print(f"  Saved crop: {crop_path}")

    print("Processing complete.")

if __name__ == "__main__":
    main()
