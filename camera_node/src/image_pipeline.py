import cv2
import numpy as np
import os
import time
import json
from src.image_alignment import calculate_canonical_targets, find_mark
from src.shadow_removal import remove_shadows_divisive
from src.grayscale_filter import cv2 as dummy_cv2
import src.image_cropping as cwb

class ImagePipeline:
    def __init__(self, camera_id, base_dir):
        self.camera_id = camera_id
        self.base_dir = base_dir
        
        self.preproc_config = None
        self.preproc_templates = None
        self.mask_config = None
        self.target_marks = None
        self.output_size = None
        self.ref_mark_points = None
        
        # Initialize CLAHE once to reuse
        self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def load_configs(self, enable_align, enable_crop):
        """Loads alignment and crop configs similarly to the old init_preprocessing()."""
        # 1. Load Alignment Config
        if enable_align:
            try:
                self.preproc_config, self.preproc_templates = cwb.load_calibration(f"cam{self.camera_id}")
                
                if len(self.preproc_config.get("calibration_marks", [])) > 1:
                    self.target_marks, self.output_size = calculate_canonical_targets(self.preproc_config)
                else:
                    self.target_marks = None
                    self.output_size = None
                
                self.ref_mark_points = np.array([
                    [m.get("center_x", m["x"]), m.get("center_y", m["y"])] 
                    for m in self.preproc_config["calibration_marks"]
                ], dtype=np.float32)
                
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Failed to load calibration config: {e}. Program will terminate.")
            
        # 2. Load Mask Config
        if enable_crop:
            mask_config_path = os.path.join(self.base_dir, "configs", f"cam{self.camera_id}_crop_regions.json")
            try:
                with open(mask_config_path, "r") as f:
                    self.mask_config = json.load(f)
            except FileNotFoundError:
                self.mask_config = None
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Failed to load mask config: {e}. Program will terminate.")

    def process_frame(self, frame, prep_config, debug=False, mock_name=None, disable_clahe=False):
        """Processes the frame and returns an ordered list of tuples: (image_id, frame_data)"""
        
        enable_align = prep_config.get("enable_alignment", True)
        enable_shadow = prep_config.get("enable_shadow_removal", True)
        enable_gray = prep_config.get("enable_grayscale", True)
        enable_clahe = prep_config.get("enable_clahe", True)
        enable_crop = prep_config.get("enable_box_cropping", True)
        
        missing_align = enable_align and not self.preproc_config
        missing_crop = enable_crop and not self.mask_config
        
        calibration_mode_active = (not enable_align) and (not enable_crop)
        fallback_to_raw = calibration_mode_active or (missing_align and missing_crop)
        
        if missing_align and enable_align:
            fallback_to_raw = True
        
        # Create output list holding (id, image_array) tuples
        enable_timestamp_on_raw = prep_config.get("enable_timestamp_on_raw", False)
        
        def finalize_output(images):
            if enable_timestamp_on_raw:
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                for i in range(len(images)):
                    img_id, img_data = images[i]
                    if img_id not in ["debug_align", "inference_ready"]:
                        stamped = img_data.copy()
                        h_img = stamped.shape[0]
                        font_scale = max(0.5, h_img / 1000.0)
                        thickness = max(1, int(font_scale * 2))
                        y_pos = max(20, int(30 * font_scale))
                        cv2.putText(stamped, timestamp_str, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                        images[i] = (img_id, stamped)
            return images

        output_images = [("reference_raw", frame)]
        
        if fallback_to_raw:
            output_images.append(("inference_ready", frame))
            
            # Save local copy for calibration if we are falling back
            calib_out_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(calib_out_dir, exist_ok=True)
            save_path = os.path.join(calib_out_dir, f"cam{self.camera_id}_calibration_target.jpg")
            cv2.imwrite(save_path, frame)
            
            return finalize_output(output_images)

        # --- 1. Alignment & Cropping ---
        if enable_align:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_marks = []
            
            debug_img = frame.copy()
            
            tmpl1 = cv2.cvtColor(self.preproc_templates[0], cv2.COLOR_BGR2GRAY)
            loc, score = find_mark(img_gray, tmpl1)
            
            if score < 0.2:
                if debug:
                    save_path = os.path.join(self.base_dir, "logs", f"debug_align_fail_m1_{int(time.time()*100)}.jpg")
                    cv2.imwrite(save_path, debug_img)
                raise ValueError("Mark 1 not found. Alignment failed.")
                
            th, tw = tmpl1.shape
            m1_cx, m1_cy = loc[0] + tw//2, loc[1] + th//2
            found_marks.append([m1_cx, m1_cy])
            
            cv2.rectangle(debug_img, loc, (loc[0] + tw, loc[1] + th), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(m1_cx), int(m1_cy)), 5, (0, 0, 255), -1)
            cv2.putText(debug_img, "M1", (loc[0], loc[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            is_single_mark = len(self.preproc_templates) == 1 if self.preproc_templates else False
            
            if is_single_mark:
                # SINGLE MARK CROP
                mark_w = self.preproc_config["calibration_marks"][0].get("width", tw)
                mark_h = self.preproc_config["calibration_marks"][0].get("height", th)
                
                start_x = int(m1_cx - (mark_w / 2))
                start_y = int(m1_cy - (mark_h / 2))
                
                end_x = start_x + int(mark_w)
                end_y = start_y + int(mark_h)
                
                h_img, w_img = frame.shape[:2]
                
                # Calculate visible region in source image
                src_x1 = max(0, start_x)
                src_y1 = max(0, start_y)
                src_x2 = min(w_img, end_x)
                src_y2 = min(h_img, end_y)
                
                # Calculate placement in destination (padded) canvas
                dst_x1 = src_x1 - start_x
                dst_y1 = src_y1 - start_y
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                
                aligned_img = np.zeros((int(mark_h), int(mark_w), 3), dtype=np.uint8)
                if src_x2 > src_x1 and src_y2 > src_y1:
                    aligned_img[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
                    
                self.output_size = (int(mark_w), int(mark_h))
                
                cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3)
                if debug:
                    base_name = mock_name if mock_name else f"{int(time.time()*100)}"
                    save_path = os.path.join(self.base_dir, "logs", f"debug_align_success_{base_name}.jpg")
                    cv2.imwrite(save_path, debug_img)
                    
            else:
                # MULTI-MARK ALIGNMENT
                ref_m1 = self.ref_mark_points[0]
                
                for i in range(1, len(self.ref_mark_points)):
                    tmpl = cv2.cvtColor(self.preproc_templates[i], cv2.COLOR_BGR2GRAY)
                    ref_m = self.ref_mark_points[i]
                    dx, dy = ref_m[0] - ref_m1[0], ref_m[1] - ref_m1[1]
                    exp_cx, exp_cy = m1_cx + dx, m1_cy + dy
                    
                    th_i, tw_i = tmpl.shape
                    search_pad_x, search_pad_y = 200, 200
                    w_box = tw_i + (search_pad_x * 2)
                    h_box = th_i + (search_pad_y * 2)
                    
                    roi_x, roi_y = int(exp_cx - w_box/2), int(exp_cy - h_box/2)
                    roi_rect = (roi_x, roi_y, w_box, h_box)
                    
                    cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + w_box, roi_y + h_box), (255, 0, 0), 2)
                    cv2.putText(debug_img, f"ROI M{i+1}", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
                    loc_i, score_i = find_mark(img_gray, tmpl, roi_rect)
                    
                    if score_i < 0.4:
                        if debug:
                            save_path = os.path.join(self.base_dir, "logs", f"debug_align_fail_m{i+1}_{int(time.time()*100)}.jpg")
                            cv2.imwrite(save_path, debug_img)
                        raise ValueError(f"Mark {i+1} not found. Alignment failed.")
                        
                    found_cx = loc_i[0] + tw_i//2
                    found_cy = loc_i[1] + th_i//2
                    found_marks.append([found_cx, found_cy])
                    
                    cv2.rectangle(debug_img, loc_i, (loc_i[0] + tw_i, loc_i[1] + th_i), (0, 255, 0), 2)
                    cv2.circle(debug_img, (int(found_cx), int(found_cy)), 5, (0, 0, 255), -1)
                    cv2.putText(debug_img, f"M{i+1}", (loc_i[0], loc_i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                if debug:
                    base_name = mock_name if mock_name else f"{int(time.time()*100)}"
                    save_path = os.path.join(self.base_dir, "logs", f"debug_align_success_{base_name}.jpg")
                    cv2.imwrite(save_path, debug_img)
                    
                input_marks = np.array(found_marks, dtype=np.float32)
                
                if len(input_marks) == 4:
                    H, _ = cv2.findHomography(input_marks, self.target_marks, cv2.RANSAC, 5.0)
                    if H is None:
                        raise ValueError("Homography Matrix calculation failed.")
                    aligned_img = cv2.warpPerspective(frame, H, self.output_size)
                elif len(input_marks) == 2:
                    M, _ = cv2.estimateAffinePartial2D(input_marks, self.target_marks)
                    if M is None:
                        raise ValueError("Affine Matrix calculation failed.")
                    aligned_img = cv2.warpAffine(frame, M, self.output_size)
                else:
                    raise ValueError(f"Unsupported number of marks for alignment: {len(input_marks)}")
        else:
            aligned_img = frame.copy()
        
        # --- 2. Shadow Removal ---
        if enable_shadow:
            shadow_removed = remove_shadows_divisive(aligned_img, sigma=50)
            if shadow_removed is None:
                shadow_removed = aligned_img
        else:
            shadow_removed = aligned_img
            
        # --- 2.5 Pre-Crop ---
        c_top = c_bottom = c_left = c_right = 0
        enable_pre_crop = prep_config.get("enable_pre_crop", False)
        if enable_pre_crop:
            crop_cfg = prep_config.get("pre_crop", {})
            c_top = int(crop_cfg.get("top", 0))
            c_bottom = int(crop_cfg.get("bottom", 0))
            c_left = int(crop_cfg.get("left", 0))
            c_right = int(crop_cfg.get("right", 0))
            
            h, w = shadow_removed.shape[:2]
            
            c_top = max(0, min(c_top, h - 1))
            c_bottom = max(0, min(c_bottom, h - 1 - c_top))
            c_left = max(0, min(c_left, w - 1))
            c_right = max(0, min(c_right, w - 1 - c_left))
            
            pre_cropped = shadow_removed[c_top:h-c_bottom, c_left:w-c_right]
        else:
            pre_cropped = shadow_removed

        # --- 3. Grayscale & CLAHE ---
        if enable_gray:
            gray_img = cv2.cvtColor(pre_cropped, cv2.COLOR_BGR2GRAY)
            
            if enable_clahe and not disable_clahe:
                processed_gray = self.clahe_obj.apply(gray_img)
            else:
                processed_gray = gray_img
                
            final_surface = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
        else:
            final_surface = pre_cropped
        
        # --- 4. Process Masks & Crops ---
        cropped_sub_images = []
        masked_surface = final_surface.copy()
        
        if enable_crop and self.mask_config:
            h_out, w_out = masked_surface.shape[:2]
            
            ref_size = self.mask_config.get("reference_image_size", {})
            ref_w = ref_size.get("width", w_out)
            ref_h = ref_size.get("height", h_out)
            
            offset_x = (w_out - ref_w) // 2
            offset_y = (h_out - ref_h) // 2
            
            mask_regions = self.mask_config.get("mask_regions", [])
            for region in mask_regions:
                rx = int(region["x"] + offset_x)
                ry = int(region["y"] + offset_y)
                rw = int(region.get("width", region.get("w", 0)))
                rh = int(region.get("height", region.get("h", 0)))
                
                crop_x1 = max(0, rx)
                crop_y1 = max(0, ry)
                crop_x2 = min(w_out, rx + rw)
                crop_y2 = min(h_out, ry + rh)
                
                if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                    mark_crop = final_surface[crop_y1:crop_y2, crop_x1:crop_x2]
                    raw_id = region.get("id", f"mask_{len(cropped_sub_images) + 1}")
                    crop_id = raw_id.replace("mask_", "crop_") if raw_id.startswith("mask_") else raw_id
                    cropped_sub_images.append((crop_id, mark_crop))
                    
                cv2.rectangle(masked_surface, (rx, ry), (rx+rw, ry+rh), (0, 0, 0), -1)
            
        # --- 5. Return images sequentially ---
        if debug:
            debug_out_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(debug_out_dir, exist_ok=True)
            base_name = mock_name if mock_name else f"{int(time.time()*100)}"
            cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_masked_surface.jpg"), masked_surface)
            if enable_pre_crop:
                cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_pre_crop.jpg"), final_surface)
            for crop_id, crop in cropped_sub_images:
                cv2.imwrite(os.path.join(debug_out_dir, f"preproc_{base_name}_{crop_id}.jpg"), crop)
        
        output_images.append(("inference_ready", masked_surface))
        if enable_pre_crop:
            output_images.append(("pre_crop", final_surface))
        output_images.extend(cropped_sub_images)
        
        if enable_align:
            output_images.append(("debug_align", debug_img))
        
        return finalize_output(output_images)
