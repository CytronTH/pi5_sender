import cv2
import numpy as np
import os
import time
import json

from .image_alignment import (
    load_calibration,
    calculate_canonical_targets,
    find_mark,
)
from .shadow_removal import remove_shadows_divisive


class ImagePipeline:
    """
    Standalone image pre-processing pipeline.

    Replicates the exact pre-processing steps performed in the pi5_sender
    camera node before images are sent over TCP. Can be configured per camera
    using the same JSON config files used by the original system.

    Pipeline steps (in order):
        1. Alignment      - Template-matching based image registration
        2. Shadow Removal - Divisive normalization in LAB space
        2.5 Pre-Crop      - Edge trimming (top/bottom/left/right pixels)
        3. Grayscale + CLAHE - Contrast enhancement
        4. Mask Cropping  - Extract regions of interest, blank them on surface

    Usage:
        pipeline = ImagePipeline(camera_id=0, base_dir="/path/to/image_preprocessor")
        pipeline.load_configs(enable_align=True, enable_crop=True)
        results = pipeline.process_frame(frame, prep_config)
        for img_id, img_data in results:
            cv2.imwrite(f"output_{img_id}.jpg", img_data)
    """

    def __init__(self, camera_id: int, base_dir: str):
        """
        Args:
            camera_id: Integer camera index (0 or 1).
            base_dir: Absolute path to the root of image_preprocessor
                      (the folder that contains configs/ and src/).
        """
        self.camera_id = camera_id
        self.base_dir = base_dir

        self.preproc_config = None
        self.preproc_templates = None
        self.mask_config = None
        self.target_marks = None
        self.output_size = None
        self.ref_mark_points = None

        # Create CLAHE object once and reuse (clipLimit and tileGridSize match original)
        self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Config Loading
    # ------------------------------------------------------------------

    def load_configs(self, enable_align: bool = True, enable_crop: bool = True):
        """
        Load alignment calibration and crop mask configs from disk.

        Must be called before process_frame().

        Args:
            enable_align: Whether to load alignment calibration.
            enable_crop:  Whether to load mask/crop config.

        Raises:
            RuntimeError if any required config cannot be loaded.
        """
        # 1. Load Alignment Config
        if enable_align:
            try:
                self.preproc_config, self.preproc_templates = load_calibration(
                    f"cam{self.camera_id}", base_dir=self.base_dir
                )

                if len(self.preproc_config.get("calibration_marks", [])) > 1:
                    self.target_marks, self.output_size = calculate_canonical_targets(
                        self.preproc_config
                    )
                else:
                    self.target_marks = None
                    self.output_size = None

                self.ref_mark_points = np.array(
                    [
                        [m.get("center_x", m["x"]), m.get("center_y", m["y"])]
                        for m in self.preproc_config["calibration_marks"]
                    ],
                    dtype=np.float32,
                )

            except Exception as e:
                raise RuntimeError(
                    f"CRITICAL ERROR: Failed to load calibration config: {e}. "
                    f"Ensure configs/cam{self.camera_id}_calibration_points.json "
                    f"and configs/templates/ exist."
                )

        # 2. Load Mask/Crop Config
        if enable_crop:
            mask_config_path = os.path.join(
                self.base_dir, "configs", f"cam{self.camera_id}_crop_regions.json"
            )
            try:
                with open(mask_config_path, "r") as f:
                    self.mask_config = json.load(f)
            except FileNotFoundError:
                self.mask_config = None
            except Exception as e:
                raise RuntimeError(
                    f"CRITICAL ERROR: Failed to load mask config from {mask_config_path}: {e}"
                )

    # ------------------------------------------------------------------
    # Frame Processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame,
        prep_config: dict,
        debug: bool = False,
        mock_name: str = None,
        disable_clahe: bool = False,
    ):
        """
        Run the full pre-processing pipeline on a single frame.

        Args:
            frame:         BGR numpy array (the raw captured image).
            prep_config:   Dict with boolean flags controlling which steps run.
                           Expected keys:
                             enable_alignment      (bool, default True)
                             enable_shadow_removal (bool, default True)
                             enable_pre_crop       (bool, default False)
                             pre_crop              (dict: top/bottom/left/right)
                             enable_grayscale      (bool, default True)
                             enable_clahe          (bool, default True)
                             enable_box_cropping   (bool, default True)
            debug:         If True, save intermediate images to logs/.
            mock_name:     Optional filename label used in debug output names.
            disable_clahe: Force-disable CLAHE even if enabled in prep_config.

        Returns:
            List of (image_id: str, image_data: numpy array) tuples.

            Possible IDs:
              "reference_raw"  - Original input frame (always included)
              "inference_ready"- Final processed surface (always in normal mode)
              "pre_crop"     - Post-CLAHE image before masking (if enable_pre_crop)
              "crop_X"       - Individual cropped regions from mask config
              "debug_align"  - Alignment debug overlay (only if debug=True)
        """
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

        # --- Auto-resize input frame to match calibration reference ---
        h_in, w_in = frame.shape[:2]
        if self.mask_config and "reference_image_size" in self.mask_config:
            ref_w = self.mask_config["reference_image_size"].get("width", w_in)
            ref_h = self.mask_config["reference_image_size"].get("height", h_in)
            if (w_in != ref_w or h_in != ref_h) and ref_w > 0 and ref_h > 0:
                # To handle aspect ratio differences (e.g., 4056x3040 vs 1280x720)
                # We perform a center-crop to match the target aspect ratio first
                target_aspect = ref_w / ref_h
                in_aspect = w_in / h_in
                
                if abs(target_aspect - in_aspect) > 0.01:
                    if in_aspect > target_aspect:
                        # Input is wider than target aspect ratio (crop left/right)
                        new_w = int(h_in * target_aspect)
                        offset = (w_in - new_w) // 2
                        frame = frame[:, offset:offset+new_w]
                    else:
                        # Input is taller than target aspect ratio (crop top/bottom)
                        new_h = int(w_in / target_aspect)
                        offset = (h_in - new_h) // 2
                        frame = frame[offset:offset+new_h, :]
                
                # Now resize directly (aspect ratios match)
                frame = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

        # Always include the raw source frame first (after potential resize to match output domain)
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

        # --- Fallback: No calibration data, return raw ---
        if fallback_to_raw:
            output_images.append(("inference_ready", frame))
            calib_out_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(calib_out_dir, exist_ok=True)
            save_path = os.path.join(
                calib_out_dir, f"cam{self.camera_id}_calibration_target.jpg"
            )
            cv2.imwrite(save_path, frame)
            return finalize_output(output_images)

        # -------------------------------------------------------
        # Step 1: Alignment
        # -------------------------------------------------------
        debug_img = frame.copy()

        if enable_align:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_marks = []

            # --- Find Mark 1 (full image search) ---
            tmpl1 = cv2.cvtColor(self.preproc_templates[0], cv2.COLOR_BGR2GRAY)
            loc, score = find_mark(img_gray, tmpl1)

            if score < 0.2:
                if debug:
                    save_path = os.path.join(
                        self.base_dir, "logs",
                        f"debug_align_fail_m1_{int(time.time() * 100)}.jpg"
                    )
                    cv2.imwrite(save_path, debug_img)
                raise ValueError("Mark 1 not found (score < 0.2). Alignment failed.")

            th, tw = tmpl1.shape
            m1_cx, m1_cy = loc[0] + tw // 2, loc[1] + th // 2
            found_marks.append([m1_cx, m1_cy])

            cv2.rectangle(debug_img, loc, (loc[0] + tw, loc[1] + th), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(m1_cx), int(m1_cy)), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_img, "M1", (loc[0], loc[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            is_single_mark = (
                len(self.preproc_templates) == 1 if self.preproc_templates else False
            )

            if is_single_mark:
                # --- Single Mark: crop a fixed bounding box around the mark ---
                mark_w = self.preproc_config["calibration_marks"][0].get("width", tw)
                mark_h = self.preproc_config["calibration_marks"][0].get("height", th)

                start_x = int(m1_cx - (mark_w / 2))
                start_y = int(m1_cy - (mark_h / 2))
                end_x = start_x + int(mark_w)
                end_y = start_y + int(mark_h)

                h_img, w_img = frame.shape[:2]

                src_x1 = max(0, start_x)
                src_y1 = max(0, start_y)
                src_x2 = min(w_img, end_x)
                src_y2 = min(h_img, end_y)

                dst_x1 = src_x1 - start_x
                dst_y1 = src_y1 - start_y
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                aligned_img = np.zeros((int(mark_h), int(mark_w), 3), dtype=np.uint8)
                if src_x2 > src_x1 and src_y2 > src_y1:
                    aligned_img[dst_y1:dst_y2, dst_x1:dst_x2] = frame[
                        src_y1:src_y2, src_x1:src_x2
                    ]

                self.output_size = (int(mark_w), int(mark_h))

                cv2.rectangle(
                    debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 3
                )
                if debug:
                    base_name = mock_name if mock_name else f"{int(time.time() * 100)}"
                    cv2.imwrite(
                        os.path.join(
                            self.base_dir, "logs", f"debug_align_success_{base_name}.jpg"
                        ),
                        debug_img,
                    )

            else:
                # --- Multi-Mark: find remaining marks in ROI, then warp ---
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

                    roi_x, roi_y = int(exp_cx - w_box / 2), int(exp_cy - h_box / 2)
                    roi_rect = (roi_x, roi_y, w_box, h_box)

                    cv2.rectangle(
                        debug_img, (roi_x, roi_y), (roi_x + w_box, roi_y + h_box),
                        (255, 0, 0), 2
                    )
                    cv2.putText(
                        debug_img, f"ROI M{i + 1}", (roi_x, roi_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                    )

                    loc_i, score_i = find_mark(img_gray, tmpl, roi_rect)

                    if score_i < 0.4:
                        if debug:
                            cv2.imwrite(
                                os.path.join(
                                    self.base_dir, "logs",
                                    f"debug_align_fail_m{i + 1}_{int(time.time() * 100)}.jpg"
                                ),
                                debug_img,
                            )
                        raise ValueError(
                            f"Mark {i + 1} not found (score < 0.4). Alignment failed."
                        )

                    found_cx = loc_i[0] + tw_i // 2
                    found_cy = loc_i[1] + th_i // 2
                    found_marks.append([found_cx, found_cy])

                    cv2.rectangle(
                        debug_img, loc_i, (loc_i[0] + tw_i, loc_i[1] + th_i),
                        (0, 255, 0), 2
                    )
                    cv2.circle(debug_img, (int(found_cx), int(found_cy)), 5, (0, 0, 255), -1)
                    cv2.putText(
                        debug_img, f"M{i + 1}", (loc_i[0], loc_i[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                    )

                if debug:
                    base_name = mock_name if mock_name else f"{int(time.time() * 100)}"
                    cv2.imwrite(
                        os.path.join(
                            self.base_dir, "logs", f"debug_align_success_{base_name}.jpg"
                        ),
                        debug_img,
                    )

                input_marks = np.array(found_marks, dtype=np.float32)

                if len(input_marks) == 4:
                    H, _ = cv2.findHomography(input_marks, self.target_marks, cv2.RANSAC, 5.0)
                    if H is None:
                        raise ValueError("Homography matrix calculation failed.")
                    aligned_img = cv2.warpPerspective(frame, H, self.output_size)
                elif len(input_marks) == 2:
                    M, _ = cv2.estimateAffinePartial2D(input_marks, self.target_marks)
                    if M is None:
                        raise ValueError("Affine matrix calculation failed.")
                    aligned_img = cv2.warpAffine(frame, M, self.output_size)
                else:
                    raise ValueError(
                        f"Unsupported number of marks for alignment: {len(input_marks)}"
                    )
        else:
            aligned_img = frame.copy()

        # -------------------------------------------------------
        # Step 2: Shadow Removal
        # -------------------------------------------------------
        if enable_shadow:
            shadow_removed = remove_shadows_divisive(aligned_img, sigma=50)
            if shadow_removed is None:
                shadow_removed = aligned_img
        else:
            shadow_removed = aligned_img

        # -------------------------------------------------------
        # Step 2.5: Pre-Crop (trim edges by pixel count)
        # -------------------------------------------------------
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

            pre_cropped = shadow_removed[c_top: h - c_bottom, c_left: w - c_right]
        else:
            pre_cropped = shadow_removed

        # -------------------------------------------------------
        # Step 3: Grayscale + CLAHE
        # -------------------------------------------------------
        if enable_gray:
            gray_img = cv2.cvtColor(pre_cropped, cv2.COLOR_BGR2GRAY)

            if enable_clahe and not disable_clahe:
                processed_gray = self.clahe_obj.apply(gray_img)
            else:
                processed_gray = gray_img

            final_surface = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
        else:
            final_surface = pre_cropped

        # -------------------------------------------------------
        # Step 4: Mask-based Cropping
        # -------------------------------------------------------
        cropped_sub_images = []
        masked_surface = final_surface.copy()

        if enable_crop and self.mask_config:
            h_out, w_out = masked_surface.shape[:2]

            ref_size = self.mask_config.get("reference_image_size", {})
            ref_w = ref_size.get("width", w_out)
            ref_h = ref_size.get("height", h_out)

            # Center-align mask coordinates if image size differs from reference
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
                    crop_id = (
                        raw_id.replace("mask_", "crop_")
                        if raw_id.startswith("mask_")
                        else raw_id
                    )
                    cropped_sub_images.append((crop_id, mark_crop))

                # Black out the region on the surface image
                cv2.rectangle(
                    masked_surface, (rx, ry), (rx + rw, ry + rh), (0, 0, 0), -1
                )

        # -------------------------------------------------------
        # Step 5: Debug output (optional)
        # -------------------------------------------------------
        if debug:
            debug_out_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(debug_out_dir, exist_ok=True)
            base_name = mock_name if mock_name else f"{int(time.time() * 100)}"

            cv2.imwrite(
                os.path.join(debug_out_dir, f"preproc_{base_name}_masked_surface.jpg"),
                masked_surface,
            )
            if enable_pre_crop:
                cv2.imwrite(
                    os.path.join(debug_out_dir, f"preproc_{base_name}_pre_crop.jpg"),
                    final_surface,
                )
            for crop_id, crop in cropped_sub_images:
                cv2.imwrite(
                    os.path.join(debug_out_dir, f"preproc_{base_name}_{crop_id}.jpg"),
                    crop,
                )

        # -------------------------------------------------------
        # Assemble final output list
        # -------------------------------------------------------
        output_images.append(("inference_ready", masked_surface))
        if enable_pre_crop:
            output_images.append(("pre_crop", final_surface))
        output_images.extend(cropped_sub_images)

        if enable_align:
            # debug_align is included for inspection but NOT sent over TCP in the original
            output_images.append(("debug_align", debug_img))

        return finalize_output(output_images)
