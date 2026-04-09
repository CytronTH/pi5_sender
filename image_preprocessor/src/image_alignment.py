import cv2
import numpy as np
import json
import os


def load_calibration(cam_id="cam0", base_dir=None):
    """
    Load calibration points and template images for a given camera.

    Args:
        cam_id: Camera ID string e.g. "cam0" or "cam1"
        base_dir: Root directory of the preprocessor (where configs/ lives).
                  Defaults to the parent of this file's directory.

    Returns:
        (config dict, list of template images as BGR numpy arrays)

    Raises:
        ValueError if calibration file or templates are not found.
    """
    if base_dir is None:
        # src/ -> image_preprocessor/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = os.path.join(base_dir, "configs", f"{cam_id}_calibration_points.json")

    if not os.path.exists(config_path):
        raise ValueError(
            f"CRITICAL ERROR: configs/{cam_id}_calibration_points.json not found "
            f"at expected location: {config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    templates_dir = os.path.join(base_dir, "configs", "templates")
    templates = []

    for mark in config["calibration_marks"]:
        path = os.path.basename(mark["template"])
        template_file = os.path.join(templates_dir, path)

        if not os.path.exists(template_file):
            raise ValueError(
                f"CRITICAL ERROR: Template image '{path}' not found at {template_file}. "
                f"Please ensure all template images exist before running."
            )

        img = cv2.imread(template_file)
        if img is None:
            raise ValueError(f"CRITICAL ERROR: Failed to read template image: {template_file}")
        templates.append(img)

    return config, templates


def find_mark_full(img_gray, template):
    """
    Optimized hierarchical template matching.
    Scales down the image to find an approximate match, then refines at full resolution.

    Args:
        img_gray: Grayscale image to search in.
        template: Grayscale template to find.

    Returns:
        ((x, y), score) - top-left location and match confidence.
    """
    scale = 0.25  # Search at 25% resolution first (16x faster)
    h_tmpl, w_tmpl = template.shape
    h_img, w_img = img_gray.shape

    # For small templates, skip pyramid and do direct search
    if h_tmpl < 40 or w_tmpl < 40:
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_loc, max_val

    small_img = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_tmpl = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    res_small = cv2.matchTemplate(small_img, small_tmpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc_small = cv2.minMaxLoc(res_small)

    # Convert back to full resolution
    approx_x = int(max_loc_small[0] / scale)
    approx_y = int(max_loc_small[1] / scale)

    # Create a tight ROI around the approximate match
    margin_x = int(w_tmpl * 0.15)
    margin_y = int(h_tmpl * 0.15)

    roi_x = max(0, approx_x - margin_x)
    roi_y = max(0, approx_y - margin_y)
    roi_w = min(w_img - roi_x, w_tmpl + 2 * margin_x)
    roi_h = min(h_img - roi_y, h_tmpl + 2 * margin_y)

    roi = img_gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Fallback to full search if ROI is too small
    if roi.shape[0] < h_tmpl or roi.shape[1] < w_tmpl:
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_loc, max_val

    res_full = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val_full, _, max_loc_full = cv2.minMaxLoc(res_full)

    final_x = max_loc_full[0] + roi_x
    final_y = max_loc_full[1] + roi_y
    return (final_x, final_y), max_val_full


def find_mark(img_gray, template, search_roi=None):
    """
    Finds a template in an image, optionally within a restricted area.

    Args:
        img_gray: Grayscale image to search in.
        template: Grayscale template to find.
        search_roi: Optional (x, y, w, h) bounding box to restrict the search area.
                    If None, searches the full image using pyramid optimization.

    Returns:
        ((x, y), score) - top-left location in global image coords and match confidence.
    """
    h_img, w_img = img_gray.shape
    h_tmpl, w_tmpl = template.shape

    # Clip oversized template
    if h_tmpl > h_img or w_tmpl > w_img:
        print("WARNING: Template is larger than image! Clipping template...", flush=True)
        template = template[:min(h_tmpl, h_img), :min(w_tmpl, w_img)]
        h_tmpl, w_tmpl = template.shape

    if search_roi:
        x, y, w, h = search_roi
        x = max(0, x)
        y = max(0, y)
        w = min(w_img - x, w)
        h = min(h_img - y, h)

        roi = img_gray[y:y + h, x:x + w]
        if roi.size == 0 or roi.shape[0] < h_tmpl or roi.shape[1] < w_tmpl:
            print("WARNING: ROI too small for template. Falling back to full image search.", flush=True)
            return find_mark_full(img_gray, template)

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        global_x = max_loc[0] + x
        global_y = max_loc[1] + y
        return (global_x, global_y), max_val
    else:
        return find_mark_full(img_gray, template)


def calculate_canonical_targets(config):
    """
    Calculate the 'ideal' mark positions in the output (warped) image space.

    This maps calibration mark positions through the same perspective transform
    that maps calibration corners to the standard output rectangle.

    Args:
        config: Calibration config dict with 'calibration_marks',
                'calibration_corners', and optional 'padding'/'padding_x'/'padding_y'.

    Returns:
        (target_marks, output_size) where:
            target_marks: numpy array (N, 2) of mark positions in output space
            output_size: (width, height) tuple of the output image
    """
    calib_marks = config["calibration_marks"]
    calib_corners = config["calibration_corners"]

    pts_marks = np.array(
        [[m.get("center_x", m["x"]), m.get("center_y", m["y"])] for m in calib_marks],
        dtype=np.float32
    )
    pts_corners = np.array([[c["x"], c["y"]] for c in calib_corners], dtype=np.float32)

    # Compute output size from corner distances
    w_top = np.linalg.norm(pts_corners[0] - pts_corners[1])
    w_bot = np.linalg.norm(pts_corners[3] - pts_corners[2])
    out_w = int((w_top + w_bot) / 2)

    h_left = np.linalg.norm(pts_corners[0] - pts_corners[3])
    h_right = np.linalg.norm(pts_corners[1] - pts_corners[2])
    out_h = int((h_left + h_right) / 2)

    # Padding support (same as original pipeline)
    padding = config.get("padding", 0)
    pad_x = config.get("padding_x", padding)
    pad_y = config.get("padding_y", padding)

    out_w += pad_x * 2
    out_h += pad_y * 2

    if out_w <= 0 or out_h <= 0:
        out_w -= pad_x * 2
        out_h -= pad_y * 2
        pad_x = 0
        pad_y = 0

    target_corners = np.array([
        [pad_x, pad_y],
        [out_w - pad_x - 1, pad_y],
        [out_w - pad_x - 1, out_h - pad_y - 1],
        [pad_x, out_h - pad_y - 1]
    ], dtype=np.float32)

    M_calib_to_out = cv2.getPerspectiveTransform(pts_corners, target_corners)

    pts_marks_reshaped = pts_marks.reshape(-1, 1, 2)
    target_marks = cv2.perspectiveTransform(pts_marks_reshaped, M_calib_to_out)
    target_marks = target_marks.reshape(-1, 2)

    return target_marks, (out_w, out_h)
