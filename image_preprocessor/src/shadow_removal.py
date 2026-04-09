import cv2
import numpy as np


def remove_shadows_divisive(image, sigma=30):
    """
    Remove shadows using background division (Divisive Normalization).
    Works in LAB color space to isolate the Luminance channel.

    Args:
        image: BGR image (numpy array)
        sigma: Gaussian blur sigma for background estimation (larger = smoother bg)

    Returns:
        BGR image with shadows removed, or None if input is None.
    """
    if image is None:
        return None

    # Work in LAB space to separate Luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Smooth L channel to get background light map
    bg_l = cv2.GaussianBlur(l, (0, 0), sigma)

    # Avoid division by zero
    bg_l = np.maximum(bg_l, 1)

    # Divide: Result = Image / Background * mean_luminance
    mean_l = np.mean(l)
    result_l = (l.astype(np.float32) / bg_l.astype(np.float32)) * mean_l

    # Clip and convert back
    result_l = np.clip(result_l, 0, 255).astype(np.uint8)

    # Merge channels and convert back to BGR
    result_lab = cv2.merge((result_l, a, b))
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
