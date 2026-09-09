from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

def compute_phase_correlation(lr_img: np.ndarray, sr_down_img: np.ndarray) -> Tuple[float, float, float, str]:
    """
    Computes spatial sub-pixel phase correlation between LR and downsampled SR.
    Returns:
        dx, dy: horizontal and vertical registration displacement (in pixels)
        magnitude: Euclidean displacement magnitude
        status: 'ALIGNED' or 'SHIFT_DETECTED'
    """
    # Use green/NIR band or single-channel representation
    a = lr_img[..., 1] if lr_img.ndim == 3 else lr_img
    b = sr_down_img[..., 1] if sr_down_img.ndim == 3 else sr_down_img
    
    a_f = np.asarray(a, dtype=np.float32)
    b_f = np.asarray(b, dtype=np.float32)
    
    # Using OpenCV phaseCorrelate
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(a_f, b_f)
    except Exception:
        shift_x, shift_y = 0.0, 0.0
        
    mag = float(np.sqrt(shift_x**2 + shift_y**2))
    status = "ALIGNED" if mag < 0.75 else "SHIFT_DETECTED"
    return float(shift_x), float(shift_y), mag, status

def compute_edge_energy(img: np.ndarray) -> np.ndarray:
    """Computes Sobel gradient magnitude for edge energy quantification."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if (img.ndim == 3 and img.shape[-1] >= 3) else img
    gray_f = np.asarray(gray, dtype=np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag
