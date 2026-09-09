from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import cv2

def analyze_field_boundaries(
    lr_base: np.ndarray,
    raw_sr: np.ndarray,
    safe_sr: np.ndarray
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates edge enhancement for smallholder agricultural parcel boundaries.
    Computes gradient edge responses and boundary sharpness gain.
    Returns:
        metrics_dict: sharpness gain, edge density, and legal disclaimer
        edge_map_sr: binary/gradient edge map of super-resolved imagery
    """
    def get_gradient_map(img: np.ndarray) -> np.ndarray:
        # Use Green and NIR bands if 4-band, else RGB
        if img.ndim == 3 and img.shape[-1] >= 4:
            ch = (img[..., 1] + img[..., 3]) / 2.0  # (Green + NIR)/2 gives strong crop contrast
        elif img.ndim == 3 and img.shape[-1] >= 3:
            ch = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            ch = img
            
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return mag

    grad_lr = get_gradient_map(lr_base)
    grad_sr = get_gradient_map(raw_sr)
    grad_safe = get_gradient_map(safe_sr)
    
    # Sharpness: mean gradient energy
    sharpness_lr = float(np.mean(grad_lr))
    sharpness_sr = float(np.mean(grad_sr))
    sharpness_safe = float(np.mean(grad_safe))
    
    gain_pct = float(((sharpness_sr - sharpness_lr) / (sharpness_lr + 1e-6)) * 100.0)
    
    # Edge density threshold
    thresh = float(np.percentile(grad_sr, 80))
    edge_mask = (grad_sr > thresh).astype(np.uint8) * 255
    
    metrics = {
        "sharpness_gain_percent": round(gain_pct, 2),
        "lr_baseline_gradient": round(sharpness_lr, 4),
        "sr_boundary_gradient": round(sharpness_sr, 4),
        "safe_sr_boundary_gradient": round(sharpness_safe, 4),
        "edge_contrast_multiplier": round(sharpness_sr / (sharpness_lr + 1e-6), 2),
        "smallholder_parcel_clarity": "SUBSTANTIALLY_ENHANCED" if gain_pct > 30 else "MODERATELY_ENHANCED",
        "legal_disclaimer": "Agricultural field boundary enhancement is designed exclusively for agronomic farm management and crop monitoring. It does not constitute legal land title or cadastral survey proof."
    }
    
    return metrics, edge_mask
