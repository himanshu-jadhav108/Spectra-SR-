from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from backend.app.trust.spectral import spectral_angle_deg, compute_ndvi_drift
from backend.app.trust.spatial import compute_phase_correlation

FEATURE_NAMES = [
    "mae_consistency",
    "rmse_consistency",
    "spectral_angle_deg",
    "ndvi_drift",
    "gradient_delta",
    "variance_delta",
    "laplacian_energy",
    "phase_shift_magnitude"
]

def extract_trust_feature_maps(
    lr: np.ndarray,
    sr: np.ndarray,
    scl: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Extracts the 8 core live trust feature maps comparing observed Sentinel-2 LR
    and Super-Resolved SR.
    Returns:
        feature_maps: dictionary of 2D spatial feature maps at SR resolution
        global_summary: summary scalar values for the entire scene
    """
    h_sr, w_sr = sr.shape[:2]
    h_lr, w_lr = lr.shape[:2]
    
    # 1. Resample SR back to LR grid with box/area degradation (official degradation model)
    sr_down = cv2.resize(sr, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    
    # Common channels alignment (supports 3-band RGB and 4-band RGB+NIR)
    c_common = min(lr.shape[-1], sr_down.shape[-1])
    lr_c = lr[..., :c_common]
    sr_down_c = sr_down[..., :c_common]
    
    # Consistency: MAE & RMSE maps
    diff_lr = np.abs(lr_c - sr_down_c)
    mae_lr = np.mean(diff_lr, axis=-1)
    rmse_lr = np.sqrt(np.mean(diff_lr ** 2, axis=-1))
    
    # Upsample consistency maps to SR resolution for pixel/patch alignment
    mae_map = cv2.resize(mae_lr, (w_sr, h_sr), interpolation=cv2.INTER_LINEAR)
    rmse_map = cv2.resize(rmse_lr, (w_sr, h_sr), interpolation=cv2.INTER_LINEAR)
    
    # 2. Spectral Angle Distance map
    sad_lr = spectral_angle_deg(lr_c, sr_down_c)
    sad_map = cv2.resize(sad_lr, (w_sr, h_sr), interpolation=cv2.INTER_LINEAR)
    
    # 3. NDVI Drift map (where B04 and B08 exist)
    if lr.shape[-1] >= 4 and sr.shape[-1] >= 4:
        ndvi_drift_lr = compute_ndvi_drift(lr, sr_down)
    else:
        ndvi_drift_lr = np.zeros((h_lr, w_lr), dtype=np.float32)
    ndvi_drift_map = cv2.resize(ndvi_drift_lr, (w_sr, h_sr), interpolation=cv2.INTER_LINEAR)
    
    # 4. Spatial phase shift
    dx, dy, phase_mag, phase_status = compute_phase_correlation(lr_c, sr_down_c)
    phase_map = np.full((h_sr, w_sr), phase_mag, dtype=np.float32)
    
    # 5. Gradient & Laplacian energy deltas (SR vs bicubic baseline)
    lr_bicubic = cv2.resize(lr, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    gray_sr = cv2.cvtColor((sr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 if (sr.ndim == 3 and sr.shape[-1] >= 3) else sr
    gray_base = cv2.cvtColor((lr_bicubic * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 if (lr_bicubic.ndim == 3 and lr_bicubic.shape[-1] >= 3) else lr_bicubic
    
    grad_sr = cv2.magnitude(cv2.Sobel(gray_sr, cv2.CV_32F, 1, 0), cv2.Sobel(gray_sr, cv2.CV_32F, 0, 1))
    grad_base = cv2.magnitude(cv2.Sobel(gray_base, cv2.CV_32F, 1, 0), cv2.Sobel(gray_base, cv2.CV_32F, 0, 1))
    grad_delta_map = np.abs(grad_sr - grad_base)
    
    laplacian_sr = np.abs(cv2.Laplacian(gray_sr, cv2.CV_32F))
    
    # 6. Local variance delta (15x15 window)
    def local_variance(img: np.ndarray, size: int = 15) -> np.ndarray:
        mean = uniform_filter(img, size)
        sq_mean = uniform_filter(img ** 2, size)
        return np.maximum(0.0, sq_mean - mean ** 2)
        
    var_sr = local_variance(gray_sr)
    var_base = local_variance(gray_base)
    var_delta_map = np.abs(var_sr - var_base)
    
    feature_maps = {
        "mae_consistency": mae_map,
        "rmse_consistency": rmse_map,
        "spectral_angle_deg": sad_map,
        "ndvi_drift": ndvi_drift_map,
        "gradient_delta": grad_delta_map,
        "variance_delta": var_delta_map,
        "laplacian_energy": laplacian_sr,
        "phase_shift_magnitude": phase_map
    }
    
    global_summary = {
        "mean_mae": float(np.mean(mae_map)),
        "mean_rmse": float(np.mean(rmse_map)),
        "mean_spectral_angle_deg": float(np.mean(sad_map)),
        "max_spectral_angle_deg": float(np.max(sad_map)),
        "mean_ndvi_drift": float(np.mean(ndvi_drift_map)),
        "phase_shift_px": float(phase_mag),
        "phase_shift_status": phase_status,
        "mean_gradient_delta": float(np.mean(grad_delta_map))
    }
    
    return feature_maps, global_summary
