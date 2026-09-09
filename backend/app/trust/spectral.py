from __future__ import annotations

import numpy as np

def spectral_angle_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute spectral angle in degrees for arrays shaped (..., bands).
    SAD(x) = arccos( dot(a,b) / (||a|| ||b|| + eps) )
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.clip(dot / (denom + eps), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def compute_ndvi(bands: np.ndarray, b04_idx: int = 0, b08_idx: int = 3, eps: float = 1e-6) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index: (B08 - B04) / (B08 + B04 + eps)."""
    red = bands[..., b04_idx]
    nir = bands[..., b08_idx]
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0)

def compute_ndvi_drift(lr_bands: np.ndarray, sr_downsampled: np.ndarray) -> np.ndarray:
    """Compute absolute NDVI drift between source LR and downsampled SR."""
    lr_ndvi = compute_ndvi(lr_bands)
    sr_ndvi = compute_ndvi(sr_downsampled)
    return np.abs(sr_ndvi - lr_ndvi)
