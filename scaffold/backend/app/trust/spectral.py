from __future__ import annotations
import numpy as np

def spectral_angle_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute spectral angle in degrees for arrays shaped (..., bands)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = np.clip(dot / (denom + eps), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
