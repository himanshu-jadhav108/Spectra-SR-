from __future__ import annotations
import numpy as np

def trust_gate(base: np.ndarray, raw_sr: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Blend toward the conservative base in low-confidence regions."""
    c = np.clip(confidence, 0.0, 1.0)
    while c.ndim < raw_sr.ndim:
        c = c[..., None]
    return base + c * (raw_sr - base)
