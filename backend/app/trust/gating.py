from __future__ import annotations

import numpy as np
import cv2

def trust_gate(
    base: np.ndarray,
    raw_sr: np.ndarray,
    confidence: np.ndarray,
    smooth_transitions: bool = True
) -> np.ndarray:
    """
    Blends the high-resolution detail toward the conservative base in low-confidence regions:
    safe_sr = base + confidence * (raw_sr - base)
    
    In areas of high predicted risk (low confidence), detail smoothly reverts to the observed
    bicubic baseline to prevent hallucination artifacts from propagating to downstream workflows.
    """
    c = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    
    if smooth_transitions and c.shape[:2] == raw_sr.shape[:2]:
        # Light Gaussian blur to eliminate block artifacts or tile-edge stepping
        c = cv2.GaussianBlur(c, (5, 5), sigmaX=1.0)
        c = np.clip(c, 0.0, 1.0)
        
    while c.ndim < raw_sr.ndim:
        c = c[..., None]
        
    safe_sr = base + c * (raw_sr - base)
    return np.clip(safe_sr, 0.0, 1.0)
