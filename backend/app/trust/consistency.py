from __future__ import annotations

from typing import Dict, Any
import numpy as np

def compute_reconstruction_consistency(lr: np.ndarray, sr_down: np.ndarray) -> Dict[str, float]:
    """
    Computes consistency metrics between the original LR source and the degraded/downsampled SR.
    Returns MAE, RMSE, Pearson Correlation, and Max Drift.
    """
    a = np.asarray(lr, dtype=np.float32)
    b = np.asarray(sr_down, dtype=np.float32)
    
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    
    # Pearson correlation across all pixels
    a_flat = a.flatten()
    b_flat = b.flatten()
    if np.std(a_flat) > 1e-6 and np.std(b_flat) > 1e-6:
        corr_matrix = np.corrcoef(a_flat, b_flat)
        corr = float(corr_matrix[0, 1])
    else:
        corr = 1.0
        
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": np.clip(corr, -1.0, 1.0),
        "max_drift": float(np.max(np.abs(diff)))
    }
