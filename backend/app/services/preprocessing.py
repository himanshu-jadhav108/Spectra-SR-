from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
from backend.app.geo.raster import read_geotiff

def preprocess_scene(scene_bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses Sentinel-2 L2A scene:
    1. Loads bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
    2. Stacks in standardized order [B04, B03, B02, B08]
    3. Extracts SCL mask
    4. Records spatial shape, nodata fraction, and provenance
    Returns:
        lr_stack: shape (H, W, 4) float32 in [0, 1]
        scl_mask: shape (H, W) uint8
        metadata: dict with CRS, pixel_scale, tiepoint, etc.
    """
    bands_dict = scene_bundle["bands"]
    
    b04_arr, meta = read_geotiff(bands_dict["B04"])
    b03_arr, _ = read_geotiff(bands_dict["B03"])
    b02_arr, _ = read_geotiff(bands_dict["B02"])
    b08_arr, _ = read_geotiff(bands_dict["B08"])
    
    # Check for SCL mask
    if "SCL" in bands_dict and bands_dict["SCL"].exists():
        scl_arr, _ = read_geotiff(bands_dict["SCL"])
    else:
        scl_arr = np.zeros(b04_arr.shape[:2], dtype=np.uint8)
        
    def to_float(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        elif arr.dtype in (np.uint16, np.int32):
            return np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
        
    b04_f = to_float(b04_arr)
    b03_f = to_float(b03_arr)
    b02_f = to_float(b02_arr)
    b08_f = to_float(b08_arr)
    
    lr_stack = np.stack([b04_f, b03_f, b02_f, b08_f], axis=-1)
    
    h, w, c = lr_stack.shape
    nodata_fraction = float(np.mean(lr_stack <= 1e-6))
    
    meta_out = {
        "scene_id": scene_bundle["id"],
        "provider": scene_bundle["provider"],
        "source_item_id": scene_bundle["source_item_id"],
        "crs": meta.get("crs", "EPSG:32643"),
        "pixel_scale": meta.get("pixel_scale", [10.0, 10.0, 0.0]),
        "tiepoint": meta.get("tiepoint", [0.0, 0.0, 0.0, 380000.0, 2040000.0, 0.0]),
        "height": h,
        "width": w,
        "bands": ["B04", "B03", "B02", "B08"],
        "nodata_fraction": nodata_fraction,
        "source_gsd_m": 10.0,
        "target_gsd_m": 2.5
    }
    
    return lr_stack, scl_arr, meta_out
