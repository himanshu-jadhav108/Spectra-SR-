from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, TiffImagePlugin

# Standard GeoTIFF Tag IDs
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_GEO_KEY_DIRECTORY = 34735

def read_geotiff(path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads a TIFF file and extracts array data along with GeoTIFF georeferencing tags."""
    img = Image.open(str(path))
    arr = np.array(img)
    
    meta: Dict[str, Any] = {
        "width": img.width,
        "height": img.height,
        "bands": arr.shape[-1] if arr.ndim > 2 else 1,
        "dtype": str(arr.dtype)
    }
    
    if hasattr(img, "tag_v2"):
        scale = img.tag_v2.get(TAG_MODEL_PIXEL_SCALE)
        tiepoint = img.tag_v2.get(TAG_MODEL_TIEPOINT)
        geokeys = img.tag_v2.get(TAG_GEO_KEY_DIRECTORY)
        if scale:
            meta["pixel_scale"] = list(scale)
        if tiepoint:
            meta["tiepoint"] = list(tiepoint)
        if geokeys:
            meta["geokeys"] = list(geokeys)
            
    return arr, meta

def write_geotiff(
    path: Path | str,
    data: np.ndarray,
    pixel_scale: Tuple[float, float, float] = (2.5, 2.5, 0.0),
    tiepoint: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 380000.0, 2040000.0, 0.0),
    epsg: int = 32643
) -> Path:
    """Writes a NumPy array to a valid GeoTIFF file with georeferencing metadata tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure proper data type and shape
    if data.dtype in (np.float32, np.float64):
        # If float normalized [0, 1] or reflectance [0, 10000], clip and scale to uint8 for preview-friendly GeoTIFF
        if data.max() <= 1.05:
            arr_uint8 = np.clip(data * 255.0, 0, 255).astype(np.uint8)
        else:
            arr_uint8 = np.clip(data, 0, 255).astype(np.uint8)
    else:
        arr_uint8 = data.astype(np.uint8)
        
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[TAG_MODEL_PIXEL_SCALE] = (float(pixel_scale[0]), float(pixel_scale[1]), float(pixel_scale[2]))
    info[TAG_MODEL_TIEPOINT] = tuple(float(x) for x in tiepoint)
    # GeoKeyDirectoryTag: standard GeoTIFF header + ProjectedCSTypeGeoKey (3072)
    info[TAG_GEO_KEY_DIRECTORY] = (1, 1, 0, 1, 3072, 0, 1, int(epsg))
    
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8)
    elif arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 3:
        img = Image.fromarray(arr_uint8)
    elif arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 4:
        img = Image.fromarray(arr_uint8)
    else:
        # Fallback to first 3 bands
        img = Image.fromarray(arr_uint8[..., :3])
        
    img.save(str(path), format="TIFF", tiffinfo=info)
    return path

def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converts a 2D or 3D NumPy array into compressed PNG bytes for web streaming."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.05:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        img = Image.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr[..., :3])
        
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def render_risk_colormap(risk_map: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """
    Renders risk [0, 1] into RGBA heatmap:
    - <= 0.25: Low Risk (emerald green #2E7D32)
    - 0.25 - 0.50: Medium Risk (amber #B7791F)
    - > 0.50: High Risk (crimson red #C62828)
    """
    h, w = risk_map.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Low risk (< 0.25): Green (46, 125, 50)
    mask_low = risk_map <= 0.25
    rgba[mask_low] = [46, 125, 50, int(255 * alpha * 0.4)]  # subtle green
    
    # Medium risk (0.25 - 0.50): Amber (183, 121, 31)
    mask_med = (risk_map > 0.25) & (risk_map <= 0.50)
    rgba[mask_med] = [183, 121, 31, int(255 * alpha * 0.7)]
    
    # High risk (> 0.50): Red (198, 40, 40)
    mask_high = risk_map > 0.50
    rgba[mask_high] = [198, 40, 40, int(255 * alpha * 0.9)]
    
    return rgba

def render_agri_boundary_overlay(rgb: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Overlays high-contrast boundary lines (amber/cyan) onto an RGB image."""
    result = rgb.copy()
    if result.dtype != np.uint8:
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    edge_mask = edges > 30
    # Cyan highlight for crisp boundary visibility [0, 230, 255]
    result[edge_mask] = [0, 230, 255]
    return result
