from __future__ import annotations

from typing import List
from pathlib import Path
from fastapi import APIRouter, Response
from backend.app.schemas.scene import SceneSearchRequest, SceneSummary
from backend.app.services.scene_provider import get_scene_provider
from backend.app.core.errors import SceneNotFoundError
from backend.app.geo.raster import read_geotiff, array_to_png_bytes

router = APIRouter(tags=["Scenes"])

@router.post("/scenes/search", response_model=List[SceneSummary])
def search_scenes(req: SceneSearchRequest) -> List[SceneSummary]:
    provider = get_scene_provider()
    return provider.search(req)

@router.get("/scenes/{scene_id}/preview")
def get_scene_preview(scene_id: str) -> Response:
    provider = get_scene_provider()
    try:
        bundle = provider.get_scene_bundle(scene_id)
        preview_path = bundle.get("preview")
        if not preview_path or not Path(preview_path).exists():
            raise SceneNotFoundError(scene_id)
        arr, _ = read_geotiff(preview_path)
        png_bytes = array_to_png_bytes(arr)
        return Response(content=png_bytes, media_type="image/png")
    except KeyError:
        raise SceneNotFoundError(scene_id)

import uuid
import json
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile, File, Form
from backend.app.geo.raster import write_geotiff
from backend.app.db.database import insert_scene
from backend.app.core.config import settings

@router.get("/scenes", response_model=List[SceneSummary])
def list_scenes() -> List[SceneSummary]:
    provider = get_scene_provider()
    # Return all available demonstration scenes, prioritized
    req = SceneSearchRequest(lat=20.0, lon=78.0, radius_km=1000.0, start_date="2026-01-01", end_date="2026-03-01", max_cloud_percent=100.0, limit=50)
    scenes = provider.search(req)
    
    def scene_priority(s: SceneSummary) -> int:
        if s.id.startswith("upload_"):
            return 0
        if s.id.startswith("opensr_"):
            return 1
        return 2
    return sorted(scenes, key=scene_priority)

@router.post("/scenes/upload", response_model=SceneSummary)
async def upload_custom_scene(
    file: UploadFile = File(...),
    location_name: str = Form("Custom Uploaded Satellite AOI")
) -> SceneSummary:
    contents = await file.read()
    scene_id = f"upload_{uuid.uuid4().hex[:8]}"
    scene_dir = settings.scene_cache_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    filename_lower = file.filename.lower()
    r, g, b, nir = None, None, None, None
    
    # 1. Check if GeoTIFF
    if filename_lower.endswith((".tif", ".tiff")):
        try:
            import rasterio
            from rasterio.io import MemoryFile
            with MemoryFile(contents) as memfile:
                with memfile.open() as src:
                    count = src.count
                    # Read bands
                    if count >= 4:
                        data = src.read([1, 2, 3, 4]).astype(np.float32)
                        r_raw, g_raw, b_raw, nir_raw = data[0], data[1], data[2], data[3]
                    elif count == 3:
                        data = src.read([1, 2, 3]).astype(np.float32)
                        r_raw, g_raw, b_raw = data[0], data[1], data[2]
                        nir_raw = np.clip(1.5 * g_raw - 0.3 * r_raw + 0.1, 0.0, 1.0)
                    else:
                        data = src.read(1).astype(np.float32)
                        r_raw = g_raw = b_raw = data
                        nir_raw = data * 1.1
                    
                    # Normalize based on dynamic range
                    max_val = max(float(r_raw.max()), float(nir_raw.max()), 1.0)
                    if max_val > 255.0:
                        scale = 10000.0  # standard Sentinel-2 L2A DN
                    elif max_val > 1.0:
                        scale = 255.0
                    else:
                        scale = 1.0
                        
                    r = np.clip(r_raw / scale, 0.0, 1.0)
                    g = np.clip(g_raw / scale, 0.0, 1.0)
                    b = np.clip(b_raw / scale, 0.0, 1.0)
                    nir = np.clip(nir_raw / scale, 0.0, 1.0)
        except Exception:
            r = None

    # 2. Check if Numpy Array (.npy)
    if r is None and filename_lower.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(contents))
            if arr.ndim == 3 and arr.shape[0] >= 4:
                r, g, b, nir = arr[0], arr[1], arr[2], arr[3]
            elif arr.ndim == 3 and arr.shape[-1] >= 4:
                r, g, b, nir = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
                nir = np.clip(1.5 * g - 0.3 * r + 0.1, 0.0, 1.0)
            if r is not None and r.max() > 1.0:
                scale = 10000.0 if r.max() > 255.0 else 255.0
                r, g, b, nir = r / scale, g / scale, b / scale, nir / scale
        except Exception:
            r = None

    # 3. Fall back to standard image decoding (PNG / JPEG)
    if r is None:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = img.size
        dim = min(w, h)
        left = (w - dim) // 2
        top = (h - dim) // 2
        # Crop center square and resize to clean 128x128 resolution for 4x SR
        img_crop = img.crop((left, top, left + dim, top + dim)).resize((128, 128), Image.Resampling.LANCZOS)
        
        arr = np.array(img_crop, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        nir = np.clip(1.5 * g - 0.3 * r + 0.1, 0.0, 1.0).astype(np.float32)
        
    # Resize arrays to standard 128x128 if needed
    if r.shape != (128, 128):
        def resize_channel(ch: np.ndarray) -> np.ndarray:
            im = Image.fromarray((np.clip(ch, 0.0, 1.0) * 255.0).astype(np.uint8))
            return np.array(im.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        r = resize_channel(r)
        g = resize_channel(g)
        b = resize_channel(b)
        nir = resize_channel(nir)

    # Calculate Land Cover SCL
    denom_ndvi = nir + r + 1e-6
    ndvi = (nir - r) / denom_ndvi
    denom_ndwi = g + nir + 1e-6
    ndwi = (g - nir) / denom_ndwi
    
    scl = np.full(r.shape, 5, dtype=np.uint8)  # bare soil / urban
    scl[ndvi > 0.25] = 4                       # vegetation
    scl[ndwi > 0.08] = 6                       # water
    
    # Write GeoTIFF bands
    write_geotiff(scene_dir / "B04.tif", r, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    write_geotiff(scene_dir / "B03.tif", g, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    write_geotiff(scene_dir / "B02.tif", b, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    write_geotiff(scene_dir / "B08.tif", nir, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    write_geotiff(scene_dir / "SCL.tif", scl, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    
    rgb_disp = np.stack([r, g, b], axis=-1)
    write_geotiff(scene_dir / "preview_rgb.tif", rgb_disp, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    
    bbox = [77.0, 20.0, 77.1, 20.1]
    manifest = {
        "id": scene_id,
        "provider": "user_uploaded_image",
        "source_item_id": f"UPLOAD_{file.filename.upper()}",
        "location_name": location_name or file.filename,
        "acquisition_datetime": "2026-03-09T00:00:00Z",
        "cloud_cover": 0.0,
        "bands": ["B04", "B03", "B02", "B08"],
        "crs": "EPSG:32643",
        "source_gsd_m": 10.0,
        "target_gsd_m": 2.5,
        "bbox": bbox,
        "original_filename": file.filename
    }
    with open(scene_dir / "scene.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    insert_scene({
        "id": scene_id,
        "provider": "user_uploaded_image",
        "source_item_id": f"UPLOAD_{file.filename.upper()}",
        "acquisition_datetime": "2026-03-09T00:00:00Z",
        "cloud_cover": 0.0,
        "bbox_json": json.dumps(bbox),
        "crs": "EPSG:32643",
        "source_gsd_m": 10.0,
        "local_path": str(scene_dir),
        "provenance_json": json.dumps(manifest)
    })
    
    return SceneSummary(
        id=scene_id,
        provider="user_uploaded_image",
        source_item_id=f"UPLOAD_{file.filename.upper()}",
        acquisition_datetime="2026-03-09T00:00:00Z",
        cloud_cover=0.0,
        bbox=bbox,
        crs="EPSG:32643",
        source_gsd_m=10.0,
        preview_url=f"/api/v1/scenes/{scene_id}/preview",
        is_cached=True,
        location_name=location_name or file.filename,
        provenance=manifest
    )
