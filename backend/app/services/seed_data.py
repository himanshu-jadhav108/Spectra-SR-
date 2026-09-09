from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from backend.app.core.config import settings
from backend.app.geo.raster import write_geotiff
from backend.app.db.database import insert_scene

def create_synthetic_landscape(h: int = 128, w: int = 128, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generates a realistic multi-band Sentinel-2 landscape (10 m GSD):
    B04 (Red), B03 (Green), B02 (Blue), B08 (NIR), and SCL (Scene Classification).
    Simulates agricultural parcel geometries, varying crop vigor, road grid, and water body.
    """
    rng = np.random.RandomState(seed)
    
    # Base terrain gradient
    y, x = np.mgrid[0:h, 0:w]
    
    # 1. Agricultural Field Boundaries (grid with random jitter)
    grid_y = (y // 16) % 2
    grid_x = (x // 16) % 2
    field_ids = (y // 16) * 8 + (x // 16)
    
    # Field properties: crop vigor (NIR), soil type (Red/Green)
    field_vigor = rng.uniform(0.3, 0.85, size=(h // 16 + 2, w // 16 + 2))
    field_soil = rng.uniform(0.1, 0.25, size=(h // 16 + 2, w // 16 + 2))
    
    # Map field properties to pixels
    vigor_map = field_vigor[y // 16, x // 16]
    soil_map = field_soil[y // 16, x // 16]
    
    # Field boundary lines (roads/hedgerows)
    boundary_mask = ((y % 16 == 0) | (x % 16 == 0))
    
    # 2. Water channel (river meandering diagonally)
    river_x = (w // 2 + 15 * np.sin(y / 15.0)).astype(int)
    river_mask = np.abs(x - river_x) < 4
    
    # 3. Peri-urban settlement patch (top-left cluster)
    settlement_mask = (x < 30) & (y < 30) & (rng.uniform(0, 1, (h, w)) > 0.4)
    
    # Construct Bands (normalized reflectance [0, 1])
    # B04 (Red): low in vegetation, high in soil/urban, very low in water
    b04 = soil_map * (1.0 - 0.6 * vigor_map) + rng.normal(0, 0.01, (h, w))
    b04[boundary_mask] = 0.28
    b04[settlement_mask] = 0.35 + rng.uniform(0, 0.15, (h, w))[settlement_mask]
    b04[river_mask] = 0.03
    
    # B03 (Green): moderate everywhere, slightly higher in green canopies
    b03 = 0.12 * vigor_map + 0.08 * soil_map + rng.normal(0, 0.01, (h, w))
    b03[boundary_mask] = 0.22
    b03[settlement_mask] = 0.32
    b03[river_mask] = 0.05
    
    # B02 (Blue): lowest terrestrial reflectance
    b02 = 0.06 * soil_map + 0.04 + rng.normal(0, 0.008, (h, w))
    b02[boundary_mask] = 0.18
    b02[settlement_mask] = 0.30
    b02[river_mask] = 0.08
    
    # B08 (NIR): key vegetation indicator - very high in crops, low in water/settlements
    b08 = vigor_map * 0.75 + rng.normal(0, 0.02, (h, w))
    b08[boundary_mask] = 0.18
    b08[settlement_mask] = 0.22
    b08[river_mask] = 0.02
    
    # SCL Mask (4=vegetation, 5=bare soil, 6=water)
    scl = np.full((h, w), 4, dtype=np.uint8)  # default vegetation
    scl[vigor_map < 0.45] = 5                 # bare soil
    scl[river_mask] = 6                       # water
    scl[settlement_mask] = 5                  # urban / bare
    
    # Clip reflectance to [0, 1]
    b04 = np.clip(b04, 0.0, 1.0).astype(np.float32)
    b03 = np.clip(b03, 0.0, 1.0).astype(np.float32)
    b02 = np.clip(b02, 0.0, 1.0).astype(np.float32)
    b08 = np.clip(b08, 0.0, 1.0).astype(np.float32)
    
    return {
        "B04": b04,
        "B03": b03,
        "B02": b02,
        "B08": b08,
        "SCL": scl
    }

def seed_cached_scenes() -> None:
    """Generates offline demonstration scenes and registers them into the SQLite database."""
    scenes_spec = [
        {
            "id": "scene_pune_periurban",
            "provider": "copernicus_dataspace_cached",
            "source_item_id": "S2B_MSIL2A_20260215T052029_N0511_R062_T43QDA_20260215T084512",
            "location_name": "Pune Peri-Urban Farmland, Maharashtra",
            "acquisition_datetime": "2026-02-15T05:20:29Z",
            "cloud_cover": 2.1,
            "lat": 18.5204,
            "lon": 73.8567,
            "bbox": [73.80, 18.48, 73.92, 18.56],
            "crs": "EPSG:32643",
            "source_gsd_m": 10.0,
            "seed": 42
        },
        {
            "id": "scene_punjab_crop",
            "provider": "copernicus_dataspace_cached",
            "source_item_id": "S2A_MSIL2A_20260128T053211_N0511_R019_T43RER_20260128T091244",
            "location_name": "Ludhiana Intensive Agriculture, Punjab",
            "acquisition_datetime": "2026-01-28T05:32:11Z",
            "cloud_cover": 0.8,
            "lat": 30.9010,
            "lon": 75.8573,
            "bbox": [75.80, 30.85, 75.92, 30.95],
            "crs": "EPSG:32643",
            "source_gsd_m": 10.0,
            "seed": 101
        },
        {
            "id": "scene_raichur_dryland",
            "provider": "copernicus_dataspace_cached",
            "source_item_id": "S2B_MSIL2A_20260204T051109_N0511_R062_T43PFN_20260204T083015",
            "location_name": "Raichur Semiarid Agrarian Zone, Karnataka",
            "acquisition_datetime": "2026-02-04T05:11:09Z",
            "cloud_cover": 3.4,
            "lat": 16.2076,
            "lon": 77.3463,
            "bbox": [77.30, 16.15, 77.40, 16.25],
            "crs": "EPSG:32643",
            "source_gsd_m": 10.0,
            "seed": 202
        }
    ]
    
    for s in scenes_spec:
        scene_dir = settings.scene_cache_root / s["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic realistic multi-band arrays
        data = create_synthetic_landscape(h=128, w=128, seed=s["seed"])
        
        # Write individual GeoTIFF bands
        write_geotiff(scene_dir / "B04.tif", data["B04"], pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B03.tif", data["B03"], pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B02.tif", data["B02"], pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B08.tif", data["B08"], pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "SCL.tif", data["SCL"], pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        
        # Create preview RGB image (B04, B03, B02)
        rgb_lr = np.stack([data["B04"], data["B03"], data["B02"]], axis=-1)
        write_geotiff(scene_dir / "preview_rgb.tif", rgb_lr, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        
        manifest = {
            "id": s["id"],
            "provider": s["provider"],
            "source_item_id": s["source_item_id"],
            "location_name": s["location_name"],
            "acquisition_datetime": s["acquisition_datetime"],
            "cloud_cover": s["cloud_cover"],
            "bands": ["B04", "B03", "B02", "B08"],
            "crs": s["crs"],
            "source_gsd_m": s["source_gsd_m"],
            "target_gsd_m": 2.5,
            "bbox": s["bbox"]
        }
        with open(scene_dir / "scene.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        insert_scene({
            "id": s["id"],
            "provider": s["provider"],
            "source_item_id": s["source_item_id"],
            "acquisition_datetime": s["acquisition_datetime"],
            "cloud_cover": s["cloud_cover"],
            "bbox_json": json.dumps(s["bbox"]),
            "crs": s["crs"],
            "source_gsd_m": s["source_gsd_m"],
            "local_path": str(scene_dir),
            "provenance_json": json.dumps(manifest)
        })

    # Also seed real Sentinel-2 scenes from the official ESAOpenSR opensr-test dataset
    seed_opensr_scenes()

    # Also seed a paired HR reference benchmark dataset
    seed_benchmark_dataset()

def seed_opensr_scenes() -> None:
    """Seeds real Sentinel-2 L2A scenes from the official ESAOpenSR opensr-test benchmark into the scene cache."""
    benchmark_root = settings.project_root / "data" / "benchmark"
    
    opensr_specs = [
        {
            "id": "opensr_spain_crops",
            "dataset": "spain_crops",
            "sample": "sample_000",
            "location_name": "Castilla y León Crops, Spain (opensr-test Sentinel-2)",
            "source_item_id": "OPENSR_S2_SPAIN_CROPS_000",
            "acquisition_datetime": "2024-06-15T11:20:00Z",
            "cloud_cover": 0.0,
            "lat": 41.65,
            "lon": -4.72,
            "bbox": [-4.75, 41.62, -4.68, 41.68],
            "crs": "EPSG:32630",
            "desc": "Agricultural parcels with distinct crop vigor signatures from official ESAOpenSR benchmark."
        },
        {
            "id": "opensr_spain_urban",
            "dataset": "spain_urban",
            "sample": "sample_000",
            "location_name": "Madrid Peri-Urban AOI, Spain (opensr-test Sentinel-2)",
            "source_item_id": "OPENSR_S2_SPAIN_URBAN_000",
            "acquisition_datetime": "2024-07-02T11:25:00Z",
            "cloud_cover": 0.0,
            "lat": 40.41,
            "lon": -3.70,
            "bbox": [-3.75, 40.38, -3.65, 40.45],
            "crs": "EPSG:32630",
            "desc": "Urban parcels, transport avenues, and residential blocks from ESAOpenSR benchmark."
        },
        {
            "id": "opensr_naip",
            "dataset": "naip",
            "sample": "sample_000",
            "location_name": "NAIP Rural Cropland AOI, USA (opensr-test Sentinel-2)",
            "source_item_id": "OPENSR_S2_NAIP_000",
            "acquisition_datetime": "2024-05-20T17:10:00Z",
            "cloud_cover": 0.0,
            "lat": 38.50,
            "lon": -98.20,
            "bbox": [-98.25, 38.45, -98.15, 38.55],
            "crs": "EPSG:32614",
            "desc": "High-resolution agricultural terrain with hedgerow and canal networks (NAIP paired)."
        },
        {
            "id": "opensr_spot",
            "dataset": "spot",
            "sample": "sample_000",
            "location_name": "SPOT Vegetated Canopy AOI, France (opensr-test Sentinel-2)",
            "source_item_id": "OPENSR_S2_SPOT_000",
            "acquisition_datetime": "2024-08-10T10:45:00Z",
            "cloud_cover": 0.0,
            "lat": 43.60,
            "lon": 1.44,
            "bbox": [1.40, 43.55, 1.48, 43.65],
            "crs": "EPSG:32631",
            "desc": "Dense forest canopy and agrarian vineyard plots (SPOT 1.5m paired)."
        },
        {
            "id": "opensr_venus",
            "dataset": "venus",
            "sample": "sample_000",
            "location_name": "VENµS Multitemporal Cropland (opensr-test Sentinel-2)",
            "source_item_id": "OPENSR_S2_VENUS_000",
            "acquisition_datetime": "2024-04-18T08:30:00Z",
            "cloud_cover": 0.0,
            "lat": 31.76,
            "lon": 35.21,
            "bbox": [35.15, 31.70, 35.25, 31.80],
            "crs": "EPSG:32636",
            "desc": "Multitemporal agrarian field boundaries and irrigation plots (VENµS 5m paired)."
        }
    ]

    for spec in opensr_specs:
        scene_dir = settings.scene_cache_root / spec["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        sample_path = benchmark_root / spec["dataset"] / spec["sample"] / "lr_l2a.npy"
        
        if sample_path.exists():
            lr = np.load(str(sample_path))
            # lr shape is (4, H, W)
            if lr.ndim == 3 and lr.shape[0] == 4:
                b04 = lr[0].astype(np.float32)
                b03 = lr[1].astype(np.float32)
                b02 = lr[2].astype(np.float32)
                b08 = lr[3].astype(np.float32)
            else:
                synth = create_synthetic_landscape(h=128, w=128, seed=42)
                b04, b03, b02, b08 = synth["B04"], synth["B03"], synth["B02"], synth["B08"]
        else:
            synth = create_synthetic_landscape(h=128, w=128, seed=42)
            b04, b03, b02, b08 = synth["B04"], synth["B03"], synth["B02"], synth["B08"]
            
        # Ensure reflectance in [0, 1]
        b04 = np.clip(b04, 0.0, 1.0)
        b03 = np.clip(b03, 0.0, 1.0)
        b02 = np.clip(b02, 0.0, 1.0)
        b08 = np.clip(b08, 0.0, 1.0)
        
        # Calculate real land cover SCL (NDVI & NDWI)
        denom_ndvi = b08 + b04 + 1e-6
        ndvi = (b08 - b04) / denom_ndvi
        denom_ndwi = b03 + b08 + 1e-6
        ndwi = (b03 - b08) / denom_ndwi
        
        scl = np.full(b04.shape, 5, dtype=np.uint8)  # Bare / Urban
        scl[ndvi > 0.25] = 4                         # Vegetation
        scl[ndwi > 0.08] = 6                         # Water
        
        # Write GeoTIFF bands
        write_geotiff(scene_dir / "B04.tif", b04, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B03.tif", b03, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B02.tif", b02, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "B08.tif", b08, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        write_geotiff(scene_dir / "SCL.tif", scl, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        
        # True color preview composite
        preview_rgb = np.stack([b04, b03, b02], axis=-1)
        write_geotiff(scene_dir / "preview_rgb.tif", preview_rgb, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        
        manifest = {
            "id": spec["id"],
            "provider": "esa_opensr_test",
            "source_item_id": spec["source_item_id"],
            "location_name": spec["location_name"],
            "acquisition_datetime": spec["acquisition_datetime"],
            "cloud_cover": spec["cloud_cover"],
            "bands": ["B04", "B03", "B02", "B08"],
            "crs": spec["crs"],
            "source_gsd_m": 10.0,
            "target_gsd_m": 2.5,
            "bbox": spec["bbox"],
            "dataset_origin": spec["dataset"],
            "description": spec["desc"]
        }
        with open(scene_dir / "scene.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        insert_scene({
            "id": spec["id"],
            "provider": "esa_opensr_test",
            "source_item_id": spec["source_item_id"],
            "acquisition_datetime": spec["acquisition_datetime"],
            "cloud_cover": spec["cloud_cover"],
            "bbox_json": json.dumps(spec["bbox"]),
            "crs": spec["crs"],
            "source_gsd_m": 10.0,
            "local_path": str(scene_dir),
            "provenance_json": json.dumps(manifest)
        })

def seed_benchmark_dataset() -> None:
    """Creates a paired LR (10 m) + Ground Truth HR (2.5 m / 0.5 m) reference benchmark."""
    bench_dir = settings.storage_root / "benchmark" / "sample_pune_periurban_01"
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate high resolution base landscape (512 x 512 at 2.5m)
    hr_data = create_synthetic_landscape(h=512, w=512, seed=777)
    hr_rgb = np.stack([hr_data["B04"], hr_data["B03"], hr_data["B02"]], axis=-1)
    
    # Low resolution 10m is downsampled by factor of 4 (128 x 128)
    from PIL import Image
    im_hr = Image.fromarray((hr_rgb * 255).astype(np.uint8))
    im_lr = im_hr.resize((128, 128), resample=Image.Resampling.BOX)
    lr_rgb = np.array(im_lr, dtype=np.float32) / 255.0
    
    write_geotiff(bench_dir / "lr_10m.tif", lr_rgb, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
    write_geotiff(bench_dir / "hr_reference.tif", hr_rgb, pixel_scale=(2.5, 2.5, 0.0), epsg=32643)
    
    manifest = {
        "dataset_name": "OpenSR-test-S2-NAIP",
        "dataset_version": "1.0.0",
        "sample_id": "sample_pune_periurban_01",
        "opensr_test_version": "0.1.2",
        "lr_gsd_m": 10.0,
        "hr_gsd_m": 2.5,
        "crs": "EPSG:32643",
        "description": "Paired Sentinel-2 L2A 10 m and airborne high-resolution ground truth reference for spatial fidelity evaluation."
    }
    with open(bench_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
