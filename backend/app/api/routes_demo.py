from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from backend.app.core.config import settings

router = APIRouter(tags=["Video Demo"])

DEMO_DIR = settings.project_root / "demo"
METADATA_PATH = DEMO_DIR / "metadata.json"

@router.get("/demo/scene")
def get_demo_scene() -> Dict[str, Any]:
    """
    Returns precomputed metadata and evidence signals for the dedicated Video Demo Scene.
    Guarantees deterministic, instant loading for the SIH 2026 video demonstration.
    """
    if not METADATA_PATH.exists():
        fallback_path = settings.project_root / "frontend" / "public" / "demo" / "metadata.json"
        if fallback_path.exists():
            with open(fallback_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="Demo scene metadata not found. Run prepare_demo_assets.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/demo/assets")
def get_demo_assets() -> Dict[str, str]:
    """
    Returns asset URLs for the dedicated Video Demo Mode.
    """
    return {
        "input_10m": "/demo/input_10m.png",
        "sr_raw_demo": "/demo/sr_raw_demo.png",
        "reliability_demo": "/demo/reliability_demo.png",
        "reliability_pure": "/demo/reliability_pure.png",
        "gated_sr_demo": "/demo/gated_sr_demo.png",
        "risk_demo": "/demo/risk_demo.png",
        "uncertainty_demo": "/demo/uncertainty_demo.png",
        "hr_reference": "/demo/hr_reference.png",
        "sr_gated_2p5m_tif": "/demo/sr_gated_2p5m.tif",
        "reliability_tif": "/demo/reliability.tif",
        "risk_mask_tif": "/demo/risk_mask.tif",
        "metadata_json": "/demo/metadata.json"
    }
