from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse
from backend.app.core.config import settings
from backend.app.core.errors import ArtifactNotFoundError

router = APIRouter(tags=["Artifacts"])

ARTIFACT_MAP = {
    "lr": ("input/lr_10m.tif", "image/tiff"),
    "sr": ("output/raw_sr_2.5m.tif", "image/tiff"),
    "safe_sr": ("output/safe_sr_2.5m.tif", "image/tiff"),
    "risk_map": ("trust/risk_map_2.5m.tif", "image/tiff"),
    "preview_lr": ("previews/preview_lr.png", "image/png"),
    "preview_raw_sr": ("previews/preview_raw_sr.png", "image/png"),
    "preview_safe_sr": ("previews/preview_safe_sr.png", "image/png"),
    "preview_risk_map": ("previews/preview_risk_map.png", "image/png"),
    "preview_agri_boundary": ("previews/preview_agri_boundary.png", "image/png"),
    "metrics_json": ("metrics/metrics.json", "application/json"),
    "manifest_json": ("manifest.json", "application/json")
}

@router.get("/jobs/{job_id}/artifacts/{artifact_type}")
def get_job_artifact(job_id: str, artifact_type: str):
    if artifact_type not in ARTIFACT_MAP:
        raise ArtifactNotFoundError(artifact_type, job_id)
        
    rel_path, media_type = ARTIFACT_MAP[artifact_type]
    file_path = settings.artifact_root / job_id / rel_path
    
    if not file_path.exists():
        raise ArtifactNotFoundError(artifact_type, job_id)
        
    filename = f"{job_id}_{artifact_type}_{file_path.name}"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )
