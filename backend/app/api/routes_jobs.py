from __future__ import annotations

import uuid
import json
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, status
from backend.app.schemas.job import JobCreateRequest, JobResponse
from backend.app.schemas.result import JobResultResponse, TrustScorecard
from backend.app.db.database import (
    insert_job,
    get_job,
    get_job_artifacts,
    get_scene
)
from backend.app.core.errors import JobNotFoundError, SceneNotFoundError
from backend.app.services.job_runner import run_pipeline_job
from backend.app.core.config import settings

router = APIRouter(tags=["Jobs"])

@router.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(req: JobCreateRequest, background_tasks: BackgroundTasks) -> JobResponse:
    # Verify scene exists
    scene = get_scene(req.scene_id)
    if not scene:
        raise SceneNotFoundError(req.scene_id)
        
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()
    
    job_record = {
        "id": job_id,
        "scene_id": req.scene_id,
        "mode": req.mode,
        "model_name": req.model,
        "status": "QUEUED",
        "step": "QUEUED",
        "progress": 0,
        "error_code": None,
        "error_message": None,
        "started_at": None,
        "completed_at": None,
        "created_at": now_iso
    }
    insert_job(job_record)
    
    # Launch background processing task
    background_tasks.add_task(
        run_pipeline_job,
        job_id=job_id,
        scene_id=req.scene_id,
        model_name=req.model,
        enable_trust_gating=req.enable_trust_gating
    )
    
    return JobResponse(
        job_id=job_id,
        scene_id=req.scene_id,
        mode=req.mode,
        model=req.model,
        status="QUEUED",
        step="QUEUED",
        progress=0,
        created_at=now_iso
    )

@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_state(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)
        
    return JobResponse(
        job_id=job["id"],
        scene_id=job["scene_id"],
        mode=job["mode"],
        model=job["model_name"],
        status=job["status"],
        step=job["step"],
        progress=job["progress"],
        error_code=job.get("error_code"),
        error_message=job.get("error_message"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        created_at=job["created_at"]
    )

@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str) -> JobResultResponse:
    job = get_job(job_id)
    if not job:
        raise JobNotFoundError(job_id)
        
    job_dir = settings.artifact_root / job_id
    metrics_file = job_dir / "metrics" / "metrics.json"
    manifest_file = job_dir / "manifest.json"
    
    if not metrics_file.exists():
        # Job not finished or metrics missing
        raise JobNotFoundError(f"Result for {job_id} not available yet (Status: {job['status']})")
        
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        
    provenance_data = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            provenance_data = manifest.get("metadata", {})
            
    artifacts_map = {
        "lr_geotiff": f"/api/v1/jobs/{job_id}/artifacts/lr",
        "raw_sr_geotiff": f"/api/v1/jobs/{job_id}/artifacts/sr",
        "safe_sr_geotiff": f"/api/v1/jobs/{job_id}/artifacts/safe_sr",
        "risk_map_geotiff": f"/api/v1/jobs/{job_id}/artifacts/risk_map",
        "preview_lr": f"/api/v1/jobs/{job_id}/artifacts/preview_lr",
        "preview_raw_sr": f"/api/v1/jobs/{job_id}/artifacts/preview_raw_sr",
        "preview_safe_sr": f"/api/v1/jobs/{job_id}/artifacts/preview_safe_sr",
        "preview_risk_map": f"/api/v1/jobs/{job_id}/artifacts/preview_risk_map",
        "preview_agri_boundary": f"/api/v1/jobs/{job_id}/artifacts/preview_agri_boundary",
        "metrics_json": f"/api/v1/jobs/{job_id}/artifacts/metrics_json"
    }
    
    scorecard_data = metrics_data.get("scorecard", {})
    
    return JobResultResponse(
        job_id=job_id,
        scene_id=job["scene_id"],
        mode=job["mode"],
        model=job["model_name"],
        status=job["status"],
        target_gsd_m=2.5,
        artifacts=artifacts_map,
        trust_scorecard=TrustScorecard(**scorecard_data),
        metrics=metrics_data.get("global_summary", {}),
        agri_boundary_analysis=metrics_data.get("agri_boundary"),
        provenance=provenance_data
    )
