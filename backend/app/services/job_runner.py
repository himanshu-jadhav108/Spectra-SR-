from __future__ import annotations

import asyncio
import traceback
import cv2
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from backend.app.db.database import (
    update_job_status,
    insert_metric,
    get_job
)
from backend.app.services.scene_provider import get_scene_provider
from backend.app.services.preprocessing import preprocess_scene
from backend.app.services.sr_engine import get_sr_engine
from backend.app.trust.features import extract_trust_feature_maps
from backend.app.trust.trust_head import TrustHead
from backend.app.trust.gating import trust_gate
from backend.app.trust.agri_boundary import analyze_field_boundaries
from backend.app.services.artifact_service import ArtifactService

async def run_pipeline_job(job_id: str, scene_id: str, model_name: str, enable_trust_gating: bool = True) -> None:
    """
    Executes the full 8-step Super-Resolution and Trust pipeline asynchronously:
    1. DISCOVER_SCENE
    2. PREPROCESS
    3. SR_INFERENCE
    4. TRUST_ANALYSIS
    5. TRUST_GATING
    6. AGRI_BOUNDARY
    7. ARTIFACT_WRITE
    8. COMPLETE
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    update_job_status(job_id, status="RUNNING", step="DISCOVER_SCENE", progress=5, started_at=now_iso)
    await asyncio.sleep(0.3)
    
    try:
        # Step 1: Discover & Load Scene
        update_job_status(job_id, status="RUNNING", step="DISCOVER_SCENE", progress=15)
        provider = get_scene_provider()
        scene_bundle = provider.get_scene_bundle(scene_id)
        
        # Step 2: Preprocessing
        update_job_status(job_id, status="RUNNING", step="PREPROCESS", progress=30)
        lr_stack, scl_mask, metadata = preprocess_scene(scene_bundle)
        await asyncio.sleep(0.2)
        
        # Step 3: SR Inference (SEN2SRLite 4× upsampling)
        update_job_status(job_id, status="RUNNING", step="SR_INFERENCE", progress=50)
        sr_engine = get_sr_engine()
        raw_sr = sr_engine.predict_tiled(lr_stack, tile_size=128, overlap=16)
        await asyncio.sleep(0.3)
        
        # Step 4: Trust Analysis (8 live feature maps & Trust Head)
        update_job_status(job_id, status="RUNNING", step="TRUST_ANALYSIS", progress=70)
        feature_maps, global_summary = extract_trust_feature_maps(lr_stack, raw_sr, scl_mask)
        
        trust_head = TrustHead()
        risk_map, confidence_map = trust_head.predict_risk_map(feature_maps)
        scorecard = trust_head.generate_scorecard(global_summary, risk_map, is_benchmark=False)
        await asyncio.sleep(0.2)
        
        # Step 5: Trust Gating
        update_job_status(job_id, status="RUNNING", step="TRUST_GATING", progress=85)
        h_sr, w_sr = raw_sr.shape[:2]
        # Conservative base: bicubic upsampled LR
        base_upsampled = cv2.resize(lr_stack, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
        
        if enable_trust_gating:
            safe_sr = trust_gate(base_upsampled, raw_sr, confidence_map)
        else:
            safe_sr = raw_sr.copy()
            
        # Step 6: Agricultural Field Boundary Utility
        agri_metrics, edge_map = analyze_field_boundaries(base_upsampled, raw_sr, safe_sr)
        
        # Step 7: Artifact Generation & Storage Registration
        update_job_status(job_id, status="RUNNING", step="ARTIFACT_WRITE", progress=95)
        
        metrics_payload = {
            "scorecard": scorecard,
            "global_summary": global_summary,
            "agri_boundary": agri_metrics,
            "input_gsd_m": 10.0,
            "target_gsd_m": 2.5
        }
        
        ArtifactService.save_job_artifacts(
            job_id=job_id,
            lr_stack=lr_stack,
            raw_sr=raw_sr,
            safe_sr=safe_sr,
            risk_map=risk_map,
            edge_map=edge_map,
            metadata=metadata,
            metrics=metrics_payload
        )
        
        # Insert summary metrics into SQLite
        insert_metric({
            "id": f"{job_id}_rmse",
            "job_id": job_id,
            "metric_name": "rmse_consistency",
            "value": global_summary["mean_rmse"],
            "unit": "reflectance",
            "scope": "live",
            "region_json": None,
            "metadata_json": None
        })
        insert_metric({
            "id": f"{job_id}_sad",
            "job_id": job_id,
            "metric_name": "spectral_angle_deg",
            "value": global_summary["mean_spectral_angle_deg"],
            "unit": "degrees",
            "scope": "live",
            "region_json": None,
            "metadata_json": None
        })
        insert_metric({
            "id": f"{job_id}_risk",
            "job_id": job_id,
            "metric_name": "predicted_risk_mean",
            "value": scorecard["predicted_risk"]["mean_score"],
            "unit": "score",
            "scope": "live",
            "region_json": None,
            "metadata_json": None
        })
        
        # Step 8: Complete
        comp_iso = datetime.now(timezone.utc).isoformat()
        update_job_status(job_id, status="COMPLETED", step="COMPLETE", progress=100, completed_at=comp_iso)
        
    except Exception as exc:
        traceback.print_exc()
        err_iso = datetime.now(timezone.utc).isoformat()
        update_job_status(
            job_id,
            status="FAILED",
            step="ERROR",
            progress=0,
            error_code="PIPELINE_EXECUTION_ERROR",
            error_message=str(exc),
            completed_at=err_iso
        )
