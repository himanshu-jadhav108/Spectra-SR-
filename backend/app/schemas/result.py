from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TrustScorecard(BaseModel):
    spectral_drift: Dict[str, Any]
    spatial_alignment: Dict[str, Any]
    source_consistency: Dict[str, Any]
    predicted_risk: Dict[str, Any]
    validation_coverage: str
    disclaimer: str = "Predicted reliability — not ground-truth confirmation"

class JobResultResponse(BaseModel):
    job_id: str
    scene_id: str
    mode: str
    model: str
    status: str
    target_gsd_m: float = 2.5
    artifacts: Dict[str, str]
    trust_scorecard: TrustScorecard
    metrics: Dict[str, Any]
    agri_boundary_analysis: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any]
