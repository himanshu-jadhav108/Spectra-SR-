from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field

class JobCreateRequest(BaseModel):
    scene_id: str
    mode: Literal["live", "benchmark"] = "live"
    model: Literal["sen2sr_lite", "ldsr_s2"] = "sen2sr_lite"
    enable_trust_gating: bool = True
    application: Optional[Literal["agri_boundary"]] = None

class JobResponse(BaseModel):
    job_id: str
    scene_id: str
    mode: str
    model: str
    status: str
    step: str
    progress: int
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
