from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel

# Project Root is 3 levels up from backend/app/core
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseModel):
    app_name: str = "Spectra SR"
    app_env: str = os.getenv("APP_ENV", "development")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    
    # Paths
    project_root: Path = PROJECT_ROOT
    storage_root: Path = PROJECT_ROOT / "storage"
    database_path: Path = PROJECT_ROOT / "storage" / "db" / "spectra_sr.sqlite3"
    artifact_root: Path = PROJECT_ROOT / "storage" / "jobs"
    scene_cache_root: Path = PROJECT_ROOT / "storage" / "cache" / "scenes"
    model_root: Path = PROJECT_ROOT / "storage" / "models"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"
    
    # CDSE credentials
    cdse_client_id: str = os.getenv("CDSE_CLIENT_ID", "")
    cdse_client_secret: str = os.getenv("CDSE_CLIENT_SECRET", "")
    data_source: str = os.getenv("DATA_SOURCE", "local")
    
    # Processing Defaults
    device: str = os.getenv("DEVICE", "cuda")
    sr_model: str = os.getenv("SR_MODEL", "sen2sr_lite")
    target_gsd_m: float = float(os.getenv("TARGET_GSD_M", "2.5"))
    window_size: int = int(os.getenv("WINDOW_SIZE", "128"))
    window_overlap: int = int(os.getenv("WINDOW_OVERLAP", "16"))
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
    
    # Trust thresholds
    low_risk_threshold: float = float(os.getenv("LOW_RISK_THRESHOLD", "0.25"))
    medium_risk_threshold: float = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.50"))

settings = Settings()

# Ensure directories exist
settings.database_path.parent.mkdir(parents=True, exist_ok=True)
settings.artifact_root.mkdir(parents=True, exist_ok=True)
settings.scene_cache_root.mkdir(parents=True, exist_ok=True)
settings.model_root.mkdir(parents=True, exist_ok=True)
