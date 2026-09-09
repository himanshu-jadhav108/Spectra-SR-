from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SceneSearchRequest(BaseModel):
    lat: float
    lon: float
    radius_km: float = 10.0
    start_date: str
    end_date: str
    max_cloud_percent: float = 20.0
    limit: int = 5

class SceneSummary(BaseModel):
    id: str
    provider: str
    source_item_id: str
    acquisition_datetime: str
    cloud_cover: float
    bbox: List[float]
    crs: str
    source_gsd_m: float
    preview_url: Optional[str] = None
    is_cached: bool = True
    location_name: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
