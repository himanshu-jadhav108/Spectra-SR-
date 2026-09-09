from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class BenchmarkSampleInfo(BaseModel):
    sample_id: str
    dataset_name: str
    index: int
    roi: Optional[str] = None
    description: str
    lr_shape: List[int]
    hr_shape: List[int]
    canonical_bands: List[str]
    lr_resolution: str
    hr_resolution: str
    provenance: str
    citation: str
    domain_tag: str

class BenchmarkDatasetInfo(BaseModel):
    name: str
    title: str
    description: str
    lr_resolution: str
    hr_resolution: str
    provenance: str
    citation: str
    sample_count: int
    samples: List[BenchmarkSampleInfo]

class BenchmarkRunRequest(BaseModel):
    dataset_name: str = "spain_crops"
    sample_id: str = "sample_000"
    model: str = "sen2sr_lite"

class BenchmarkMetrics(BaseModel):
    hallucination_score: float = Field(..., description="Fraction/percentage of false high frequencies")
    omission_score: float = Field(..., description="Fraction/percentage of missed real high frequencies")
    improvement_score: float = Field(..., description="Percentage fidelity improvement over conservative baseline")
    synthesis_score: float = Field(..., description="Structural frequency consistency ratio")
    reflectance_consistency_rmse: float = Field(..., description="RMSE error against harmonized ground truth")
    spectral_angle_deg: float = Field(..., description="Mean Spectral Angle Distance in degrees")
    spatial_consistency_corr: float = Field(..., description="Spatial gradient cross-correlation")
    hr_reference_gsd_m: float = 2.5
    lr_input_gsd_m: float = 10.0
    sr_output_gsd_m: float = 2.5
    opensr_native: Optional[Dict[str, Any]] = None

class BenchmarkRunResponse(BaseModel):
    benchmark_id: str
    job_id: str
    dataset_name: str
    dataset_version: str
    sample_id: str
    opensr_test_version: str
    status: str
    metrics: BenchmarkMetrics
    artifacts: Dict[str, str]
    scorecard: Optional[Dict[str, Any]] = None
    provenance: str = "ESAOpenSR opensr-test (Aybar et al., 2024)"
    citation: str = "Aybar et al. (2024), OpenSR-test: A comprehensive benchmark dataset and suite for satellite super-resolution"
    domain_tag: str = "Scientific Validation Lab (ESAOpenSR benchmark reference)"
    disclaimer: str = "OpenSR benchmark reference — not operational Indian Sentinel-2 mission imagery."

class TrustModelInfoResponse(BaseModel):
    model_type: str
    version: str
    training_source: str
    train_scenes: int
    test_scenes: int
    train_instances: int
    test_instances: int
    features: List[str]
    feature_importances: Dict[str, float]
    classes: List[str]
    metrics: Dict[str, Any]
    calibration_thresholds: Dict[str, float]
    disclaimer: str
