import pytest
import numpy as np
from pathlib import Path
from backend.app.db.database import init_db, list_all_scenes
from backend.app.services.seed_data import seed_cached_scenes
from backend.app.services.scene_provider import get_scene_provider
from backend.app.services.preprocessing import preprocess_scene
from backend.app.services.sr_engine import get_sr_engine
from backend.app.trust.spectral import spectral_angle_deg, compute_ndvi
from backend.app.trust.spatial import compute_phase_correlation
from backend.app.trust.features import extract_trust_feature_maps
from backend.app.trust.trust_head import TrustHead
from backend.app.trust.gating import trust_gate
from backend.app.trust.agri_boundary import analyze_field_boundaries
from backend.app.trust.benchmark import evaluate_hr_benchmark
from backend.app.services.artifact_service import ArtifactService

def test_pipeline_smoke():
    # 1. Initialize DB and Seed
    init_db()
    seed_cached_scenes()
    
    scenes = list_all_scenes()
    assert len(scenes) >= 3, "Expected at least 3 cached demonstration scenes"
    
    # 2. Provider and Preprocessing
    provider = get_scene_provider()
    bundle = provider.get_scene_bundle("scene_pune_periurban")
    assert bundle is not None
    
    lr_stack, scl_mask, metadata = preprocess_scene(bundle)
    assert lr_stack.shape[-1] == 4
    assert lr_stack.shape[0] == 128
    
    # 3. SEN2SRLite Inference
    engine = get_sr_engine()
    raw_sr = engine.predict(lr_stack)
    assert raw_sr.shape == (512, 512, 4)
    assert 0.0 <= raw_sr.min() <= raw_sr.max() <= 1.0
    
    # 4. Trust Features & Trust Head
    feature_maps, global_summary = extract_trust_feature_maps(lr_stack, raw_sr, scl_mask)
    assert "rmse_consistency" in feature_maps
    assert "spectral_angle_deg" in feature_maps
    
    trust_head = TrustHead()
    risk_map, confidence_map = trust_head.predict_risk_map(feature_maps)
    assert risk_map.shape == (512, 512)
    assert confidence_map.shape == (512, 512)
    
    scorecard = trust_head.generate_scorecard(global_summary, risk_map)
    assert "spectral_drift" in scorecard
    assert "disclaimer" in scorecard
    assert scorecard["disclaimer"] == "Predicted reliability — not ground-truth confirmation"
    
    # 5. Trust-Gating
    import cv2
    base_up = cv2.resize(lr_stack, (512, 512), interpolation=cv2.INTER_CUBIC)
    safe_sr = trust_gate(base_up, raw_sr, confidence_map)
    assert safe_sr.shape == (512, 512, 4)
    
    # 6. Agricultural Boundary Utility
    agri_metrics, edge_map = analyze_field_boundaries(base_up, raw_sr, safe_sr)
    assert "sharpness_gain_percent" in agri_metrics
    assert "legal_disclaimer" in agri_metrics
    
    # 7. Ground-truth Benchmark
    bench_metrics = evaluate_hr_benchmark(lr_base=base_up, sr=raw_sr, hr_ref=safe_sr)
    assert "hallucination_score" in bench_metrics
    assert "omission_score" in bench_metrics
    assert "improvement_score" in bench_metrics
