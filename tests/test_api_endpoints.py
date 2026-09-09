import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["app"] == "Spectra SR"
    assert "gpu" in data

def test_static_assets(client):
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"Spectra SR" in res_index.content
    
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert b"--bg-app" in res_css.content
    
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert b"Spectra SR" in res_js.content
    
    res_logo = client.get("/brand/spectra-sr-logo.png")
    assert res_logo.status_code == 200
    assert res_logo.headers["content-type"] == "image/png"

def test_scene_search(client):
    res = client.get("/api/v1/scenes")
    assert res.status_code == 200
    scenes = res.json()
    assert len(scenes) >= 5
    
    scene_ids = [s["id"] for s in scenes]
    assert "opensr_spain_crops" in scene_ids
    assert "opensr_spain_urban" in scene_ids
    assert "opensr_naip" in scene_ids
    assert "opensr_spot" in scene_ids
    assert "opensr_venus" in scene_ids
    assert "scene_pune_periurban" in scene_ids

def test_scene_upload_png(client):
    import io
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(buf, format="PNG")
    buf.seek(0)
    
    res = client.post(
        "/api/v1/scenes/upload",
        files={"file": ("my_crop.png", buf, "image/png")},
        data={"location_name": "Test Crop"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["id"].startswith("upload_")
    assert data["location_name"] == "Test Crop"
    assert data["is_cached"] is True

def test_scene_upload_geotiff_and_run_job(client):
    import io
    import numpy as np
    from backend.app.geo.raster import write_geotiff
    import tempfile
    
    # Create sample 4-band GeoTIFF
    arr = np.random.uniform(0.1, 0.8, size=(64, 64, 4)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_geotiff(tmp_path, arr, pixel_scale=(10.0, 10.0, 0.0), epsg=32643)
        with open(tmp_path, "rb") as f:
            tif_bytes = f.read()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            
    res = client.post(
        "/api/v1/scenes/upload",
        files={"file": ("sentinel2_test.tif", io.BytesIO(tif_bytes), "image/tiff")},
        data={"location_name": "Uploaded S2 GeoTIFF"}
    )
    assert res.status_code == 200
    scene_data = res.json()
    uploaded_scene_id = scene_data["id"]
    assert uploaded_scene_id.startswith("upload_")
    
    # Run pipeline job on the uploaded scene
    res_job = client.post("/api/v1/jobs", json={
        "scene_id": uploaded_scene_id,
        "mode": "live",
        "model": "sen2sr_lite",
        "enable_trust_gating": True,
        "application": "agri_boundary"
    })
    assert res_job.status_code == 202
    job_id = res_job.json()["job_id"]
    
    # Verify completed result
    res_result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res_result.status_code == 200
    result_json = res_result.json()
    assert result_json["target_gsd_m"] == 2.5
    assert "preview_safe_sr" in result_json["artifacts"]

def test_opensr_job_and_result(client):
    # Run pipeline on an OpenSR-Test dataset scene
    res = client.post("/api/v1/jobs", json={
        "scene_id": "opensr_spain_crops",
        "mode": "live",
        "model": "sen2sr_lite",
        "enable_trust_gating": True,
        "application": "agri_boundary"
    })
    assert res.status_code == 202
    job_id = res.json()["job_id"]
    
    res_result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res_result.status_code == 200
    data = res_result.json()
    assert data["target_gsd_m"] == 2.5
    assert "artifacts" in data
    assert "trust_scorecard" in data
    assert data["trust_scorecard"]["disclaimer"] == "Predicted reliability — not ground-truth confirmation"

def test_end_to_end_job_and_result(client):
    # 1. Create Job
    res = client.post("/api/v1/jobs", json={
        "scene_id": "scene_pune_periurban",
        "mode": "live",
        "model": "sen2sr_lite",
        "enable_trust_gating": True,
        "application": "agri_boundary"
    })
    assert res.status_code == 202
    job_info = res.json()
    job_id = job_info["job_id"]
    assert job_id.startswith("job_")
    
    # Background task ran during client.post
    # 2. Check Job State
    res_state = client.get(f"/api/v1/jobs/{job_id}")
    assert res_state.status_code == 200
    state_data = res_state.json()
    assert state_data["status"] == "COMPLETED"
    assert state_data["progress"] == 100
    
    # 3. Check Result
    res_result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res_result.status_code == 200
    result_data = res_result.json()
    assert result_data["target_gsd_m"] == 2.5
    assert "artifacts" in result_data
    assert "trust_scorecard" in result_data
    
    # Verify mandatory disclaimer
    scorecard = result_data["trust_scorecard"]
    assert scorecard["disclaimer"] == "Predicted reliability — not ground-truth confirmation"
    assert scorecard["validation_coverage"] == "LIVE-PREDICTED"
    
    # Verify Agri boundary analysis
    assert result_data["agri_boundary_analysis"] is not None
    assert "sharpness_gain_percent" in result_data["agri_boundary_analysis"]
    assert "legal_disclaimer" in result_data["agri_boundary_analysis"]

def test_benchmark_run(client):
    res = client.post("/api/v1/benchmark/run", json={
        "dataset_name": "OpenSR-test-S2-NAIP",
        "sample_id": "sample_pune_periurban_01",
        "model": "sen2sr_lite"
    })
    assert res.status_code in (200, 202)
    bench_data = res.json()
    assert bench_data["status"] == "COMPLETED"
    assert "metrics" in bench_data
    
    metrics = bench_data["metrics"]
    assert "hallucination_score" in metrics
    assert "omission_score" in metrics
    assert "improvement_score" in metrics
    assert "synthesis_score" in metrics
    assert "reflectance_consistency_rmse" in metrics
    assert "spectral_angle_deg" in metrics
    assert "spatial_consistency_corr" in metrics
    assert "OpenSR benchmark" in bench_data["disclaimer"] or "benchmark" in bench_data["disclaimer"].lower()
