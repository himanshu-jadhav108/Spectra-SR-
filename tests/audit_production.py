import io
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

def run_audit():
    print("=" * 60)
    print(" SPECTRA SR - PRODUCTION PRE-FLIGHT AUDIT (RENDER.COM READY) ")
    print("=" * 60)
    
    passed = 0
    total = 0
    
    with TestClient(app) as client:
        # 1. Health Endpoint
        total += 1
        r = client.get("/api/v1/health")
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        h_data = r.json()
        assert h_data["status"] == "ok"
        print(f"[{total}] PASS: /api/v1/health -> status: {h_data['status']}, GPU: {h_data.get('gpu')}")
        passed += 1

        # 2. Frontend HTML index
        total += 1
        r = client.get("/")
        assert r.status_code == 200
        assert "Spectra SR" in r.text
        assert "Mission Console" in r.text
        r_idx = client.get("/index.html")
        assert r_idx.status_code == 200
        print(f"[{total}] PASS: / and /index.html -> Content-Type: {r.headers.get('content-type')}, {len(r.content)} bytes")
        passed += 1

        # 3. CSS Stylesheet (Mobile responsive rules)
        total += 1
        r = client.get("/style.css")
        assert r.status_code == 200
        assert "@media (max-width: 768px)" in r.text
        assert "touch-action: none" in r.text
        print(f"[{total}] PASS: /style.css -> Includes mobile breakpoints & touch-action rules ({len(r.content)} bytes)")
        passed += 1

        # 4. Frontend JS
        total += 1
        r = client.get("/app.js")
        assert r.status_code == 200
        assert "initSplitSlider" in r.text
        assert "setPointerCapture" in r.text
        print(f"[{total}] PASS: /app.js -> Includes PointerCapture & responsive split slider ({len(r.content)} bytes)")
        passed += 1

        # 5. Brand Logo Static Asset
        total += 1
        r = client.get("/brand/spectra-sr-logo.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        print(f"[{total}] PASS: /brand/spectra-sr-logo.png -> Clean PNG asset delivered ({len(r.content)} bytes)")
        passed += 1

        # 6. Scene Discovery & Previews
        total += 1
        r = client.get("/api/v1/scenes")
        assert r.status_code == 200
        scenes = r.json()
        assert len(scenes) >= 6, f"Expected at least 6 scenes, found {len(scenes)}"
        r_pune = client.get("/api/v1/scenes/scene_pune_periurban/preview")
        assert r_pune.status_code == 200
        assert r_pune.headers["content-type"] == "image/png"
        print(f"[{total}] PASS: /api/v1/scenes -> Found {len(scenes)} scenes; Pune Peri-Urban preview verified ({len(r_pune.content)} bytes)")
        passed += 1

        # 7. Scene Upload (Custom User GeoTIFF / Image)
        total += 1
        buf = io.BytesIO()
        test_img = Image.new("RGB", (128, 128), color=(80, 140, 90))
        test_img.save(buf, format="PNG")
        buf.seek(0)
        r = client.post(
            "/api/v1/scenes/upload",
            files={"file": ("audit_field.png", buf, "image/png")},
            data={"dataset_origin": "custom_upload"}
        )
        assert r.status_code == 200
        up_data = r.json()
        custom_scene_id = up_data["id"]
        print(f"[{total}] PASS: /api/v1/scenes/upload -> Successfully uploaded & ingested scene: {custom_scene_id}")
        passed += 1

        # 8. Pipeline Execution (Job Creation + Background Processing)
        total += 1
        r = client.post("/api/v1/jobs", json={
            "scene_id": "opensr_spain_crops",
            "mode": "live",
            "model": "sen2sr_lite",
            "enable_trust_gating": True,
            "application": "agri_boundary"
        })
        assert r.status_code in (200, 202)
        job_id = r.json()["job_id"]
        print(f"[{total}] PASS: POST /api/v1/jobs -> Created pipeline job: {job_id}")
        passed += 1

        # 9. Job Status Polling
        total += 1
        r = client.get(f"/api/v1/jobs/{job_id}")
        assert r.status_code == 200
        j_state = r.json()
        assert j_state["status"] == "COMPLETED"
        assert j_state["progress"] == 100
        print(f"[{total}] PASS: GET /api/v1/jobs/{job_id} -> Status: {j_state['status']}, Step: {j_state['step']}")
        passed += 1

        # 10. Job Result & Trust Scorecard Validation
        total += 1
        r = client.get(f"/api/v1/jobs/{job_id}/result")
        assert r.status_code == 200
        res = r.json()
        assert res["target_gsd_m"] == 2.5
        assert "artifacts" in res
        assert "trust_scorecard" in res
        scorecard = res["trust_scorecard"]
        assert scorecard["disclaimer"] == "Predicted reliability — not ground-truth confirmation"
        print(f"[{total}] PASS: GET /api/v1/jobs/{job_id}/result -> Target GSD: {res['target_gsd_m']}m, SAD: {scorecard['spectral_drift']['value_deg']} deg")
        passed += 1

        # 11. Artifact File Downloads
        total += 1
        safe_preview_url = res["artifacts"]["preview_safe_sr"]
        r = client.get(safe_preview_url)
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        
        raw_tif_url = res["artifacts"]["raw_sr_geotiff"]
        r_tif = client.get(raw_tif_url)
        assert r_tif.status_code == 200
        print(f"[{total}] PASS: Artifact Endpoints -> PNG Previews and 2.5m GeoTIFFs downloadable")
        passed += 1

        # 12. Benchmark Datasets Listing
        total += 1
        r = client.get("/api/v1/benchmark/datasets")
        assert r.status_code == 200
        b_datasets = r.json()
        assert len(b_datasets) >= 5
        print(f"[{total}] PASS: /api/v1/benchmark/datasets -> Found {len(b_datasets)} ESAOpenSR benchmark datasets")
        passed += 1

        # 13. Benchmark Trust Model Calibration Info
        total += 1
        r = client.get("/api/v1/benchmark/trust-model")
        assert r.status_code == 200
        tm = r.json()
        assert "feature_importances" in tm
        print(f"[{total}] PASS: /api/v1/benchmark/trust-model -> AUC: {tm['metrics'].get('roc_auc_ovr')}, Features: {len(tm['feature_importances'])}")
        passed += 1

        # 14. Validation Benchmark Run
        total += 1
        r = client.post("/api/v1/benchmark/run", json={
            "dataset_name": "spain_crops",
            "sample_id": "sample_000",
            "model": "sen2sr_lite"
        })
        assert r.status_code in (200, 202)
        b_run = r.json()
        assert b_run["status"] == "COMPLETED"
        assert "metrics" in b_run
        ha_score = b_run["metrics"]["hallucination_score"]
        om_score = b_run["metrics"]["omission_score"]
        print(f"[{total}] PASS: POST /api/v1/benchmark/run -> Hallucination: {ha_score:.2f}%, Omission: {om_score:.2f}%")
        passed += 1

        # 15. Benchmark 3-Way Previews (LR, SR, HR)
        total += 1
        r_lr = client.get(b_run["artifacts"]["lr_preview"])
        r_sr = client.get(b_run["artifacts"]["sr_preview"])
        r_hr = client.get(b_run["artifacts"]["hr_preview"])
        assert r_lr.status_code == 200
        assert r_sr.status_code == 200
        assert r_hr.status_code == 200
        print(f"[{total}] PASS: 3-Way Synchronized Previews -> LR (10m), SR (2.5m), HR Ground Truth verified")
        passed += 1

    print("=" * 60)
    print(f" AUDIT COMPLETE: {passed}/{total} CHECKS PASSED (100% SUCCESS) ")
    print("=" * 60)

if __name__ == "__main__":
    run_audit()
