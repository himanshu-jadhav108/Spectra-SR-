import urllib.request
import json
import time

def main():
    print("=== SPECTRA SR LIVE SMOKE VERIFICATION ===")

    # 1. Health
    with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health") as r:
        data = json.loads(r.read())
        print("[1] Health OK:", data)

    # 2. Brand Logo
    with urllib.request.urlopen("http://127.0.0.1:8000/brand/spectra-sr-logo.png") as r:
        print(f"[2] Logo HTTP Status: {r.status}, Content-Length: {len(r.read())} bytes")

    # 3. Scenes
    with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/scenes") as r:
        scenes = json.loads(r.read())
        print(f"[3] Discovered {len(scenes)} Indian scenes:")
        for s in scenes:
            print(f"    * {s['id']}: {s['location_name']} (Cloud: {s['cloud_cover']}%)")

    # 4. Create Job
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/jobs",
        data=json.dumps({
            "scene_id": "scene_pune_periurban",
            "mode": "live",
            "model": "sen2sr_lite",
            "enable_trust_gating": True,
            "application": "agri_boundary"
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        job = json.loads(r.read())
        job_id = job["job_id"]
        print(f"[4] Job Created: {job_id} (Status: {job['status']})")

    # Poll for completion
    for i in range(40):
        time.sleep(0.4)
        with urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}") as r:
            st = json.loads(r.read())
            print(f"    -> Progress: {st['progress']}% | Step: {st['step']}")
            if st["status"] == "COMPLETED":
                print("    Job completed successfully!")
                break

    # 5. Fetch Result
    with urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/result") as r:
        res = json.loads(r.read())
        print("\n[5] Job Result & Trust Scorecard:")
        print(f"    Target GSD: {res['target_gsd_m']} m")
        print("    Trust Scorecard:")
        print(json.dumps(res["trust_scorecard"], indent=6))
        print("    Agri Boundary Clarity:")
        print(json.dumps(res["agri_boundary_analysis"], indent=6))
        print(f"    Artifacts Generated ({len(res['artifacts'])} total):")
        for k, v in res["artifacts"].items():
            print(f"      * {k}: {v}")

    # 6. Benchmark Run
    bench_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/benchmark/run",
        data=json.dumps({
            "dataset_name": "OpenSR-test-S2-NAIP",
            "sample_id": "sample_pune_periurban_01",
            "model": "sen2sr_lite"
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(bench_req) as r:
        bench_res = json.loads(r.read())
        print("\n[6] Ground-Truth Benchmark Validation Metrics:")
        print(json.dumps(bench_res["metrics"], indent=6))
        print(f"    Disclaimer: {bench_res['disclaimer']}")

    print("\n=== ALL SPECTRA SR VERIFICATIONS PASSED ===")

if __name__ == "__main__":
    main()
