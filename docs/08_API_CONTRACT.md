# REST API Contract

Base path: `/api/v1`

## GET /health
Response:
```json
{
  "status": "ok",
  "gpu": true,
  "model_loaded": true,
  "version": "0.1.0"
}
```

## POST /scenes/search
Request:
```json
{
  "lat": 18.5204,
  "lon": 73.8567,
  "radius_km": 10,
  "start_date": "2026-08-01",
  "end_date": "2026-09-01",
  "max_cloud_percent": 20,
  "limit": 5
}
```
Response:
```json
{
  "items": [
    {
      "scene_id": "...",
      "source_item_id": "...",
      "datetime": "...",
      "cloud_cover": 7.4,
      "bbox": [0,0,0,0],
      "available_bands": ["B02","B03","B04","B08","SCL"]
    }
  ]
}
```

## POST /jobs
Request:
```json
{
  "scene_id": "...",
  "mode": "live",
  "model": "sen2sr_lite",
  "enable_trust_gating": true,
  "application": "agri_boundary"
}
```
Response `202`:
```json
{
  "job_id": "...",
  "status": "queued"
}
```

## GET /jobs/{job_id}
Response:
```json
{
  "job_id": "...",
  "status": "running",
  "step": "TRUST_ANALYSIS",
  "progress": 72,
  "model": "SEN2SRLite",
  "artifacts": []
}
```

## GET /jobs/{job_id}/result
Response:
```json
{
  "job_id": "...",
  "status": "completed",
  "provenance": {
    "scene_id": "...",
    "acquisition_datetime": "...",
    "source_gsd_m": 10,
    "target_gsd_m": 2.5
  },
  "trust": {
    "spectral_drift_deg": 1.8,
    "spatial_shift_px": 0.02,
    "reconstruction_rmse": 0.006,
    "predicted_risk_mean": 0.18,
    "risk_band": "LOW"
  },
  "artifacts": {
    "sr": "/api/v1/artifacts/...",
    "safe_sr": "/api/v1/artifacts/...",
    "risk_map": "/api/v1/artifacts/...",
    "metrics": "/api/v1/artifacts/..."
  }
}
```

## POST /benchmark/run
Request:
```json
{
  "dataset": "opensr_test",
  "sample_id": "...",
  "model": "sen2sr_lite"
}
```
Response:
```json
{
  "job_id": "...",
  "status": "queued"
}
```

## GET /benchmark/{job_id}
Return:
- dataset/version
- sample id
- SR model/version
- benchmark package/version
- metric values
- artifact paths

## GET /artifacts/{artifact_id}
Streams local artifact with correct MIME type.

## API rules
- All mutating operations return job ids.
- All job ids are opaque strings.
- Validation errors return 4xx with structured JSON.
- Processing errors become job `FAILED`, not a 500 page.
