# Backend Blueprint

## Python package structure

```text
backend/app/
├── main.py
├── api/
│   ├── __init__.py
│   ├── routes_health.py
│   ├── routes_scenes.py
│   ├── routes_jobs.py
│   ├── routes_artifacts.py
│   └── routes_benchmark.py
├── core/
│   ├── config.py
│   ├── logging.py
│   └── errors.py
├── db/
│   ├── database.py
│   └── migrations.py
├── models/
│   ├── job.py
│   ├── scene.py
│   ├── artifact.py
│   ├── metric.py
│   └── benchmark.py
├── schemas/
│   ├── scene.py
│   ├── job.py
│   ├── result.py
│   └── benchmark.py
├── services/
│   ├── scene_provider.py
│   ├── preprocessing.py
│   ├── sr_engine.py
│   ├── job_runner.py
│   ├── artifact_service.py
│   └── provenance.py
├── trust/
│   ├── features.py
│   ├── spectral.py
│   ├── spatial.py
│   ├── consistency.py
│   ├── trust_head.py
│   ├── gating.py
│   └── calibration.py
└── geo/
    ├── raster.py
    ├── tiling.py
    └── crs.py
```

## Service boundaries
Do not let the API layer contain raster or ML logic.

`routes_*` should validate inputs, start jobs, and return schemas.
`services/*` owns workflows.
`trust/*` owns metrics and gating.
`geo/*` owns raster/CRS operations.

## Job lifecycle
```text
QUEUED -> RUNNING -> COMPLETED
                  -> FAILED
```

Persist:
- timestamps
- step currently running
- progress 0–100
- error code/message
- model name/version
- source item id

## Suggested workflow steps
1. DISCOVER_SCENE
2. DOWNLOAD
3. PREPROCESS
4. SR_INFERENCE
5. TRUST_ANALYSIS
6. TRUST_GATING
7. ARTIFACT_WRITE
8. COMPLETE

## Error handling
Return structured errors:
```json
{
  "code": "MODEL_INIT_FAILED",
  "message": "SEN2SRLite failed to initialize",
  "recoverable": true,
  "suggested_action": "Use cached scene or CPU fallback"
}
```

Never return raw Python tracebacks to the browser.

## Logging
Use JSON logs with `job_id`, `scene_id`, `step`, `elapsed_ms`.
Do not log access tokens or secrets.
