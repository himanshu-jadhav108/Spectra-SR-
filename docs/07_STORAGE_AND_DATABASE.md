# Storage and Database Design

## Storage strategy
MVP = SQLite + filesystem.

No PostgreSQL, S3, Redis, or object-store service is required for the first local prototype.

## Directory layout
```text
storage/
├── db/spectra_sr.sqlite3
├── jobs/<job_id>/
│   ├── input/
│   ├── intermediate/
│   ├── output/
│   ├── trust/
│   ├── metrics/
│   └── manifest.json
├── models/
│   └── trust_head.joblib
└── cache/scenes/<scene_id>/
```

## DB tables

### scenes
- id UUID/text PK
- provider
- source_item_id
- acquisition_datetime
- cloud_cover
- bbox_json
- crs
- source_gsd_m
- local_path
- provenance_json
- created_at

### jobs
- id UUID/text PK
- scene_id FK
- mode (`live`, `benchmark`)
- model_name
- status
- step
- progress
- error_code
- error_message
- started_at
- completed_at
- created_at

### artifacts
- id UUID/text PK
- job_id FK
- type (`lr`, `sr`, `safe_sr`, `risk_map`, `metrics_json`, `preview`, `benchmark_report`)
- path
- mime_type
- checksum
- created_at

### metrics
- id UUID/text PK
- job_id FK
- metric_name
- value
- unit
- scope (`live`, `benchmark`)
- region_json nullable
- metadata_json
- created_at

### benchmark_runs
- id UUID/text PK
- job_id FK
- dataset_name
- dataset_version
- sample_id
- opensr_test_version
- results_json
- created_at

## Why this is enough
It gives reproducibility and job provenance without operational complexity.

Phase B migration path:
- PostgreSQL for metadata
- object storage for rasters
- Redis/Celery or another queue for distributed inference
- separate model server
