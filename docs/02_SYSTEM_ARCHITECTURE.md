# System Architecture

```text
                         ┌──────────────────────┐
                         │ React / Next.js UI   │
                         └──────────┬───────────┘
                                    │ REST
                                    ▼
                         ┌──────────────────────┐
                         │ FastAPI API          │
                         │ Job orchestration    │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼───────────────────┐
                 ▼                  ▼                   ▼
         ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
         │ Scene/Data   │   │ SR Engine    │   │ Trust Engine │
         │ Provider     │   │ SEN2SRLite   │   │ + Trust Head │
         └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                │                  │                   │
                └──────────────────┼───────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Artifact Store       │
                         │ GeoTIFF/PNG/JSON     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ SQLite metadata DB   │
                         └──────────────────────┘
```

## Runtime rule
For the hackathon use a single-machine service. Do not introduce Celery/Redis/Postgres unless performance evidence requires them.

Use FastAPI background jobs with job state persisted to SQLite. Artifacts stay on local disk.

## Components
### DataProvider
Responsibilities:
- search CDSE/STAC for Sentinel-2 L2A
- enforce date / cloud / AOI constraints
- download the four core bands and optional SCL mask
- produce a normalized local scene bundle
- support local cached data source

### Preprocessor
Responsibilities:
- harmonize band order to `[B04,B03,B02,B08]`
- convert source integer reflectance to floating representation as required by the official model path
- apply nodata handling
- create windows for large scenes
- emit provenance metadata

### SREngine
Interface:
- `load()`
- `predict(tile)`
- `predict_large(scene)`
- `model_info()`

Only the adapter knows the exact official library/import calls. Pin the tested environment after smoke-testing rather than hardcoding an assumed future API.

### TrustEngine
Pipeline:
1. compute deterministic feature maps
2. reduce to tile/region features
3. run Trust Head
4. produce `risk_map.tif/png`
5. compute a confidence map
6. generate trust-gated SR

### BenchmarkEngine
Only used where HR reference is available.
- compare LR/SR/HR
- call OpenSR-test where practical
- store benchmark provenance and package version
- never mix benchmark measurements with live predicted-risk fields

### ArtifactService
- create job directory
- save original/preprocessed/SR/trust-gated/risk/metrics artifacts
- produce signed-ish local download paths (not public cloud links)

## Data flow
`scene_search -> scene_download -> preprocess -> sr -> trust_features -> trust_head -> gating -> artifacts -> UI`
