# Spectra SR — Antigravity Master Build Prompt

You are the principal engineer building the **Spectra SR** internal hackathon MVP.

## Context
Project: Spectra SR
Problem: SIH 2026 / PS 26142 — Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries.

The product must take Sentinel-2 L2A imagery at 10 m (RGB+NIR core bands), generate a 2.5 m super-resolved product, and provide a trust layer that measures source consistency, spectral/spatial behavior, predicted reliability, and — only where HR references exist — benchmark correctness.

## Your first action
Before implementing anything:
1. inspect the existing repository and preserve useful existing code
2. inspect current Python/Node versions
3. detect NVIDIA GPU/CUDA
4. read the current official ESAOpenSR repositories linked below
5. build a minimal environment that actually loads the official model
6. write a short `IMPLEMENTATION_NOTES.md` documenting the verified package/API versions

Official references:
- https://github.com/ESAOpenSR/SEN2SR
- https://github.com/ESAOpenSR/opensr-model
- https://github.com/ESAOpenSR/opensr-test
- https://stac.dataspace.copernicus.eu/v1/
- https://documentation.dataspace.copernicus.eu/

Do not invent API names or model import paths. Use the currently documented implementation.

## Product requirements
### P0
1. FastAPI backend
2. React/Next.js frontend
3. SQLite metadata DB + filesystem artifacts
4. Sentinel-2 L2A scene provider with local-cache fallback
5. SEN2SRLite RGB+NIR 4x inference
6. georeferenced 2.5 m output
7. trust feature extraction
8. lightweight Trust Head
9. trust-gated SR output
10. analysis dashboard
11. benchmark page / integration where practical
12. agricultural field-boundary edge comparison
13. provenance and downloads

## Architecture rule
Keep these layers separate:
- API/routes
- workflow/services
- ML adapter
- trust engine
- geospatial processing
- persistence
- UI

## Primary SR model
Use official `SEN2SRLite` RGB+NIR 4× path as the guaranteed MVP backend.

LDSR-S2 is an optional experimental backend and must never break the primary path.

## Core bands
Input order:
`B04, B03, B02, B08`

## Trust layer
Implement these deterministic features:
- reconstruction consistency
- spectral angle / drift
- spatial shift
- gradient / edge / variance deltas
- nodata/cloud fraction
- optional NDVI drift

Then implement a lightweight Trust Head trained on paired benchmark samples.

The Trust Head predicts **reliability risk**, not ground truth.

## Trust gating
Implement:
`safe_sr = base + confidence * (raw_sr - base)`

Keep raw SR and safe SR as separate artifacts.

## Critical scientific rule
Never call a live red region “hallucinated ground truth”.
Use “high predicted risk / AI-inferred detail”.
Only use hallucination/omission/improvement labels with aligned HR reference data.

## Data source
Primary:
- Copernicus Data Space Sentinel-2 L2A
- current STAC root: `https://stac.dataspace.copernicus.eu/v1/`

Fallback:
`storage/cache/scenes/`

The demo must work even with no network.

## Database
SQLite tables:
- scenes
- jobs
- artifacts
- metrics
- benchmark_runs

## API
Follow `contracts/openapi.yaml`.

## Frontend
Build these screens:
1. Home/Mission Console
2. Processing
3. Analysis Workspace
4. Benchmark Validation

Analysis Workspace must include:
- before/after slider
- Raw SR vs Trust-Gated toggle
- risk map
- trust scorecard
- provenance
- downloads

## Visual design
Scientific, professional, geospatial. No flashy “AI magic” UI.

## Performance
Target one manageable demo scene at a time.
Use GPU when available.
Use patch/tiled processing.
Do not build distributed infrastructure.

## Offline fallback
Ship at least one local demo scene and a reproducible cached result path.

## Tests
Create unit and integration tests based on `docs/14_TEST_PLAN.md`.

## Definition of done
A new developer can:
1. install the app
2. configure `.env`
3. run backend
4. run frontend
5. select a cached or live scene
6. produce 2.5 m SR
7. inspect trust map
8. inspect gated output
9. inspect stored metrics
10. download GeoTIFFs

## Deliverables
Produce:
- working source tree
- setup README
- `.env.example`
- model setup instructions
- test suite
- screenshots or seeded demo state if possible
- `IMPLEMENTATION_NOTES.md`
- no secret credentials
