# Build Sequence for Antigravity

## Phase 0 — Repository reconnaissance
Before writing implementation code:
- inspect the current repository
- inspect Python and Node versions
- inspect GPU/CUDA availability
- inspect whether an existing app shell exists
- read current official SEN2SR and opensr-model documentation
- pin a known-working environment

Do not assume package APIs from memory.

## Phase 1 — Vertical slice
Deliver this first:
`local sample -> SEN2SRLite -> 2.5m artifact -> API -> UI`

Definition of done:
- backend starts
- model loads
- one sample runs
- output is saved
- frontend can render result

## Phase 2 — Data provider
Add CDSE search/download + local cache fallback.

## Phase 3 — Trust features
Implement deterministic metrics and risk map features.

## Phase 4 — Trust Head
Train from benchmark scenes; save model and schema.

## Phase 5 — Gating
Produce trust-gated GeoTIFF and side-by-side visual.

## Phase 6 — Benchmark
Integrate official OpenSR-test where practical.

## Phase 7 — Application
Add agricultural field-boundary edge comparison.

## Phase 8 — Polish
Add provenance, error states, loading UX, cached demo scenes, tests, and README.

## Stop rule
Do not proceed to the next phase if the previous phase does not have a demonstrable end-to-end artifact.
