# React / Next.js UI Specification

## Design direction
Serious geospatial / scientific interface. Dark neutral base with restrained accent colors. Avoid “AI magic” visual language.

## Screen 1 — Home / Mission Console
Components:
- Spectra SR logo/title
- short tagline
- small explanation: “10 m Sentinel-2 → 2.5 m super-resolved product + predicted reliability”
- map or preset AOI cards
- date range
- cloud threshold
- data source toggle: `CDSE Live` / `Local Demo Cache`
- model toggle: `SEN2SRLite` / `LDSR-S2 (experimental)`
- Process button

## Screen 2 — Processing
Show:
- scene metadata
- stepper
- progress bar
- elapsed time
- GPU status
- cancel button if supported

## Screen 3 — Analysis Workspace
Layout:

```text
┌────────────────────────────────────────────────────────────┐
│ Scene metadata / provenance                                 │
├──────────────────────────────┬─────────────────────────────┤
│                              │ Trust Scorecard             │
│     main comparison          │ Spectral      GOOD          │
│   LR / SR / Safe-SR slider   │ Spatial       GOOD          │
│                              │ Consistency   GOOD          │
│                              │ Risk          LOW           │
├──────────────────────────────┴─────────────────────────────┤
│ [Raw SR] [Trust-Gated SR] [Risk Map] [Edges] [NDVI]       │
└────────────────────────────────────────────────────────────┘
```

## Comparison
Use a draggable before/after slider.
Modes:
- Original 10 m visualization
- Raw SR
- Trust-gated SR

## Risk map
Color is only a UI encoding. Always provide text labels:
- LOW RISK
- MEDIUM RISK
- HIGH RISK

Tooltip:
“Predicted reliability from Trust Head. Not ground-truth confirmation.”

## Metrics panel
### Live mode
Show:
- target GSD
- spectral drift
- spatial shift
- reconstruction error
- predicted risk
- input quality

### Benchmark mode
Add:
- hallucination
- omission
- improvement
- synthesis
- official benchmark metadata

## Application panel
MVP application = agriculture / field-boundary visibility.
Show edge overlay for Original vs Raw SR vs Trust-Gated SR.

## Provenance drawer
Show:
- source provider
- collection
- item id
- acquisition time
- cloud cover
- bands
- CRS
- input GSD
- output GSD
- model name/version
- trust-head version

## Download panel
Buttons:
- Download Raw SR GeoTIFF
- Download Trust-Gated GeoTIFF
- Download Risk Map
- Download Metrics JSON
- Download Scene Manifest

## Frontend state model
Use a small typed client layer and a single `AnalysisJob` state object.
Polling interval: 1–2 seconds while running.
Stop polling at terminal state.
