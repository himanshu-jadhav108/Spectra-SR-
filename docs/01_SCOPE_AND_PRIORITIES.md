# MVP Scope and Priorities

## P0 — absolutely required
### P0.1 Sentinel-2 source
- Source: Sentinel-2 L2A
- Core bands: B04 Red, B03 Green, B02 Blue, B08 NIR
- Native spatial resolution: 10 m
- Source provenance stored with every job
- Cloud-aware scene selection or cached fallback

### P0.2 SR
- Primary engine: `SEN2SRLite` RGB+NIR 4× path
- Output target: 2.5 m
- Use official model weights / package path
- Large-image processing through tiled windows
- Preserve geographic extent and CRS

### P0.3 Trust layer
Implement deterministic features first:
- LR→SR→LR reconstruction consistency
- Spectral-angle or equivalent spectral drift metric
- Spatial alignment / shift check
- Local high-frequency change / edge-density features
- Input quality indicators (cloud/nodata ratio)

Then a lightweight Trust Head:
- input = feature vector / small local feature tensor
- output = `risk_score` in [0,1]
- output classes = LOW / MEDIUM / HIGH risk

### P0.4 Trust-gated SR
Use the learned confidence/risk map to attenuate high-frequency enhancement:

`safe_sr = base + confidence * (raw_sr - base)`

The exact base and confidence transform are tunable; the principle is conservative fallback toward source-consistent content.

### P0.5 Dashboard
- AOI / scene selection
- processing status
- before/after comparison
- trust-map toggle
- raw vs trust-gated output toggle
- metric cards
- provenance panel
- download buttons

## P1 — should be done if stable
- OpenSR-test benchmark page
- Edge/boundary utility comparison for agriculture
- cached offline demo scenes
- one-click benchmark run

## P2 — not a blocker
- LDSR-S2 research mode
- self-supervised SR fine-tuning
- 20 m band fusion
- richer downstream tasks
- cloud deployment
