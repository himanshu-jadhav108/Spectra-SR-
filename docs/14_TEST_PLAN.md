# Test Plan

## Unit tests
- SAD calculation
- reconstruction consistency
- phase correlation wrapper
- risk band thresholds
- trust gating equation
- GeoTIFF metadata preservation
- API schema validation

## Integration tests
1. cached scene → preprocessing
2. preprocessing → SR adapter
3. SR → trust engine
4. trust engine → safe SR
5. job runner → SQLite/artifacts
6. frontend → API → artifact rendering

## Scientific sanity tests
### Test A: identity-ish case
If SR is replaced with a nearest/bicubic baseline, risk features should not crash.

### Test B: deliberate perturbation
Inject a synthetic shift into SR before trust analysis. Spatial metric should increase.

### Test C: spectral perturbation
Multiply one output band by a small factor. Spectral drift should increase.

### Test D: confidence gating
Set confidence=0.0 in a region. Gated output should equal the base output there.
Set confidence=1.0. Gated output should equal raw SR there.

## Demo acceptance
- no broken images
- no infinite spinner
- no console errors that affect core flow
- live mode has cached fallback
- every displayed metric is traceable to stored JSON
