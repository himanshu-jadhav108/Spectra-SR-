# Trust Metrics and Gating

## A. Reconstruction consistency
Take SR output and degrade/resample it back to the original 10 m grid using the same declared degradation operator used for the evaluation.

Calculate:
- MAE
- RMSE
- correlation

Purpose: detect strong disagreement with the observed source. This is a consistency warning, not proof of hallucination.

## B. Spectral consistency
For each pixel/region where valid:

`SAD(x) = arccos( dot(a,b) / (||a|| ||b|| + eps) )`

Use radians internally and convert to degrees for display.

Also calculate:
- per-band drift
- NDVI drift on valid pixels

## C. Spatial alignment
Use phase correlation between an appropriate source-derived representation and the downsampled SR representation.
Store:
- dx
- dy
- magnitude
- registration status

## D. Local detail change features
Compare raw LR-upsample baseline against SR:
- gradient energy
- edge density
- local variance

Purpose: quantify where SR adds substantial new high-frequency content.

## E. Benchmark correctness
Only when HR exists:
- hallucination
- omission
- improvement
- synthesis
- reflectance consistency
- spectral consistency
- spatial consistency

Prefer the official `opensr-test` implementation where it matches the current installed version; if a metric is reimplemented, name the implementation explicitly.

## F. Trust scorecard
Do NOT collapse everything into one arbitrary number.
Show a profile:

```text
Spectral drift       GOOD
Spatial alignment    GOOD
Source consistency   GOOD
Predicted risk       LOW
Validation coverage  BENCHMARK / LIVE-PREDICTED
```

## Trust-gated output
Let:
- `base` = conservative upsampled source representation
- `raw_sr` = SR output
- `c` = confidence map in [0,1]

Prototype:
`safe_sr = base + c * (raw_sr - base)`

Optional smoothing:
- cap confidence changes over very small neighborhoods
- avoid halos at tile seams

Save both outputs; never overwrite raw SR with gated SR.

## UI wording
Use:
- “Predicted reliability”
- “AI-inferred detail”
- “High-risk reconstructed detail”
- “Benchmark-validated on reference scenes”

Avoid:
- “100% accurate”
- “hallucination-free”
- “ground truth” for SR output
