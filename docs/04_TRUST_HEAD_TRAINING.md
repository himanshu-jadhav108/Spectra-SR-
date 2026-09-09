# Trust Head — Training Pipeline

## Goal
Build a small model that predicts **reliability risk of reconstructed detail**, not a magical ground-truth hallucination detector.

## Training source
Use paired LR/HR benchmark samples where available from OpenSR-test datasets.

## Training data generation
For each benchmark sample:

1. Load aligned LR and HR reference.
2. Run the exact same SR pipeline used in production.
3. Derive deterministic SR trust features.
4. Use HR to calculate correctness-oriented target statistics.
5. Convert target statistics into a local reliability target.
6. Train the Trust Head.

## Feature set v1
For each output patch / local region:

### Source/output consistency
- MAE between original LR and downsampled SR
- RMSE between original LR and downsampled SR
- correlation

### Spectral
- mean spectral angle distance
- per-band absolute drift
- NDVI drift where B04/B08 are valid

### Spatial
- phase-correlation shift magnitude
- edge shift score

### Texture / synthesis
- gradient energy delta
- local variance delta
- Laplacian energy delta
- edge density delta

### Input quality
- nodata fraction
- cloud/shadow fraction if SCL is available

## Target construction
Do not label each patch as “hallucinated” from consistency alone.
Instead, derive a reliability target from HR-based correctness metrics.

Prototype target options:
1. binary reliable/unreliable based on a chosen HR correctness threshold
2. 3-class LOW/MEDIUM/HIGH risk based on quantiles
3. continuous error target, then convert to risk bands for UI

For a hackathon MVP, use **3-class classification** because it is easy to explain.

## Model
Start with:
- `HistGradientBoostingClassifier`
- or `RandomForestClassifier`

No GPU is required for Trust Head training.

## Data split
Split by geographic scene, not random pixels.
This avoids leakage caused by adjacent patches from the same image appearing in train and test.

Recommended:
- 60% scenes train
- 20% scenes validation
- 20% scenes test

## Outputs
Save:
```text
models/trust_head.joblib
models/feature_schema.json
models/training_metadata.json
```

`training_metadata.json` must include:
- dataset identifiers
- feature names
- target construction rule
- train/val/test scene ids
- package versions
- model hyperparameters
- metrics

## Evaluation
Report:
- macro F1
- balanced accuracy
- confusion matrix
- calibration curve if probabilities are displayed

## Confidence transformation
Let Trust Head output `risk ∈ [0,1]` after calibration or monotonic scaling.
Then:
```text
confidence = 1 - risk
```

Map bands:
- confidence >= 0.75: LOW RISK / green
- 0.50–0.75: MEDIUM RISK / amber
- < 0.50: HIGH RISK / red

These thresholds are prototype defaults only; they should be configurable and should not be presented as scientifically universal thresholds.

## Important limitation
Benchmark-trained trust estimates may not transfer perfectly to Indian scenes. The live dashboard must label the map as **predicted reliability** and preserve a “validation coverage / domain gap” note.
