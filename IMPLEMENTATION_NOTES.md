# Spectra SR — Implementation Notes & Technical Specification

## 1. ESAOpenSR opensr-test Ecosystem Integration

### Package Specification
- **Package Name**: `opensr-test`
- **Installed Version**: `1.3.3`
- **Repository**: [https://github.com/ESAOpenSR/opensr-test](https://github.com/ESAOpenSR/opensr-test)
- **Dataset Registry**: [https://huggingface.co/datasets/isp-uv-es/opensr-test](https://huggingface.co/datasets/isp-uv-es/opensr-test)
- **Primary Citation**:
  > Aybar, C., et al. (2024). *OpenSR-Test: A Comprehensive Benchmark for Real-World Sentinel-2 Imagery Super-Resolution*. European Space Agency (ESA).

### Ecosystem Architecture & Evidence Domain Separation
Spectra SR strictly isolates two distinct operational domains:

1. **MISSION MODE (Operational Indian Sentinel-2 Domain)**:
   - **Data Source**: Real / cached Indian Sentinel-2 L2A surface reflectance tiles (e.g. Pune Peri-Urban, Punjab Wheat Corridors, Raichur Drylands, and user-ingested multi-spectral crops).
   - **Resolution Scaling**: 10 m observed GSD $\rightarrow$ 2.5 m super-resolved GSD ($4\times$ spatial factor).
   - **Bands Processed**: Sentinel-2 B02 (Blue), B03 (Green), B04 (Red), B08 (NIR).
   - **Trust Engine**: Real-time spectral consistency, edge preservation, and hallucination risk gating.
   - **Agronomic Output**: Field boundary sharpness gain calculation and smallholder parcel enhancement.

2. **VALIDATION LAB (Scientific Benchmark & Validation Domain)**:
   - **Data Source**: Official ESAOpenSR `opensr-test` paired LR/HR datasets:
     - `spain_crops`: Agricultural smallholders, heterogeneous field parcels.
     - `spain_urban`: High-density urban architecture, street networks.
     - `naip`: National Agriculture Imagery Program 0.6 m reference aerial imagery.
     - `spot`: CNES SPOT 1.5 m reference imagery.
     - `venus`: Venµs 5 m super-spectral microsatellite reference.
   - **Reference Target**: `HRharm` (Harmonized High-Resolution reference corrected relative to Sentinel-2 radiometry).
   - **Strict Constraint**: Benchmark imagery is never mixed with or labeled as Indian operational scenes. All validation cards display *"OpenSR benchmark reference"*.

---

## 2. Canonical Band Order & Input Preparation

For $\times 4$ RGB+NIR super-resolution and validation, the 12-band Sentinel-2 L2A array is filtered to the 4 essential 10-meter native bands:

| Band Name | Sentinel-2 Native Band | Wavelength | Canonical Index in Spectra SR |
|-----------|------------------------|------------|-------------------------------|
| **B04**   | Red                    | 665 nm     | Index 0                       |
| **B03**   | Green                  | 560 nm     | Index 1                       |
| **B02**   | Blue                   | 490 nm     | Index 2                       |
| **B08**   | NIR                    | 842 nm     | Index 3                       |

- **RGB Sub-tensor**: `tensor[[0, 1, 2], :, :]` corresponds directly to `[B04, B03, B02]` (True Color RGB), seamlessly conforming to `opensr_test.Metrics(rgb_bands=[0, 1, 2])`.
- **NIR Channel**: Index 3 (`B08`) is reserved for vegetation index calculation (NDVI) and trust feature extraction.

---

## 3. Official opensr-test Metrics Specification

When validating super-resolution models on paired benchmark datasets, `opensr_test.Metrics.compute(lr, sr, hr)` evaluates seven fundamental dimensions:

1. **Reflectance Error** (`reflectance`): Mean Absolute Error / L1 radiometric difference between SR and Harmonized HR ground truth.
2. **Spectral Consistency** (`spectral`): Spectral Angle Distance (SAD) evaluating preservation of spectral signatures across bands.
3. **Spatial Alignment** (`spatial`): Phase shift and keypoint displacement verifying sub-pixel geo-registration.
4. **Synthesis Score** (`synthesis`): High-frequency texture and boundary reconstruction accuracy.
5. **Hallucination Rate** (`ha_metric` / `hallucination`): Percentage of pixels exhibiting high-frequency detail inconsistent with ground truth HR structures.
6. **Omission Rate** (`om_metric` / `omission`): Percentage of real ground-truth high-frequency features missing in the super-resolved output.
7. **Improvement Rate** (`im_metric` / `improvement`): Percentage of pixels demonstrating genuine, verified spatial resolution gains over the 10 m bicubic baseline.

---

## 4. Trust Head Calibration Architecture

The Trust Head is trained using benchmark LR/HR pairs from `opensr-test`:
1. **Feature Extraction**: 8-dimensional trust feature vectors (gradient magnitude, spectral angle, local variance, cycle consistency error, spatial phase shift, band contrast ratio, high-frequency residual, radiometric drift).
2. **Target Construction**: Ground-truth reliability target $Y_{risk} \in [0, 1]$ calculated directly from pixel-level spatial error against `HRharm`.
3. **Grouping Strategy**: Data is partitioned **strictly by scene / ROI** (never by random individual pixels) to eliminate spatial auto-correlation leakage across training and test splits.
4. **Model Artifacts**:
   - `models/trust_head.joblib`: Trained scikit-learn ensemble estimator.
   - `models/trust_head_schema.json`: Input feature definitions, types, and normalization bounds.
   - `models/trust_head_metadata.json`: Dataset split, validation ROC-AUC, RMSE, and training timestamp.
5. **Deterministic Fallback**: If `models/trust_head.joblib` is absent, `TrustHead` defaults to an analytical physics-based consistency engine clearly tagged as `"Prototype predicted reliability"`.

---

## 5. Validation Lab UI Architecture & Interactive Capabilities

### User Interface Layout
The Spectra SR interface features a dedicated, visually segregated **Validation Lab** (4th navigation tab: `🔬 Validation Lab [ESAOpenSR]`):
- **Domain Isolation Banner**: A purple-accented scientific header clearly states that all displayed imagery is reference data from the ESAOpenSR ecosystem, isolating it from Indian Mission Mode.
- **Dataset & Sample Selector**: Interactive chips allow instant switching between all 5 ingested benchmark datasets (`spain_crops`, `spain_urban`, `naip`, `spot`, `venus`) and individual chip ROIs (`sample_000` through `sample_004`).
- **Synchronized 3-Way Visualizer**: Side-by-side presentation of:
  1. *Sentinel-2 10 m Low-Resolution Input*
  2. *Spectra SR 2.5 m Super-Resolved Output*
  3. *Aligned Harmonized High-Resolution Reference (PNOA / NAIP / SPOT / Venµs)*
- **Interactive Dual-View Split Slider**: Toggleable swipe divider comparing Spectra SR output directly against ground-truth HR imagery at 1:1 pixel scale.
- **ESAOpenSR Protocol Metrics Dashboard**:
  - Top metric cards with dynamic status pills: Hallucination Rate (`< 3%`), Omission Rate (`< 5%`), Quantitative Improvement Gain (`> 20%`), Structural Synthesis (`> 0.85`).
  - Full comparative fidelity table including Reflectance RMSE, Spectral Angle Distance (°), Spatial Edge Correlation, and verified GSD.
- **Trust Head Calibration & Feature Importance Panel**:
  - Displays multi-class ROC-AUC (`0.744`), Weighted F1 (`0.640`), and scene-split test accuracy (`62.7%`).
  - Dynamic visual progress bars for all 8 Trust Head features ranked by predictive importance (Spectral Angle, Reflectance RMSE, Phase Shift, MAE, NDVI Drift, Variance, Laplacian, Gradient).
  - Chip concordance indicator comparing predicted risk score against actual ground-truth radiometric error.
- **Official Scientific Attribution & BibTeX Citation**:
  - Full metadata attribution to ESA and Universitat de València (Aybar et al., 2024), CC BY 4.0 license tag, and copyable BibTeX snippet.

