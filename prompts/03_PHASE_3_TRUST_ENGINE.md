# Antigravity Prompt — Phase 3 Trust Engine

Implement the deterministic Trust Engine.

Features:
1. reconstruction consistency
2. spectral angle/drift
3. spatial shift
4. gradient energy delta
5. edge density delta
6. local variance delta
7. nodata/cloud fraction
8. NDVI drift when valid

Implement tests for deliberate perturbations.

Output:
- per-pixel or per-region feature arrays
- aggregate metric JSON
- visual risk-feature overlays

Important:
The engine must not call any live feature “ground-truth hallucination”.
Use neutral names such as `predicted_risk` and `source_consistency_flag`.
