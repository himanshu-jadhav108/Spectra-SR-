# Antigravity Prompt — Phase 5 Trust-Gated SR

Implement the conservative trust-gating layer.

Inputs:
- base = conservative upsampled source representation
- raw_sr = production SR output
- confidence map from Trust Head

Compute:
`safe_sr = base + confidence * (raw_sr - base)`

Requirements:
- preserve raw SR unchanged
- generate safe SR separately
- smooth/confidence-process only when needed to avoid artifacts
- preserve GeoTIFF metadata
- expose both files in API

Add UI toggle:
`Raw SR` ↔ `Trust-Gated SR`

Add a short explainer in the UI:
“High-risk reconstructed detail is attenuated toward a conservative source-consistent representation.”
