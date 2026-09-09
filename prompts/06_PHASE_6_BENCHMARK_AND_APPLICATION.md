# Antigravity Prompt — Phase 6 Benchmark + Agriculture Application

Add benchmark mode and one lightweight downstream utility demo.

Benchmark mode:
- use official OpenSR-test integration where practical
- store dataset/version/sample id
- report benchmark correctness metrics only for HR-backed samples
- show LR vs SR vs HR

Agriculture mode:
- compute simple edge/boundary response on original, raw SR, and trust-gated SR
- visualize field-boundary clarity
- do not train a building/road/segmentation model

Acceptance:
- benchmark screen renders real computed values
- agriculture screen shows a before/after edge comparison
- benchmark and live risk metrics are visually separated
