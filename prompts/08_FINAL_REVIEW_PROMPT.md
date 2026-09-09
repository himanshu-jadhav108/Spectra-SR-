# Antigravity Prompt — Final Review / Refactor

Before declaring Spectra SR complete, perform a code and science review.

Check:
1. no deprecated Copernicus endpoint is used
2. official SR package/model path is current and documented
3. model weights are not committed if license/distribution forbids it
4. all outputs preserve georeferencing
5. raw SR and trust-gated SR are separate
6. live predicted risk is not called ground-truth hallucination
7. benchmark metrics are tied to actual HR reference data
8. no secrets are committed
9. API contracts match implementation
10. tests cover critical numerical functions
11. cached demo works offline
12. one-button path from scene to result works

Then produce `FINAL_REVIEW.md` with:
- verified components
- remaining risks
- exact commands to run demo
- exact model/data assumptions
