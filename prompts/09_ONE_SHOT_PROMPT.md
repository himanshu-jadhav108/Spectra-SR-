# One-Shot Antigravity Prompt

Build the Spectra SR MVP from the attached specification package.

Read, in order:
1. `README.md`
2. `docs/00_PROJECT_OVERVIEW.md`
3. `docs/01_SCOPE_AND_PRIORITIES.md`
4. `docs/02_SYSTEM_ARCHITECTURE.md`
5. `docs/03_BACKEND_BLUEPRINT.md`
6. `docs/04_TRUST_HEAD_TRAINING.md`
7. `docs/05_TRUST_METRICS_AND_GATING.md`
8. `docs/06_DATA_PIPELINE.md`
9. `docs/07_STORAGE_AND_DATABASE.md`
10. `docs/08_API_CONTRACT.md`
11. `docs/09_FRONTEND_UI.md`
12. `docs/14_TEST_PLAN.md`
13. `contracts/openapi.yaml`
14. `config/.env.example`

Then inspect current official documentation for:
- ESAOpenSR/SEN2SR
- ESAOpenSR/opensr-model
- ESAOpenSR/opensr-test
- Copernicus Data Space STAC/API

Build in priority order:
P0 vertical slice → data → trust → trust head → gating → benchmark → agriculture → polish.

Never block the core demo on LDSR-S2.
Never train an SR model from scratch.
Never expose credentials.
Never claim that SR output is new sensor-observed ground truth.
Never label a live pixel hallucinated merely from LR/SR consistency.

At every stage keep the application runnable.
