# Antigravity Prompt — Phase 4 Trust Head

Implement a lightweight Trust Head trained from paired benchmark data.

Workflow:
1. load an OpenSR-test paired sample
2. run the production SR adapter
3. derive Trust Engine features
4. compare SR with HR
5. construct a reliability target
6. split by scene id
7. train HistGradientBoostingClassifier or RandomForestClassifier
8. evaluate on held-out scenes
9. save model/schema/metadata
10. add inference service

Use 3 classes:
- LOW_RISK
- MEDIUM_RISK
- HIGH_RISK

Do not fabricate target labels from LR/SR consistency alone.
Target construction must use actual HR comparison for benchmark training data.

Persist:
- model file
- feature schema
- target rules
- dataset ids
- train/val/test scene ids
- metrics

Acceptance:
- model reload works
- test metrics are reproducible
- inference produces region risk score and map
