# Research and Truth Rules

## Rules every implementation and presentation must follow

1. Call the output a **super-resolved product**, not a newly observed 2.5 m satellite image.
2. Live India scenes have **predicted reliability**, not HR-ground-truth hallucination labels.
3. Hallucination/omission/improvement claims require aligned HR reference data.
4. Keep benchmark results tied to the exact dataset/sample/version.
5. Never invent metric values for slides.
6. Never hardcode scientific thresholds without labeling them as prototype defaults.
7. Preserve original input and raw SR artifacts separately from gated outputs.
8. Preserve georeferencing and provenance through every transformation.
9. Prefer official ESAOpenSR model and utility implementations over reimplementation.
10. Verify third-party licenses and model/data terms before public redistribution.

## Source-backed context
The SIH problem statement explicitly requires output finer than 4 m, geospatial and spectral consistency, validation against higher-resolution references, and uncertainty/error management.

The supplied research notes identify ESAOpenSR/SEN2SR, LDSR-S2, and OpenSR-test as the most relevant existing ecosystem and recommend making the trust/validation layer the team-authored contribution.

## What is team-authored design here
The following are Spectra SR prototype design choices, not claims that the source research has already implemented them:
- Trust Head trained on benchmark-derived features/targets.
- Trust-gated SR blending.
- Analyst-facing trust profile.
- Live-vs-benchmark evidence separation.
- Indian scene demo layer with domain-gap disclaimer.
