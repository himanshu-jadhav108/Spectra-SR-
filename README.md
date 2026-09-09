# Spectra SR — Antigravity MVP Build Package

## Purpose
This package is the implementation specification and prompt set for building the first internal hackathon prototype for SIH Problem Statement 26142: Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries.

Project name: **Spectra SR**

Prototype thesis:

> Reconstruct a 2.5 m Sentinel-2 RGB+NIR product from 10 m L2A imagery, then attach a trust layer that measures consistency and learned reliability and produces a trust-gated output rather than presenting every generated detail as ground truth.

## What is in this package
- `docs/`: complete MVP architecture and implementation blueprint
- `prompts/`: copy-paste prompts for Antigravity, split into phases
- `contracts/`: REST/OpenAPI and data contracts
- `config/`: environment/config expectations
- `scaffold/`: suggested application package layout and starter interfaces
- `tests/`: acceptance and validation checklist

## Important scientific boundary
Spectra SR must never claim that a live 10 m Sentinel-2 tile has ground-truth-confirmed 2.5 m detail. The live system exposes **predicted reliability / risk**, while true hallucination/omission/improvement measurements are reported only where an aligned HR reference exists.

## Current official references
- ESA OpenSR organization: https://github.com/ESAOpenSR
- SEN2SR: https://github.com/ESAOpenSR/SEN2SR
- OpenSR model / LDSR-S2: https://github.com/ESAOpenSR/opensr-model
- OpenSR test: https://github.com/ESAOpenSR/opensr-test
- Copernicus Data Space STAC: https://stac.dataspace.copernicus.eu/v1/
- Copernicus documentation: https://documentation.dataspace.copernicus.eu/

## MVP non-goals
Do not make these blockers:
1. Training an SR network from scratch.
2. Full 12/13-band super-resolution.
3. Building/road/damage detector training.
4. National-scale distributed deployment.
5. Formal calibrated probability claims unless calibration is actually tested.
6. Dependence on a live internet connection for demo-day success.
