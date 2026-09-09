# Official Source Notes — verified 2026-09-08

These notes record the external references checked while preparing the Spectra SR MVP blueprint. Antigravity should still re-open them at implementation time because package APIs can change.

## ESA OpenSR
Organization: https://github.com/ESAOpenSR

The official organization describes OpenSR as an ESA-funded project for trustworthy, open-source Sentinel-2 super-resolution and lists repositories for models/weights, validation workflows, datasets, and inference utilities.

## SEN2SR
Repository: https://github.com/ESAOpenSR/SEN2SR

The current README documents:
- `SEN2SRLite` as the lightweight fast path
- super-resolution up to 2.5 m
- a 4× RGB+NIR non-reference model path for B04/B03/B02/B08
- large-image tiled inference utilities
- an installation path using `sen2sr` and `mlstac`
- the full Mamba route as a separate heavier path

Implementation instruction: use the official current README/import path and pin the environment after a successful local smoke test.

## LDSR-S2 / opensr-model
Repository: https://github.com/ESAOpenSR/opensr-model

The current README/demo documents:
- RGB-NIR Sentinel-2 input
- 10 m -> 2.5 m with factor 4
- tiled processing with 128 LR window size
- uncertainty-map generation through repeated variations
- georeferenced large-file processing via OpenSR utilities
- training not being supported by default in the repository because of time/resource constraints

For the MVP, this is an optional experimental backend. It must not block SEN2SRLite.

## OpenSR-test
Repository: https://github.com/ESAOpenSR/opensr-test

The repository documents paired datasets including NAIP, SPOT, Venµs, SPAIN CROPS and SPAIN URBAN, with Sentinel-2 data and higher-resolution reference imagery.

Use this for benchmark-backed correctness evaluation. Keep its measurements separate from live predicted-risk maps.

The current GitHub release page shows `v1.2.0` as the latest visible release at the time of this note.

## Copernicus Data Space
Current STAC root:
`https://stac.dataspace.copernicus.eu/v1/`

Current Sentinel-2 L2A collection:
`sentinel-2-l2a`

The current official documentation notes that the STAC catalog uses STAC 1.1.0 and that the legacy STAC endpoint was deprecated starting 17 Nov 2025.

Implementation instruction: do not use the deprecated legacy STAC endpoint.

## Scientific wording rule
The external sources support model and benchmark use; they do not imply that a live 10 m Sentinel-2 output is ground-truth 2.5 m imagery. Spectra SR therefore uses the terms `super-resolved product`, `predicted reliability`, and `high-risk reconstructed detail` for the live path.
