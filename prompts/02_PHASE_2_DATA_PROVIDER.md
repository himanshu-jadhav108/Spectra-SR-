# Antigravity Prompt — Phase 2 Data Provider

Add the live Copernicus Data Space provider while preserving local fallback.

Implement:
- scene search
- cloud filter
- AOI search
- date window
- B02/B03/B04/B08 retrieval
- optional SCL retrieval
- local cache
- scene provenance JSON

Use the current official Copernicus documentation and current APIs.
Do not hardcode deprecated STAC endpoints.

The current official STAC root is:
`https://stac.dataspace.copernicus.eu/v1/`

Acceptance:
- live search returns candidate Sentinel-2 L2A scene(s)
- one scene can be downloaded
- local fallback still works with network disabled
- provenance is persisted
