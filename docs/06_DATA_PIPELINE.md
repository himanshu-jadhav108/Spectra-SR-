# Data Pipeline

## Scene discovery
Primary source: Copernicus Data Space Ecosystem.

Use the current STAC catalog and Sentinel-2 L2A collection:
- STAC root: `https://stac.dataspace.copernicus.eu/v1/`
- collection: `sentinel-2-l2a`

Search constraints:
- AOI intersection
- date interval
- maximum cloud cover
- limit to a small number of candidate scenes

## Download options
Preferred for the MVP:
- use the official Copernicus/Sentinel Hub APIs or a maintained Python client
- keep credentials only in `.env`
- download only B02/B03/B04/B08 + SCL when needed

## Local fallback
Directory:
```text
data/scenes/<scene_id>/
  B02.tif
  B03.tif
  B04.tif
  B08.tif
  SCL.tif
  scene.json
```

Demo mode can read this folder without network access.

## Preprocessing
1. open bands with rasterio
2. verify CRS / transform / bounds
3. align if needed using explicit resampling
4. scale digital values according to the source/model contract
5. mask invalid pixels
6. stack in `[B04,B03,B02,B08]` order
7. record shape, nodata fraction and metadata

## Large-scene tiling
Use LR-space windows around 128×128 with overlap.
Use an overlap/blending strategy consistent with the selected official model utility.

Do not implement a second tiler if the official SEN2SR utility already provides robust tiled large-image inference; wrap it.

## Georeferencing
Output should preserve:
- CRS
- geographic bounds
- acquisition/source metadata
- transform adjusted to the 2.5 m pixel size

If a utility in the OpenSR ecosystem already handles georeferenced large-image outputs, use it rather than maintaining duplicate stitching code.

## Scene provenance JSON
```json
{
  "provider": "copernicus_dataspace",
  "collection": "sentinel-2-l2a",
  "item_id": "...",
  "datetime": "...",
  "cloud_cover": 4.2,
  "bands": ["B04", "B03", "B02", "B08"],
  "source_gsd_m": 10,
  "target_gsd_m": 2.5
}
```

## Dataset rules
The live India path and paired benchmark path are separate datasets and must be shown separately in the UI and database.
