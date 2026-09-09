# Antigravity Prompt — Phase 1 Vertical Slice

Implement only the first vertical slice.

Goal:
`local cached Sentinel-2 RGB+NIR sample -> SEN2SRLite -> 2.5m artifact -> FastAPI -> simple frontend`

Tasks:
1. create Python environment
2. verify torch + CUDA
3. install the currently documented SEN2SRLite dependencies
4. download or register official model weights using the current documented method
5. load the model once at backend startup
6. add a `CachedSceneProvider`
7. add a `SREngine` adapter
8. create `/api/v1/jobs`
9. create `/api/v1/jobs/{id}`
10. process one sample asynchronously
11. save raw SR output and preview
12. render it in frontend
13. add a before/after slider

Do not add Trust Head yet.

Acceptance:
- GPU model load succeeds
- one scene completes
- output shape is exactly 4× in each spatial dimension relative to LR core band raster
- georeferencing is valid
- frontend renders result
