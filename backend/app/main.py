from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.errors import SpectraSRError
from backend.app.db.database import init_db
from backend.app.services.seed_data import seed_cached_scenes
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_scenes import router as scenes_router
from backend.app.api.routes_jobs import router as jobs_router
from backend.app.api.routes_artifacts import router as artifacts_router
from backend.app.api.routes_benchmark import router as benchmark_router
from backend.app.api.routes_demo import router as demo_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database and seed offline scenes
    init_db()
    seed_cached_scenes()
    yield
    # Shutdown

app = FastAPI(
    title="Spectra SR API",
    description="Evidence-Gated Super-Resolution for Trustworthy Satellite Analysis (Sentinel-2 10 m to 2.5 m)",
    version="0.2.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Error Handler
@app.exception_handler(SpectraSRError)
async def spectra_error_handler(request: Request, exc: SpectraSRError):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)

# Include Routers under /api/v1 prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(scenes_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(artifacts_router, prefix="/api/v1")
app.include_router(benchmark_router, prefix="/api/v1")
app.include_router(demo_router, prefix="/api/v1")

# Also include directly at root to satisfy contracts/openapi.yaml
app.include_router(health_router)
app.include_router(scenes_router)
app.include_router(jobs_router)
app.include_router(artifacts_router)
app.include_router(benchmark_router)
app.include_router(demo_router)

# Mount Brand Assets
brand_dir = settings.project_root / "frontend" / "public" / "brand"
if brand_dir.exists():
    app.mount("/brand", StaticFiles(directory=str(brand_dir)), name="brand")

# Mount Demo Assets
demo_dir = settings.project_root / "demo"
if demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_dir)), name="demo")
elif (settings.project_root / "frontend" / "public" / "demo").exists():
    app.mount("/demo", StaticFiles(directory=str(settings.project_root / "frontend" / "public" / "demo")), name="demo")

# Mount Public Assets if exists
public_dir = settings.project_root / "frontend" / "public"
if public_dir.exists():
    app.mount("/public", StaticFiles(directory=str(public_dir)), name="public")

# Mount Frontend Static/Dist if exists
frontend_dist = settings.project_root / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

frontend_dir = settings.project_root / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0"
}

@app.get("/style.css")
def get_style():
    css_path = settings.project_root / "frontend" / "style.css"
    return FileResponse(str(css_path), media_type="text/css", headers=NO_CACHE_HEADERS)

@app.get("/app.js")
def get_js():
    js_path = settings.project_root / "frontend" / "app.js"
    return FileResponse(str(js_path), media_type="application/javascript", headers=NO_CACHE_HEADERS)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon_path = settings.project_root / "frontend" / "public" / "brand" / "spectra-sr-logo.png"
    if icon_path.exists():
        return FileResponse(str(icon_path), media_type="image/png")
    return Response(status_code=204)

@app.get("/")
@app.get("/index.html")
def index():
    # If dist index.html exists, serve it; otherwise serve frontend/index.html
    dist_index = settings.project_root / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index), headers=NO_CACHE_HEADERS)
    dev_index = settings.project_root / "frontend" / "index.html"
    if dev_index.exists():
        return FileResponse(str(dev_index), headers=NO_CACHE_HEADERS)
    return {
        "status": "online",
        "app": "Spectra SR",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
