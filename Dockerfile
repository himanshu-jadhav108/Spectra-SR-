# =========================================================================
# Spectra SR — Production Dockerfile
# Multi-spectral Satellite Super-Resolution (10 m -> 2.5 m) & Trust Engine
# =========================================================================

FROM python:3.11-slim

# Prevent interactive prompts & optimize Python runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    APP_ENV=production \
    DATA_SOURCE=local \
    DEVICE=cpu

WORKDIR /app

# Install minimal OS dependencies for geospatial & image manipulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source tree
COPY . /app

# Ensure runtime storage directories exist with write permissions
RUN mkdir -p /app/storage/jobs /app/storage/cache/scenes /app/storage/db /app/storage/models

# Pre-initialize SQLite schema and offline demonstration caches
RUN python -c "from backend.app.db.database import init_db; from backend.app.services.seed_data import seed_cached_scenes; init_db(); seed_cached_scenes()"

# Expose container port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

# Launch production server dynamically binding to platform-assigned PORT
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
