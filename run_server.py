"""
SPECTRA-SR V2 — Application & Demo Server Launcher
Starts the FastAPI backend and serves the frontend application.
"""
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    
    print("=" * 65)
    print(" [SPECTRA-SR V2] VIDEO DEMO & PRODUCTION SERVER")
    print("=" * 65)
    print(f" -> Local Web UI:    http://{host}:{port}/")
    print(f" -> Demo Mode:       http://{host}:{port}/ (Active by default)")
    print(f" -> Demo API Route:  http://{host}:{port}/api/v1/demo/scene")
    print(f" -> API Health:      http://{host}:{port}/api/v1/health")
    print("=" * 65)
    print(" Press Ctrl+C to stop the server.\n")
    
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
