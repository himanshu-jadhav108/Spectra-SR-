from fastapi import FastAPI

app = FastAPI(title="Spectra SR API", version="0.1.0")

@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu": False,
        "model_loaded": False,
        "version": "0.1.0",
    }
