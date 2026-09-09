from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator, Dict, Any, List, Optional
from datetime import datetime, timezone
from backend.app.core.config import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    source_item_id TEXT NOT NULL,
    acquisition_datetime TEXT NOT NULL,
    cloud_cover REAL NOT NULL,
    bbox_json TEXT NOT NULL,
    crs TEXT NOT NULL,
    source_gsd_m REAL NOT NULL,
    local_path TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL,
    step TEXT NOT NULL,
    progress INTEGER NOT NULL,
    error_code TEXT,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(scene_id) REFERENCES scenes(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    checksum TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    scope TEXT NOT NULL,
    region_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    opensr_test_version TEXT NOT NULL,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);
"""

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(settings.database_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(SCHEMA_SQL)

# --- DB CRUD Helpers ---

def insert_scene(scene_data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scenes 
            (id, provider, source_item_id, acquisition_datetime, cloud_cover, bbox_json, crs, source_gsd_m, local_path, provenance_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scene_data["id"],
                scene_data["provider"],
                scene_data["source_item_id"],
                scene_data["acquisition_datetime"],
                scene_data["cloud_cover"],
                scene_data["bbox_json"],
                scene_data["crs"],
                scene_data["source_gsd_m"],
                scene_data["local_path"],
                scene_data["provenance_json"],
                scene_data.get("created_at", datetime.now(timezone.utc).isoformat())
            )
        )

def get_scene(scene_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def list_all_scenes() -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM scenes ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def insert_job(job_data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs 
            (id, scene_id, mode, model_name, status, step, progress, error_code, error_message, started_at, completed_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_data["id"],
                job_data["scene_id"],
                job_data["mode"],
                job_data["model_name"],
                job_data["status"],
                job_data["step"],
                job_data["progress"],
                job_data.get("error_code"),
                job_data.get("error_message"),
                job_data.get("started_at"),
                job_data.get("completed_at"),
                job_data.get("created_at", datetime.now(timezone.utc).isoformat())
            )
        )

def update_job_status(
    job_id: str,
    status: str,
    step: str,
    progress: int,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None
) -> None:
    with get_db() as conn:
        if started_at:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, step = ?, progress = ?, error_code = ?, error_message = ?, started_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, step, progress, error_code, error_message, started_at, completed_at, job_id)
            )
        else:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, step = ?, progress = ?, error_code = ?, error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, step, progress, error_code, error_message, completed_at, job_id)
            )

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def insert_artifact(artifact_data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts 
            (id, job_id, type, path, mime_type, checksum, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_data["id"],
                artifact_data["job_id"],
                artifact_data["type"],
                artifact_data["path"],
                artifact_data["mime_type"],
                artifact_data.get("checksum"),
                artifact_data.get("created_at", datetime.now(timezone.utc).isoformat())
            )
        )

def get_job_artifacts(job_id: str) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM artifacts WHERE job_id = ?", (job_id,))
        return [dict(row) for row in cursor.fetchall()]

def insert_metric(metric_data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO metrics 
            (id, job_id, metric_name, value, unit, scope, region_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric_data["id"],
                metric_data["job_id"],
                metric_data["metric_name"],
                metric_data["value"],
                metric_data["unit"],
                metric_data["scope"],
                metric_data.get("region_json"),
                metric_data.get("metadata_json"),
                metric_data.get("created_at", datetime.now(timezone.utc).isoformat())
            )
        )

def get_job_metrics(job_id: str) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM metrics WHERE job_id = ?", (job_id,))
        return [dict(row) for row in cursor.fetchall()]

def insert_benchmark_run(benchmark_data: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO benchmark_runs 
            (id, job_id, dataset_name, dataset_version, sample_id, opensr_test_version, results_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark_data["id"],
                benchmark_data["job_id"],
                benchmark_data["dataset_name"],
                benchmark_data["dataset_version"],
                benchmark_data["sample_id"],
                benchmark_data["opensr_test_version"],
                benchmark_data["results_json"],
                benchmark_data.get("created_at", datetime.now(timezone.utc).isoformat())
            )
        )

def get_job_benchmark(job_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM benchmark_runs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
