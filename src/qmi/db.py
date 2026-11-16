"""Small, dependable SQLite persistence layer for experiment and inference history."""
from __future__ import annotations
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "project_quantum.sqlite3"


def database_path() -> Path:
    configured = os.getenv("QMI_DATABASE_PATH")
    path = Path(configured).expanduser() if configured else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(database_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            created_at TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            artifact_dir TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC);
        CREATE TABLE IF NOT EXISTS prediction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            case_id TEXT NOT NULL,
            slice_index INTEGER NOT NULL,
            resolution INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            metrics_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_prediction_created_at ON prediction_events(created_at DESC);
        """)


def _clean_json(value: Any):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json(value: Any) -> str:
    return json.dumps(_clean_json(value if value is not None else {}), ensure_ascii=False, allow_nan=False, default=str)


def add_experiment(experiment_name: str, dataset_name: str, *, status: str = "completed",
                   config: dict | None = None, metrics: dict | None = None,
                   artifact_dir: str | None = None) -> int:
    init_db()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO experiments(experiment_name,dataset_name,status,created_at,config_json,metrics_json,artifact_dir) VALUES(?,?,?,?,?,?,?)",
            (experiment_name, dataset_name, status, datetime.now(timezone.utc).isoformat(), _json(config), _json(metrics), artifact_dir),
        )
        return int(cur.lastrowid)


def add_prediction_event(case_id: str, slice_index: int, resolution: int, model_name: str,
                         metrics: dict | None = None) -> int:
    init_db()
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO prediction_events(created_at,case_id,slice_index,resolution,model_name,metrics_json) VALUES(?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), case_id, int(slice_index), int(resolution), model_name, _json(metrics)),
        )
        return int(cur.lastrowid)


def list_experiments(limit: int = 50) -> list[dict]:
    init_db()
    with connection() as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        result.append(item)
    return result


def list_predictions(limit: int = 50) -> list[dict]:
    init_db()
    with connection() as conn:
        rows = conn.execute("SELECT * FROM prediction_events ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json"))
        result.append(item)
    return result
