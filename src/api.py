"""A small read-only API over outputs/store.db.

The heavy work (video, detection, staff classification) runs once, offline, in src/pipeline.py. The API only
reads the finished database through src/metrics.py, the same queries the report uses, so the two can never
disagree. Set STORE_DB to point it at another database file.
"""

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from src import metrics

app = FastAPI(
    title="CCTV store analytics",
    description="Footfall, shelf attention, billing queue, conversion and the day's bills for one store.",
)


def database_path() -> Path:
    return Path(os.environ.get("STORE_DB", "outputs/store.db"))


def conversion_window_minutes() -> int:
    return int(os.environ.get("CONVERSION_WINDOW_MINUTES", "5"))


def open_database() -> Iterator[sqlite3.Connection]:
    path = database_path()
    if not path.exists():
        raise HTTPException(status_code=503,
                            detail=f"no database at {path}: run `python -m src.pipeline` first, or set STORE_DB")
    connection = metrics.connect(path)
    try:
        yield connection
    finally:
        connection.close()


@app.get("/health")
def health() -> dict:
    """Answers even without a database, so a container check can tell "running" from "ready"."""
    path = database_path()
    return {"status": "ok", "database": str(path), "database_ready": path.exists()}


@app.get("/metrics")
def store_metrics(connection: sqlite3.Connection = Depends(open_database)) -> dict:
    return metrics.summary(connection, conversion_window_minutes())


@app.get("/zones")
def zones(connection: sqlite3.Connection = Depends(open_database)) -> list[dict]:
    return metrics.zone_attention(connection)


@app.get("/queue")
def billing_queue(connection: sqlite3.Connection = Depends(open_database)) -> dict:
    return metrics.queue(connection)


@app.get("/funnel")
def funnel(connection: sqlite3.Connection = Depends(open_database)) -> dict:
    return metrics.funnel(connection, conversion_window_minutes())


@app.get("/bills")
def bills(connection: sqlite3.Connection = Depends(open_database)) -> dict:
    return metrics.pos_daily(connection)
