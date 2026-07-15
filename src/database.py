"""Step 7 of the pipeline: load events, queue depth, staff labels and the day's bills into one SQLite file, outputs/store.db.

The database is rebuilt from scratch on every run. Staff labels live in their own table and are joined in SQL
(the customer_events view), so the events themselves are never rewritten.
"""

import argparse
import csv
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from src.config import StoreConfig, load_config

POS_COLUMNS = ("order_id", "order_date", "order_time", "store_id", "total_amount")

SCHEMA = """
CREATE TABLE cameras (
    camera TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    clock_start TEXT NOT NULL,
    clock_end TEXT NOT NULL
);
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    seconds REAL NOT NULL,
    time TEXT NOT NULL,
    zone TEXT,
    dwell_seconds REAL,
    confidence REAL,
    censored INTEGER NOT NULL
);
CREATE TABLE queue_depth (
    camera TEXT NOT NULL,
    second INTEGER NOT NULL,
    time TEXT NOT NULL,
    depth INTEGER NOT NULL,
    PRIMARY KEY (camera, second)
);
CREATE TABLE track_labels (
    camera TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    staff_score REAL NOT NULL,
    is_staff INTEGER NOT NULL,
    method TEXT NOT NULL,
    PRIMARY KEY (camera, track_id)
);
CREATE TABLE pos_orders (
    order_id TEXT PRIMARY KEY,
    order_time TEXT NOT NULL,
    basket_value REAL NOT NULL,
    items INTEGER NOT NULL
);
CREATE INDEX events_by_type ON events (event_type, camera);
CREATE INDEX orders_by_time ON pos_orders (order_time);

-- Tracks with no staff label (no crops were taken) count as customers.
CREATE VIEW customer_events AS
SELECT e.*
FROM events AS e
LEFT JOIN track_labels AS l USING (camera, track_id)
WHERE COALESCE(l.is_staff, 0) = 0;
"""


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)


def insert_rows(connection: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    columns = list(rows[0])
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                           [tuple(row[c] for c in columns) for row in rows])
    return len(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open() as file:
        return list(csv.DictReader(file))


def clip_seconds(clip_path: Path) -> float:
    capture = cv2.VideoCapture(str(clip_path))
    seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return seconds


def camera_rows(config: StoreConfig) -> list[dict]:
    rows = []
    for camera in config.cameras:
        end = camera.clock_start + timedelta(seconds=clip_seconds(camera.clip_path))
        rows.append({"camera": camera.id, "role": camera.role,
                     "clock_start": camera.clock_start.isoformat(sep=" "),
                     "clock_end": end.isoformat(sep=" ", timespec="seconds")})
    return rows


def event_rows(path: Path) -> list[dict]:
    rows = read_csv(path)
    for row in rows:
        row["censored"] = int(row["censored"] == "True")
        row["dwell_seconds"] = float(row["dwell_seconds"]) if row["dwell_seconds"] else None
        row["zone"] = row["zone"] or None
    return rows


def pos_order_rows(path: Path, store_id: str) -> list[dict]:
    """The POS file has one row per line item; a bill's value is the sum of its lines, never the first row."""
    orders: dict[str, dict] = defaultdict(lambda: {"basket_value": 0.0, "items": 0, "order_time": None})
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        for row in csv.DictReader(file):
            line = {column: (row.get(column) or "").strip() for column in POS_COLUMNS}
            if line["store_id"] != store_id or not line["order_id"]:
                continue
            order = orders[line["order_id"]]
            order["basket_value"] += float(line["total_amount"] or 0)
            order["items"] += 1
            when = datetime.strptime(f"{line['order_date']} {line['order_time']}", "%d-%m-%Y %H:%M:%S")
            order["order_time"] = when.isoformat(sep=" ")
    return [{"order_id": order_id, "order_time": o["order_time"], "basket_value": round(o["basket_value"], 2),
             "items": o["items"]} for order_id, o in sorted(orders.items())]


def build(config: StoreConfig, db_path: Path) -> dict[str, int]:
    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        create_schema(connection)
        counts = {
            "cameras": insert_rows(connection, "cameras", camera_rows(config)),
            "events": insert_rows(connection, "events", event_rows(config.outputs_dir / "events.csv")),
            "queue_depth": insert_rows(connection, "queue_depth", read_csv(config.outputs_dir / "queue_depth.csv")),
            "track_labels": insert_rows(connection, "track_labels", read_csv(config.outputs_dir / "track_labels.csv")),
            "pos_orders": insert_rows(connection, "pos_orders", pos_order_rows(config.pos.csv, config.store_id)),
        }
        connection.commit()
    finally:
        connection.close()
    return counts


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Build outputs/store.db from the pipeline's CSV outputs.")
    parser.add_argument("--db", type=Path, default=config.outputs_dir / "store.db")
    args = parser.parse_args()
    counts = build(config, args.db)
    print(f"wrote {args.db}: " + ", ".join(f"{table} {n}" for table, n in counts.items()))


if __name__ == "__main__":
    main()
