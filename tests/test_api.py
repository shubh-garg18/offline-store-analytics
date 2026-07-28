import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.database import create_schema, insert_rows


@pytest.fixture
def database(tmp_path, monkeypatch):
    path = tmp_path / "store.db"
    connection = sqlite3.connect(path)
    create_schema(connection)
    insert_rows(connection, "cameras", [
        {"camera": "entry", "role": "entry", "clock_start": "2026-04-10 20:00:00", "clock_end": "2026-04-10 20:02:00"},
        {"camera": "zone", "role": "zone", "clock_start": "2026-04-10 20:00:00", "clock_end": "2026-04-10 20:02:00"},
    ])
    insert_rows(connection, "events", [
        {"event_id": "a", "camera": "entry", "track_id": 1, "event_type": "ENTRY", "seconds": 5.0,
         "time": "2026-04-10 20:00:05.000", "zone": None, "dwell_seconds": None, "confidence": 0.9, "censored": 0},
        {"event_id": "b0", "camera": "zone", "track_id": 4, "event_type": "ZONE_ENTER", "seconds": 18.0,
         "time": "2026-04-10 20:00:18.000", "zone": "shelf", "dwell_seconds": None, "confidence": 0.9, "censored": 0},
        {"event_id": "b", "camera": "zone", "track_id": 4, "event_type": "ZONE_EXIT", "seconds": 30.0,
         "time": "2026-04-10 20:00:30.000", "zone": "shelf", "dwell_seconds": 12.0, "confidence": 0.9, "censored": 0},
    ])
    insert_rows(connection, "pos_orders", [
        {"order_id": "1", "order_time": "2026-04-10 18:30:00", "basket_value": 200.0, "items": 2},
        {"order_id": "2", "order_time": "2026-04-10 19:10:00", "basket_value": 600.0, "items": 1},
    ])
    connection.commit()
    connection.close()
    monkeypatch.setenv("STORE_DB", str(path))
    return path


client = TestClient(app)


def test_health_answers_even_without_a_database(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_DB", str(tmp_path / "missing.db"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database_ready"] is False


def test_data_endpoints_return_503_without_a_database(tmp_path, monkeypatch):
    monkeypatch.setenv("STORE_DB", str(tmp_path / "missing.db"))
    response = client.get("/metrics")
    assert response.status_code == 503
    assert "src.pipeline" in response.json()["detail"]


def test_metrics_match_the_database(database):
    body = client.get("/metrics").json()
    assert body["footfall"]["entries"] == 1
    assert body["video_window"]["start"] == "2026-04-10 20:00:00"
    assert body["conversion"]["conversion_rate"] == 0.0


def test_zones_queue_funnel_and_bills(database):
    assert client.get("/zones").json()[0]["avg_dwell_seconds"] == 12.0
    assert client.get("/queue").json()["max_depth"] == 0
    funnel = client.get("/funnel").json()
    assert [s["count"] for s in funnel["stages"]] == [1, 1, 0, 0]
    assert "stage counts" in funnel["caveat"]
    assert client.get("/bills").json()["bills"] == 2
