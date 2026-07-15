import sqlite3

import pytest

from src import metrics
from src.database import create_schema, insert_rows

CAMERAS = [
    {"camera": "entry", "role": "entry", "clock_start": "2026-04-10 20:00:00", "clock_end": "2026-04-10 20:10:00"},
    {"camera": "zone", "role": "zone", "clock_start": "2026-04-10 20:00:00", "clock_end": "2026-04-10 20:10:00"},
    {"camera": "billing", "role": "billing", "clock_start": "2026-04-10 20:00:00", "clock_end": "2026-04-10 20:10:00"},
]


def event(camera, track_id, event_type, seconds, zone=None, dwell=None, censored=0):
    return {"event_id": f"{camera}-{track_id}-{seconds}-{event_type}", "camera": camera, "track_id": track_id,
            "event_type": event_type, "seconds": seconds,
            "time": f"2026-04-10 20:{int(seconds) // 60:02d}:{int(seconds) % 60:02d}.000",
            "zone": zone, "dwell_seconds": dwell, "confidence": 0.9, "censored": censored}


def label(camera, track_id, is_staff):
    return {"camera": camera, "track_id": track_id, "staff_score": float(is_staff), "is_staff": is_staff,
            "method": "test"}


def order(order_id, time, value):
    return {"order_id": order_id, "order_time": f"2026-04-10 {time}", "basket_value": value, "items": 1}


def build(tmp_path, events, labels, orders, depths=()):
    path = tmp_path / "store.db"
    connection = sqlite3.connect(path)
    create_schema(connection)
    insert_rows(connection, "cameras", CAMERAS)
    insert_rows(connection, "events", events)
    insert_rows(connection, "track_labels", labels)
    insert_rows(connection, "pos_orders", orders)
    insert_rows(connection, "queue_depth", [{"camera": "billing", "second": s, "time": "", "depth": d}
                                            for s, d in depths])
    connection.commit()
    connection.close()
    return metrics.connect(path)


def test_staff_are_excluded_from_footfall(tmp_path):
    connection = build(tmp_path, [event("entry", 1, "ENTRY", 10), event("entry", 2, "ENTRY", 20)],
                       [label("entry", 1, 0), label("entry", 2, 1)], [])
    result = metrics.footfall(connection)
    assert result["entries"] == 1
    assert result["entries_all_people"] == 2


def test_a_bill_within_the_window_converts_and_one_outside_does_not(tmp_path):
    events = [event("entry", 1, "ENTRY", 0), event("entry", 2, "ENTRY", 5),
              event("billing", 1, "QUEUE_JOIN", 60), event("billing", 2, "QUEUE_JOIN", 270)]
    # Visitor 1 joins the queue at 20:01:00, three minutes before bill A. Visitor 2 joins at 20:04:30, after
    # bill A and five and a half minutes before bill B, so outside both windows.
    orders = [order("A", "20:04:00", 500.0), order("B", "20:10:00", 300.0)]
    connection = build(tmp_path, events, [], orders)
    result = metrics.conversion(connection, window_minutes=5)
    assert result["converted_visitors"] == 1
    assert result["matched_bills"] == 1
    assert result["conversion_rate"] == 0.5


def test_conversion_is_undefined_not_zero_when_nobody_entered(tmp_path):
    connection = build(tmp_path, [], [], [order("A", "19:00:00", 100.0)])
    result = metrics.conversion(connection, window_minutes=5)
    assert result["conversion_rate"] is None
    assert result["bills_in_window"] == 0
    assert result["last_bill_before"] == "2026-04-10 19:00:00"


def test_average_dwell_ignores_visits_cut_off_by_the_clip(tmp_path):
    events = [event("zone", 1, "ZONE_EXIT", 30, "shelf", dwell=20.0),
              event("zone", 2, "ZONE_EXIT", 40, "shelf", dwell=10.0),
              event("zone", 3, "ZONE_EXIT", 50, "shelf", dwell=2.0, censored=1)]
    connection = build(tmp_path, events, [], [])
    shelf = metrics.zone_attention(connection)[0]
    assert shelf["avg_dwell_seconds"] == 15.0
    assert shelf["person_seconds"] == 32.0


def test_queue_depth_for_customers_skips_staff(tmp_path):
    events = [event("billing", 1, "QUEUE_LEAVE", 4, dwell=4.0), event("billing", 2, "QUEUE_LEAVE", 4, dwell=4.0)]
    connection = build(tmp_path, events, [label("billing", 2, 1)], [], depths=[(s, 2) for s in range(6)])
    result = metrics.queue(connection)
    assert result["max_depth_all_people"] == 2
    assert result["max_depth"] == 1


def test_peak_hours_top_bills_and_median_on_a_hand_made_day(tmp_path):
    orders = [order("A", "12:10:00", 100.0), order("B", "18:10:00", 200.0), order("C", "18:40:00", 300.0),
              order("D", "19:20:00", 400.0), order("E", "21:00:00", 1000.0)]
    result = metrics.pos_daily(build(tmp_path, [], [], orders))
    assert result["peak_two_hours"]["start_hour"] == 18
    assert result["peak_two_hours"]["bill_share"] == pytest.approx(0.6)
    assert result["peak_two_hours"]["revenue_share"] == pytest.approx(0.45)
    assert result["top_bills"] == 1
    assert result["top_20_percent_revenue_share"] == pytest.approx(0.5)
    assert result["median_basket"] == 300.0
    assert result["mean_basket"] == 400.0
