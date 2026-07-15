"""Store metrics as SQL queries over outputs/store.db, shared by the report and the API.

Standard library only (sqlite3), so the API container needs no computer-vision packages. Every function takes
an open connection and returns plain dicts and lists, ready to print or to send as JSON. "Customers" means
tracks the staff classifier did not mark as staff; "all people" includes staff.
"""

import sqlite3
from pathlib import Path

FUNNEL_CAVEAT = (
    "Each stage is counted on a different camera, and each camera numbers people on its own, so one shopper "
    "can be a different ID at each stage. These are stage counts, not one person followed through the store."
)


def connect(path: Path) -> sqlite3.Connection:
    """Read-only, so neither the report nor the API can change the data."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def one(connection: sqlite3.Connection, sql: str, parameters: tuple = ()) -> dict:
    row = connection.execute(sql, parameters).fetchone()
    return dict(row) if row else {}


def every(connection: sqlite3.Connection, sql: str, parameters: tuple = ()) -> list[dict]:
    return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def video_window(connection: sqlite3.Connection) -> dict:
    return one(connection, "SELECT MIN(clock_start) AS start, MAX(clock_end) AS end FROM cameras")


def footfall(connection: sqlite3.Connection) -> dict:
    return one(connection, """
        SELECT COALESCE(SUM(e.event_type = 'ENTRY' AND COALESCE(l.is_staff, 0) = 0), 0) AS entries,
               COALESCE(SUM(e.event_type = 'EXIT' AND COALESCE(l.is_staff, 0) = 0), 0) AS exits,
               COALESCE(SUM(e.event_type = 'ENTRY'), 0) AS entries_all_people,
               COALESCE(SUM(e.event_type = 'EXIT'), 0) AS exits_all_people
        FROM events AS e
        LEFT JOIN track_labels AS l USING (camera, track_id)
        WHERE e.camera IN (SELECT camera FROM cameras WHERE role = 'entry')
    """)


def zone_attention(connection: sqlite3.Connection) -> list[dict]:
    """One row per zone. Person-seconds add up every visit, including visits cut off by the clip edges
    (a lower bound); average dwell uses only complete visits, because cut-off ones would drag it down."""
    return every(connection, """
        WITH visits AS (
            SELECT e.camera, e.zone, e.track_id, e.dwell_seconds, e.censored, COALESCE(l.is_staff, 0) AS is_staff
            FROM events AS e
            LEFT JOIN track_labels AS l USING (camera, track_id)
            WHERE e.event_type = 'ZONE_EXIT'
        )
        SELECT camera,
               zone,
               COUNT(DISTINCT track_id) AS people,
               COUNT(DISTINCT CASE WHEN is_staff = 0 THEN track_id END) AS customers,
               ROUND(SUM(dwell_seconds), 1) AS person_seconds,
               ROUND(COALESCE(SUM(CASE WHEN is_staff = 0 THEN dwell_seconds END), 0), 1) AS customer_person_seconds,
               SUM(censored = 0) AS complete_visits,
               ROUND(AVG(CASE WHEN censored = 0 THEN dwell_seconds END), 1) AS avg_dwell_seconds,
               ROUND(AVG(CASE WHEN censored = 0 AND is_staff = 0 THEN dwell_seconds END), 1) AS customer_avg_dwell_seconds
        FROM visits
        GROUP BY camera, zone
        ORDER BY camera, person_seconds DESC
    """)


def queue(connection: sqlite3.Connection) -> dict:
    """Queue depth each second, for all people (stored) and for customers (rebuilt from customer queue visits)."""
    return one(connection, """
        WITH customer_visits AS (
            SELECT camera, seconds - dwell_seconds AS start_second, seconds AS end_second
            FROM customer_events
            WHERE event_type = 'QUEUE_LEAVE'
        ),
        per_second AS (
            SELECT q.depth AS all_people,
                   (SELECT COUNT(*) FROM customer_visits AS v
                    WHERE v.camera = q.camera AND q.second BETWEEN v.start_second AND v.end_second) AS customers
            FROM queue_depth AS q
        )
        SELECT COUNT(*) AS seconds_observed,
               COALESCE(MAX(customers), 0) AS max_depth,
               ROUND(COALESCE(AVG(customers), 0), 2) AS avg_depth,
               COALESCE(SUM(customers >= 3), 0) AS seconds_at_3_or_more,
               COALESCE(MAX(all_people), 0) AS max_depth_all_people,
               ROUND(COALESCE(AVG(all_people), 0), 2) AS avg_depth_all_people,
               (SELECT COUNT(DISTINCT camera || ':' || track_id) FROM customer_events
                WHERE event_type = 'QUEUE_JOIN') AS customer_queue_joins,
               (SELECT COUNT(DISTINCT camera || ':' || track_id) FROM events
                WHERE event_type = 'QUEUE_JOIN') AS queue_joins_all_people
        FROM per_second
    """)


def conversion(connection: sqlite3.Connection, window_minutes: int) -> dict:
    """A customer converts if they joined the billing queue in the `window_minutes` before a bill."""
    window = f"-{int(window_minutes)} minutes"
    matched = one(connection, """
        SELECT COUNT(DISTINCT j.camera || ':' || j.track_id) AS converted_visitors,
               COUNT(DISTINCT o.order_id) AS matched_bills
        FROM customer_events AS j
        JOIN pos_orders AS o
          ON datetime(j.time) BETWEEN datetime(o.order_time, ?) AND datetime(o.order_time)
        WHERE j.event_type = 'QUEUE_JOIN'
    """, (window,))
    entries = footfall(connection)["entries"]
    joiners = queue(connection)["customer_queue_joins"]
    nearby = one(connection, """
        WITH w AS (SELECT MIN(clock_start) AS start, MAX(clock_end) AS end FROM cameras)
        SELECT (SELECT COUNT(*) FROM pos_orders, w
                WHERE datetime(order_time) BETWEEN datetime(w.start) AND datetime(w.end, ?)) AS bills_in_window,
               (SELECT MAX(order_time) FROM pos_orders, w WHERE order_time < w.start) AS last_bill_before,
               (SELECT MIN(order_time) FROM pos_orders, w WHERE order_time > w.end) AS first_bill_after
    """, (f"+{int(window_minutes)} minutes",))
    converted = matched["converted_visitors"]
    return {
        "window_minutes": window_minutes,
        "entries": entries,
        "customer_queue_joins": joiners,
        **matched,
        "conversion_rate": round(converted / entries, 4) if entries else None,
        "queue_conversion_rate": round(converted / joiners, 4) if joiners else None,
        **nearby,
    }


def funnel(connection: sqlite3.Connection, window_minutes: int) -> dict:
    zone_visitors = one(connection, """
        SELECT COUNT(DISTINCT camera || ':' || track_id) AS n FROM customer_events WHERE event_type = 'ZONE_ENTER'
    """)["n"]
    sale = conversion(connection, window_minutes)
    stages = [
        {"stage": "entered through the door", "count": sale["entries"]},
        {"stage": "visited a shelf zone", "count": zone_visitors},
        {"stage": "joined the billing queue", "count": sale["customer_queue_joins"]},
        {"stage": "matched to a bill", "count": sale["converted_visitors"]},
    ]
    return {"stages": stages, "caveat": FUNNEL_CAVEAT}


def pos_daily(connection: sqlite3.Connection) -> dict:
    """The whole day's bills: by hour, the busiest two hours, how concentrated revenue is, and basket size."""
    hourly = every(connection, """
        SELECT CAST(strftime('%H', order_time) AS INTEGER) AS hour,
               COUNT(*) AS bills,
               ROUND(SUM(basket_value), 2) AS revenue
        FROM pos_orders
        GROUP BY hour
        ORDER BY hour
    """)
    totals = one(connection, "SELECT COUNT(*) AS bills, ROUND(SUM(basket_value), 2) AS revenue FROM pos_orders")
    if not totals["bills"]:
        return {"bills": 0, "revenue": 0.0, "hourly": []}
    peak = one(connection, """
        WITH hourly AS (
            SELECT CAST(strftime('%H', order_time) AS INTEGER) AS hour, COUNT(*) AS bills, SUM(basket_value) AS revenue
            FROM pos_orders
            GROUP BY hour
        )
        SELECT h.hour AS start_hour,
               h.bills + COALESCE(n.bills, 0) AS bills,
               ROUND(h.revenue + COALESCE(n.revenue, 0), 2) AS revenue
        FROM hourly AS h
        LEFT JOIN hourly AS n ON n.hour = h.hour + 1
        ORDER BY bills DESC, revenue DESC
        LIMIT 1
    """)
    top = one(connection, """
        WITH ranked AS (
            SELECT basket_value,
                   ROW_NUMBER() OVER (ORDER BY basket_value DESC) AS rank,
                   COUNT(*) OVER () AS n
            FROM pos_orders
        )
        SELECT CAST(ROUND(0.2 * MAX(n)) AS INTEGER) AS top_bills,
               ROUND(SUM(CASE WHEN rank <= ROUND(0.2 * n) THEN basket_value ELSE 0 END) / SUM(basket_value), 4)
                   AS top_20_percent_revenue_share
        FROM ranked
    """)
    basket = one(connection, """
        WITH ordered AS (
            SELECT basket_value, ROW_NUMBER() OVER (ORDER BY basket_value) AS rank, COUNT(*) OVER () AS n
            FROM pos_orders
        )
        SELECT ROUND((SELECT AVG(basket_value) FROM pos_orders), 2) AS mean_basket,
               ROUND(AVG(basket_value), 2) AS median_basket
        FROM ordered
        WHERE rank IN ((n + 1) / 2, (n + 2) / 2)
    """)
    return {
        **totals,
        "hourly": hourly,
        "peak_two_hours": {**peak, "end_hour": peak["start_hour"] + 2,
                           "bill_share": round(peak["bills"] / totals["bills"], 4),
                           "revenue_share": round(peak["revenue"] / totals["revenue"], 4)},
        **top,
        **basket,
        "mean_to_median": round(basket["mean_basket"] / basket["median_basket"], 2),
    }


def summary(connection: sqlite3.Connection, window_minutes: int) -> dict:
    return {
        "video_window": video_window(connection),
        "footfall": footfall(connection),
        "queue": queue(connection),
        "conversion": conversion(connection, window_minutes),
    }
