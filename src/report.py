"""Step 8 of the pipeline: write outputs/report.md, the store report a manager would read, plus how far to trust it.

Numbers come from outputs/store.db through src/metrics.py (the same functions the API serves) and from the
evaluation steps. The report is private: it shows rupee amounts, which the public README never does.
"""

import argparse
from pathlib import Path

from src import metrics
from src.config import StoreConfig, load_config
from src.evaluate import GROUND_TRUTH as ENTRY_TRUTH
from src.evaluate import ablations
from src.evaluate import read_ground_truth as read_entry_truth
from src.evaluate_zones import GROUND_TRUTH as ZONE_TRUTH
from src.evaluate_zones import ZONES, read_ground_truth, sampling_step, score_zone, zone_intervals
from src.track_table import read_tracks_csv


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |" for row in rows]
    return lines


def lines_containing(path: Path, marker: str) -> list[str]:
    return [line for line in path.read_text().splitlines() if marker in line] if path.exists() else []


def section_table(path: Path, heading: str) -> list[str]:
    """The markdown table that follows `heading` in another report file."""
    if not path.exists():
        return [f"({path} not found)"]
    lines = path.read_text().splitlines()
    start = lines.index(heading) + 1
    found = []
    for line in lines[start:]:
        if line.startswith("|"):
            found.append(line)
        elif found:
            break
    return found


def footfall_section(connection) -> list[str]:
    f = metrics.footfall(connection)
    return [
        "## 1. Footfall (entry camera)",
        "",
        f"Entries {f['entries']}, exits {f['exits']} (customers); {f['entries_all_people']} and "
        f"{f['exits_all_people']} counting staff.",
        "",
        "Nobody walked through the door in this clip. About 20 to 25 people passed along the street outside and one",
        "man stood at the door on his phone without entering; none of them counts (section 7).",
        "",
    ]


def zone_section(connection) -> list[str]:
    rows = [[z["camera"], z["zone"], z["people"], z["person_seconds"], z["complete_visits"], z["avg_dwell_seconds"],
             z["customers"], z["customer_person_seconds"]] for z in metrics.zone_attention(connection)]
    return [
        "## 2. Zone attention",
        "",
        "Person-seconds add up every visit, including visits cut off by the start or end of the clip, so they are a",
        "lower bound. Average dwell uses only complete visits. Each camera numbers people on its own.",
        "",
        *table(["Camera", "Zone", "People", "Person-seconds", "Complete visits", "Avg dwell (s)", "Customers",
                "Customer person-seconds"], rows),
        "",
        "The customer columns are a lower bound: the staff classifier keeps only some in-store customers (section 7),",
        "so most attention here is attributed to staff, who restock and serve at these same shelves.",
        "",
    ]


def queue_section(connection) -> list[str]:
    q = metrics.queue(connection)
    return [
        "## 3. Billing queue",
        "",
        f"Over {q['seconds_observed']} seconds: customer queue depth max {q['max_depth']}, average {q['avg_depth']}, "
        f"{q['seconds_at_3_or_more']} seconds at 3 or more. Counting staff: max {q['max_depth_all_people']}, "
        f"average {q['avg_depth_all_people']}, from {q['queue_joins_all_people']} people in the queue area.",
        "",
        "Everyone seen in the queue area is labelled staff (the cashier and colleagues working at the counter), so",
        "no customer queued during the clip.",
        "",
    ]


def conversion_section(connection, window_minutes: int) -> list[str]:
    c = metrics.conversion(connection, window_minutes)
    window = metrics.video_window(connection)
    rate = "undefined (no entries)" if c["conversion_rate"] is None else percent(c["conversion_rate"])
    return [
        "## 4. Conversion",
        "",
        f"Rule: a customer converts if they joined the billing queue in the {window_minutes} minutes before a bill.",
        f"Converted visitors {c['converted_visitors']}, matched bills {c['matched_bills']}, entries {c['entries']}: "
        f"conversion {rate}.",
        "",
        f"This is a real zero, not a failure. The video covers {window['start'][11:]} to {window['end'][11:]}; the",
        f"bills nearest to it are at {c['last_bill_before'][11:]} and {c['first_bill_after'][11:]}, and "
        f"{c['bills_in_window']} bills fall inside the video window or within {window_minutes} minutes after it.",
        "Measuring conversion needs footage that overlaps the day's bills, ideally the whole trading day.",
        "",
    ]


def funnel_section(connection, window_minutes: int) -> list[str]:
    f = metrics.funnel(connection, window_minutes)
    return ["## 5. Funnel", "", *table(["Stage", "Customers"], [[s["stage"], s["count"]] for s in f["stages"]]), "",
            f["caveat"], ""]


def pos_section(connection) -> list[str]:
    p = metrics.pos_daily(connection)
    peak = p["peak_two_hours"]
    return [
        "## 6. The day's bills (POS, whole day)",
        "",
        f"{p['bills']} bills, Rs {p['revenue']:,.2f} in total.",
        "",
        f"- Busiest two hours: {peak['start_hour']}:00 to {peak['end_hour']}:00 with {peak['bills']} bills "
        f"({percent(peak['bill_share'])} of bills) and {percent(peak['revenue_share'])} of revenue.",
        f"- The top 20% of bills ({p['top_bills']} bills) hold {percent(p['top_20_percent_revenue_share'])} of revenue.",
        f"- Mean basket Rs {p['mean_basket']:,.2f}, median Rs {p['median_basket']:,.2f}: the mean is "
        f"{p['mean_to_median']}x the median, so a few large baskets pull the average up.",
        "",
        *table(["Hour", "Bills", "Revenue (Rs)"], [[f"{h['hour']}:00", h["bills"], f"{h['revenue']:,.2f}"]
                                                  for h in p["hourly"]]),
        "",
    ]


def trust_section(config: StoreConfig) -> list[str]:
    entry = config.camera("entry")
    entry_rows = read_tracks_csv(config.outputs_dir / "tracks" / "entry.csv")
    entry_scores = ablations(entry, entry_rows, config.events, read_entry_truth(ENTRY_TRUTH, 0.0), 0.0)
    zone_2 = config.camera("zone_2")
    intervals = zone_intervals(zone_2, read_tracks_csv(config.outputs_dir / "tracks" / "zone_2.csv"), config.events)
    truth = read_ground_truth(ZONE_TRUTH)
    zone_scores = {z: score_zone("", z, truth, intervals[z], sampling_step(ZONE_TRUTH)) for z in ZONES}
    lakme, summer = zone_scores["lakme_wall"], zone_scores["summer_display"]
    staff_file = config.outputs_dir / "staff_comparison.md"

    return [
        "## 7. How far to trust these numbers",
        "",
        "**Detector** (outputs/finetune.md). yolo11n fine-tuned on 328 frames from this store, on held-out frames:",
        "",
        *table(["Model", "Frames", "Precision", "Recall", "mAP50", "mAP50-95", "CPU ms per frame"],
               [[cell.strip() for cell in line.strip("|").split("|")[0:7]]
                for line in lines_containing(config.outputs_dir / "finetune.md", "| all cameras |")]),
        "",
        f"**Door counts** (outputs/evaluation.md). The ground truth has 0 crossings in 148 s. The pipeline counts "
        f"{entry_scores[0].false_positives} false crossings and rejects the man at the door; the same tracks with the "
        f"door line stretched across the frame would count {entry_scores[3].false_positives}.",
        "",
        f"**Zone occupancy** (outputs/zone_evaluation.md, zone_2, 63 samples every 2 s, each about plus or minus 1). "
        f"Mean absolute error {lakme.mean_absolute_error:.2f} people at the Lakme wall and "
        f"{summer.mean_absolute_error:.2f} at the summer display. Shopper time at the Lakme wall is "
        f"{lakme.true_person_seconds / summer.true_person_seconds:.1f}x the summer display's in the ground truth and "
        f"{lakme.predicted_person_seconds / summer.predicted_person_seconds:.1f}x from the pipeline: the direction is "
        "right, the gap is understated, because the display hides the feet of people at the wall behind it.",
        "",
        "**Staff** (outputs/staff_comparison.md), in-store cameras:",
        "",
        *section_table(staff_file, "### In-store cameras"),
        "",
        *lines_containing(staff_file, "The weak spot"),
        "",
        *lines_containing(staff_file, "No threshold fixes it"),
        "",
        "**Ground truth** for doors, zones, staff and detector boxes was labelled from raw frames and crops, never",
        "from model output; so far only the door labels have been double-checked.",
        "",
    ]


def build_report(config: StoreConfig, db_path: Path) -> str:
    connection = metrics.connect(db_path)
    window = metrics.video_window(connection)
    window_minutes = config.pos.conversion_window_minutes
    header = [
        f"# Store report: {config.store_name} ({config.store_id})",
        "",
        f"Video: {window['start'][:10]}, {window['start'][11:]} to {window['end'][11:]} IST, four cameras of about "
        "2.5 minutes each (entrance, two shelf areas, billing counter). Bills: the whole day.",
        "",
    ]
    sections = [footfall_section(connection), zone_section(connection), queue_section(connection),
                conversion_section(connection, window_minutes), funnel_section(connection, window_minutes),
                pos_section(connection), trust_section(config)]
    return "\n".join(header + [line for section in sections for line in section])


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Write outputs/report.md from outputs/store.db.")
    parser.add_argument("--db", type=Path, default=config.outputs_dir / "store.db")
    args = parser.parse_args()
    path = config.outputs_dir / "report.md"
    path.write_text(build_report(config, args.db) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
