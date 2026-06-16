"""Scores zone_2's zone visits against occupancy ground truth in data/ground_truth/zone_occupancy.csv.

The ground truth says how many people are at each zone every 2 s. For the same moments we count how many
tracks the pipeline places in that zone, and compare. Writes outputs/zone_evaluation.md.
"""

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

from src.config import Camera, EventSettings, load_config
from src.events import area_events, drop_short_tracks, group_by_track
from src.track_table import TrackRow, read_tracks_csv

GROUND_TRUTH = Path("data/ground_truth/zone_occupancy.csv")
CAMERA_ID = "zone_2"
ZONES = ["lakme_wall", "summer_display"]
# Stretches of the clip with different crowding, from data/ground_truth/zone_review_notes.md.
PERIODS = [("0 to 60 s, one or two people", 0, 61),
           ("62 to 106 s, three people in black side by side", 62, 107),
           ("108 to 124 s, one person", 108, 125)]


@dataclass
class ZoneScore:
    setting: str
    zone: str
    samples: int
    mean_absolute_error: float
    bias: float
    exact_share: float
    true_person_seconds: float
    predicted_person_seconds: float


def read_ground_truth(path: Path, from_seconds: float = 0.0) -> dict[float, dict[str, int]]:
    with path.open() as file:
        return {
            float(row["seconds"]): {zone: int(row[zone]) for zone in ZONES}
            for row in csv.DictReader(file)
            if float(row["seconds"]) >= from_seconds
        }


def sampling_step(path: Path) -> float:
    times = sorted(read_ground_truth(path))
    return times[1] - times[0]


def zone_intervals(camera: Camera, rows: list[TrackRow], settings: EventSettings) -> dict[str, list[tuple[float, float]]]:
    """Time spans each track spent in each zone, after the same filtering the events step uses."""
    intervals: dict[str, list[tuple[float, float]]] = {zone: [] for zone in camera.zones}
    tracks = drop_short_tracks(group_by_track(rows), settings.min_track_seconds)
    for track_id, samples in tracks.items():
        _, visits = area_events(camera, track_id, samples, camera.zones, settings.debounce_seconds,
                                "ZONE_ENTER", "ZONE_EXIT")
        for visit in visits:
            intervals[visit.zone].append((samples[visit.start].seconds, samples[visit.end].seconds))
    return intervals


def occupancy_at(intervals: list[tuple[float, float]], seconds: float) -> int:
    return sum(1 for start, end in intervals if start <= seconds <= end)


def score_zone(setting: str, zone: str, truth: dict[float, dict[str, int]],
               intervals: list[tuple[float, float]], step_seconds: float) -> ZoneScore:
    errors = [occupancy_at(intervals, t) - counts[zone] for t, counts in truth.items()]
    true_total = sum(counts[zone] for counts in truth.values())
    return ZoneScore(
        setting=setting,
        zone=zone,
        samples=len(errors),
        mean_absolute_error=sum(abs(e) for e in errors) / len(errors),
        bias=sum(errors) / len(errors),
        exact_share=sum(e == 0 for e in errors) / len(errors),
        true_person_seconds=true_total * step_seconds,
        predicted_person_seconds=(true_total + sum(errors)) * step_seconds,
    )


def table(scores: list[ZoneScore]) -> str:
    lines = [
        "| Setting | Zone | Samples | Mean abs. error | Bias | Exact | True person-s | Predicted person-s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in scores:
        lines.append(
            f"| {s.setting} | {s.zone} | {s.samples} | {s.mean_absolute_error:.2f} | {s.bias:+.2f} "
            f"| {s.exact_share:.0%} | {s.true_person_seconds:.0f} | {s.predicted_person_seconds:.0f} |"
        )
    return "\n".join(lines)


def bias_by_period(truth: dict[float, dict[str, int]], intervals: dict[str, list[tuple[float, float]]]) -> str:
    lines = ["| Period | Zone | True people (mean) | Bias |", "| --- | --- | --- | --- |"]
    for name, start, end in PERIODS:
        times = [t for t in truth if start <= t < end]
        if not times:
            continue
        for zone in ZONES:
            true_mean = sum(truth[t][zone] for t in times) / len(times)
            bias = sum(occupancy_at(intervals[zone], t) - truth[t][zone] for t in times) / len(times)
            lines.append(f"| {name} | {zone} | {true_mean:.2f} | {bias:+.2f} |")
    return "\n".join(lines)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Score zone_2 zone occupancy against the ground truth.")
    parser.add_argument("--model", help="retrack zone_2 with these weights instead of reading outputs/tracks/")
    parser.add_argument("--conf", type=float, default=config.confidence)
    parser.add_argument("--from-seconds", type=float, default=0.0, help="score only samples after this clip time")
    args = parser.parse_args()

    camera = config.camera(CAMERA_ID)
    if args.model:
        from src.track import track_clip  # only a retrack needs the detector

        rows, _ = track_clip(camera, args.model, args.conf)
    else:
        rows = read_tracks_csv(config.outputs_dir / "tracks" / f"{camera.id}.csv")
    truth = read_ground_truth(GROUND_TRUTH, args.from_seconds)
    times = sorted(truth)
    step_seconds = sampling_step(GROUND_TRUTH)

    settings_to_try = [
        ("chosen settings", config.events),
        ("no debounce, no minimum length", replace(config.events, debounce_seconds=0.0, min_track_seconds=0.0)),
    ]
    scores = []
    for name, settings in settings_to_try:
        intervals = zone_intervals(camera, rows, settings)
        scores += [score_zone(name, zone, truth, intervals[zone], step_seconds) for zone in ZONES]

    report = "\n".join([
        "# Zone evaluation (zone_2)",
        "",
        f"Ground truth: people at each zone every {step_seconds:.0f} s over {len(times)} samples "
        "(labelled from raw frames by where people stand, not by box position; see data/ground_truth/zone_review_notes.md).",
        "Bias is predicted minus true people per sample: positive means over-counting.",
        "Person-seconds approximate total dwell: the sum of people present times the sampling step.",
        "",
        table(scores),
        "",
        "## Where the error comes from (chosen settings)",
        "",
        bias_by_period(truth, zone_intervals(camera, rows, config.events)),
        "",
        "When the Lakme wall is crowded, its shortfall roughly equals the summer display's surplus: the display hides",
        "the feet of people standing behind it, so their box ends on the display and they are counted there.",
    ])
    print(report)
    if args.model or args.from_seconds:
        return  # a what-if run: print only, keep the main report for the configured tracks
    path = config.outputs_dir / "zone_evaluation.md"
    path.write_text(report + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
