"""Scores the entry camera's ENTRY / EXIT events against the ground truth in data/ground_truth/entry_exit.csv.

Count error alone can hide mistakes (two misses plus two false alarms looks perfect), so each predicted
crossing is matched to a real one of the same direction within a time tolerance before counting.
--sweep also reruns tracking with other models and confidences and writes outputs/evaluation.md.
"""

import argparse
import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path

from src.config import Camera, EventSettings, StoreConfig, load_config, to_pixels
from src.events import camera_events, drop_short_tracks, group_by_track
from src.track_table import TrackRow, read_tracks_csv

GROUND_TRUTH = Path("data/ground_truth/entry_exit.csv")
# A man walks up to the door, stands just outside on his phone and leaves without entering
# (see data/ground_truth/review_notes.md). Any crossing here means the door line is in the wrong place.
NEAR_MISS_WINDOW = (101.8, 111.8)
# Two careless placements: the right door line stretched across the whole frame (what you get if the
# code tested the infinite line instead of the door segment), and a line drawn across the street.
FULL_WIDTH_LINE = ([0.0, 0.95], [1.0, 0.95])
ACROSS_STREET_LINE = ([0.0, 0.75], [1.0, 0.75])
SWEEP = [("yolo11n.pt", 0.25), ("yolo11n.pt", 0.40), ("yolo11s.pt", 0.25), ("yolo11s.pt", 0.40)]
SWEEP_HEADING = "## Detector sweep (entry clip retracked for each setting)"


@dataclass
class Crossing:
    seconds: float
    direction: str


@dataclass
class Score:
    setting: str
    predicted_in: int
    predicted_out: int
    true_crossings: int
    true_positives: int
    false_positives: int
    false_negatives: int
    near_miss_rejected: bool
    tracks: int
    tracks_under_1s: int

    @property
    def precision(self) -> str:
        predicted = self.true_positives + self.false_positives
        return f"{self.true_positives / predicted:.2f}" if predicted else "n/a (nothing predicted)"

    @property
    def recall(self) -> str:
        return f"{self.true_positives / self.true_crossings:.2f}" if self.true_crossings else "n/a (no true crossings)"


def read_ground_truth(path: Path, from_seconds: float) -> list[Crossing]:
    with path.open() as file:
        return [
            Crossing(float(row["seconds"]), row["direction"].strip())
            for row in csv.DictReader(file)
            if float(row["seconds"]) >= from_seconds
        ]


def predicted_crossings(camera: Camera, rows: list[TrackRow], settings: EventSettings, from_seconds: float) -> list[Crossing]:
    events, _ = camera_events(camera, rows, settings)
    direction_of = {"ENTRY": "in", "EXIT": "out"}
    return [
        Crossing(event.seconds, direction_of[event.event_type])
        for event in events
        if event.event_type in direction_of and event.seconds >= from_seconds
    ]


def count_matches(predicted: list[Crossing], truth: list[Crossing], tolerance: float) -> int:
    """Greedy one-to-one matching: each true crossing can be claimed by at most one prediction."""
    unmatched = list(truth)
    matches = 0
    for guess in sorted(predicted, key=lambda c: c.seconds):
        for real in unmatched:
            if real.direction == guess.direction and abs(real.seconds - guess.seconds) <= tolerance:
                unmatched.remove(real)
                matches += 1
                break
    return matches


def score(setting: str, camera: Camera, rows: list[TrackRow], settings: EventSettings,
          truth: list[Crossing], from_seconds: float) -> Score:
    predicted = predicted_crossings(camera, rows, settings, from_seconds)
    matches = count_matches(predicted, truth, settings.match_tolerance_seconds)
    tracks = group_by_track(rows)
    start, end = NEAR_MISS_WINDOW
    return Score(
        setting=setting,
        predicted_in=sum(c.direction == "in" for c in predicted),
        predicted_out=sum(c.direction == "out" for c in predicted),
        true_crossings=len(truth),
        true_positives=matches,
        false_positives=len(predicted) - matches,
        false_negatives=len(truth) - matches,
        near_miss_rejected=not any(start - 2 <= c.seconds <= end + 2 for c in predicted),
        tracks=len(tracks),
        tracks_under_1s=len(tracks) - len(drop_short_tracks(tracks, 1.0)),
    )


def ablations(camera: Camera, rows: list[TrackRow], settings: EventSettings,
              truth: list[Crossing], from_seconds: float) -> list[Score]:
    """The same tracks with one safeguard removed at a time, to show what each one protects against."""
    def with_line(line: tuple[list[float], list[float]]) -> Camera:
        return replace(camera, door_line=(to_pixels(line[0]), to_pixels(line[1])))

    return [
        score("chosen settings", camera, rows, settings, truth, from_seconds),
        score("no debounce", camera, rows, replace(settings, debounce_seconds=0.0), truth, from_seconds),
        score("no minimum track length", camera, rows, replace(settings, min_track_seconds=0.0), truth, from_seconds),
        score("door line stretched to full width", with_line(FULL_WIDTH_LINE), rows, settings, truth, from_seconds),
        score("line drawn across the street", with_line(ACROSS_STREET_LINE), rows, settings, truth, from_seconds),
        score("across the street, no debounce, no min length", with_line(ACROSS_STREET_LINE), rows,
              replace(settings, debounce_seconds=0.0, min_track_seconds=0.0), truth, from_seconds),
    ]


def table(scores: list[Score], extra: dict[str, str] | None = None) -> str:
    extra = extra or {}
    header = ("| Setting | Predicted in | Predicted out | True | False positives | False negatives "
              "| Precision | Recall | Near miss rejected | Tracks | Tracks under 1 s |" + (" Run s |" if extra else ""))
    lines = [header, "|" + " --- |" * (header.count("|") - 1)]
    for s in scores:
        cells = [s.setting, s.predicted_in, s.predicted_out, s.true_crossings, s.false_positives,
                 s.false_negatives, s.precision, s.recall, "yes" if s.near_miss_rejected else "NO",
                 s.tracks, s.tracks_under_1s]
        if extra:
            cells.append(extra.get(s.setting, ""))
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def run_sweep(config: StoreConfig, camera: Camera, truth: list[Crossing], from_seconds: float) -> tuple[list[Score], dict[str, str]]:
    from src.track import track_clip  # imported here: only the sweep needs the detector

    scores, run_seconds = [], {}
    for model_name, confidence in SWEEP:
        setting = f"{model_name.removesuffix('.pt')}, conf {confidence}"
        print(f"tracking entry with {setting} ...", flush=True)
        started = time.perf_counter()
        rows, _ = track_clip(camera, model_name, confidence)
        run_seconds[setting] = f"{time.perf_counter() - started:.0f}"
        scores.append(score(setting, camera, rows, config.events, truth, from_seconds))
    return scores, run_seconds


def write_report(path: Path, ablation_scores: list[Score], sweep: tuple[list[Score], dict[str, str]] | None,
                 truth: list[Crossing], from_seconds: float) -> None:
    lines = [
        "# Entry camera evaluation",
        "",
        f"Ground truth: {len(truth)} door crossings from {from_seconds:.0f} s on "
        "(labelled from raw frames and double-checked; see data/ground_truth/review_notes.md).",
        "With no true crossings, precision and recall are undefined; every predicted crossing is a false positive.",
        "",
        "## What each safeguard protects against (same tracks, one safeguard removed)",
        "",
        table(ablation_scores),
    ]
    if sweep:
        lines += ["", SWEEP_HEADING, "", table(*sweep)]
    elif path.exists() and SWEEP_HEADING in path.read_text():
        # The sweep takes about 25 minutes; a quick rerun without --sweep keeps the last results.
        lines += ["", path.read_text()[path.read_text().index(SWEEP_HEADING):].rstrip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Score entry/exit events against the ground truth.")
    parser.add_argument("--model", help="retrack the entry clip with these weights instead of reading outputs/tracks/")
    parser.add_argument("--conf", type=float, default=config.confidence)
    parser.add_argument("--from-seconds", type=float, default=0.0, help="score only crossings after this clip time")
    parser.add_argument("--sweep", action="store_true", help="also retrack with the models and confidences in SWEEP")
    args = parser.parse_args()

    camera = config.camera("entry")
    truth = read_ground_truth(GROUND_TRUTH, args.from_seconds)
    if args.model:
        from src.track import track_clip

        rows, _ = track_clip(camera, args.model, args.conf)
    else:
        rows = read_tracks_csv(config.outputs_dir / "tracks" / "entry.csv")

    ablation_scores = ablations(camera, rows, config.events, truth, args.from_seconds)
    print(table(ablation_scores))
    sweep = run_sweep(config, camera, truth, args.from_seconds) if args.sweep else None
    if sweep:
        print(table(*sweep))
    write_report(config.outputs_dir / "evaluation.md", ablation_scores, sweep, truth, args.from_seconds)
    print(f"wrote {config.outputs_dir / 'evaluation.md'}")


if __name__ == "__main__":
    main()
