"""Fine-tuning, step 6: compare detectors before and after fine-tuning, and write outputs/finetune.md.

Two levels of evidence:
  model level     precision, recall and mAP on the held-out val frames (labels: teacher boxes corrected by review);
  business level  what the pipeline actually reports: zone_2 occupancy against its ground truth on the held-out
                  tail of the clip, entry false crossings, flicker tracks and censored zone visits.
Tracks for each detector are cached in outputs/compare/<detector>/ so reruns are cheap.
"""

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from src.apply_corrections import CORRECTIONS, TEACHER_BOXES, apply, read_corrections
from src.config import Camera, StoreConfig, load_config
from src.evaluate import GROUND_TRUTH as ENTRY_TRUTH
from src.evaluate import read_ground_truth as read_entry_truth
from src.evaluate import score as score_entry
from src.evaluate_zones import GROUND_TRUTH as ZONE_TRUTH
from src.evaluate_zones import ZONES, read_ground_truth, sampling_step, score_zone, zone_intervals
from src.events import area_events, drop_short_tracks, group_by_track
from src.labels import images_by_split, read_boxes
from src.sample_frames import DATA_DIR, camera_of, write_dataset_yaml
from src.track_table import TrackRow, read_tracks_csv, write_tracks_csv

COMPARE_DIR = Path("outputs/compare")
REPORT = Path("outputs/finetune.md")
FINETUNED = "outputs/models/store_yolo11n.pt"
ZONE_TAIL_FROM = 90.0  # the zone_2 samples inside its val split (the last 30% of the clip)
TIMING_IMAGES = 30


@dataclass
class Detector:
    label: str
    weights: str
    tracker_confidence: float

    @property
    def cache_dir(self) -> Path:
        return COMPARE_DIR / f"{Path(self.weights).stem}_conf{self.tracker_confidence:.2f}"


def tracks_for(detector: Detector, camera: Camera) -> list[TrackRow]:
    path = detector.cache_dir / f"{camera.id}.csv"
    if not path.exists():
        from src.track import track_clip  # only a missing cache needs the detector

        print(f"tracking {camera.id} with {detector.label} ...", flush=True)
        rows, _ = track_clip(camera, detector.weights, detector.tracker_confidence)
        write_tracks_csv(rows, path)
    return read_tracks_csv(path)


def tracks_under_one_second(rows: list[TrackRow]) -> str:
    tracks = group_by_track(rows)
    short = len(tracks) - len(drop_short_tracks(tracks, 1.0))
    return f"{short} of {len(tracks)}"


def censored_share(camera: Camera, rows: list[TrackRow], config: StoreConfig) -> str:
    visits = []
    for track_id, samples in drop_short_tracks(group_by_track(rows), config.events.min_track_seconds).items():
        visits += area_events(camera, track_id, samples, camera.zones, config.events.debounce_seconds,
                              "ZONE_ENTER", "ZONE_EXIT")[1]
    censored = sum(visit.censored for visit in visits)
    return f"{censored / len(visits):.0%} ({censored} of {len(visits)})" if visits else "n/a"


def zone_cells(camera: Camera, rows: list[TrackRow], config: StoreConfig, from_seconds: float) -> list[str]:
    truth = read_ground_truth(ZONE_TRUTH, from_seconds)
    intervals = zone_intervals(camera, rows, config.events)
    cells = []
    for zone in ZONES:
        s = score_zone("", zone, truth, intervals[zone], sampling_step(ZONE_TRUTH))
        cells.append(f"{s.mean_absolute_error:.2f} / {s.bias:+.2f} / {s.exact_share:.0%}")
    return cells


def entry_cells(camera: Camera, rows: list[TrackRow], config: StoreConfig, from_seconds: float) -> list[str]:
    truth = read_entry_truth(ENTRY_TRUTH, from_seconds)
    result = score_entry("", camera, rows, config.events, truth, from_seconds)
    return [str(result.false_positives), "yes" if result.near_miss_rejected else "NO"]


def business_table(detectors: list[Detector], config: StoreConfig, zone_from: float, entry_from: float,
                   with_track_quality: bool = True) -> str:
    """Zone cells are mean absolute error / bias / exact share. Track-quality columns always cover whole clips."""
    entry, zone_2 = config.camera("entry"), config.camera("zone_2")
    header = ["Detector", "Lakme MAE / bias / exact", "Summer MAE / bias / exact", "Entry false crossings",
              "Near miss rejected"]
    if with_track_quality:
        header += ["zone_2 tracks under 1 s", "zone_2 censored visits", "Entry tracks under 1 s"]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for d in detectors:
        zone_rows, entry_rows = tracks_for(d, zone_2), tracks_for(d, entry)
        cells = [d.label, *zone_cells(zone_2, zone_rows, config, zone_from),
                 *entry_cells(entry, entry_rows, config, entry_from)]
        if with_track_quality:
            cells += [tracks_under_one_second(zone_rows), censored_share(zone_2, zone_rows, config),
                      tracks_under_one_second(entry_rows)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def val_yaml(camera: str | None, work_dir: Path) -> Path:
    """Val frames of one camera (or all), as an Ultralytics image list."""
    images = sorted((DATA_DIR / "images" / "val").glob("*.jpg"))
    chosen = [str(p.resolve()) for p in images if camera is None or camera_of(p.name) == camera]
    list_path = work_dir / f"val_{camera or 'all'}.txt"
    list_path.write_text("\n".join(chosen) + "\n")
    return write_dataset_yaml(DATA_DIR, list_path, list_path, work_dir / f"val_{camera or 'all'}.yaml")


def ms_per_frame(weights: str, imgsz: int) -> float:
    from ultralytics import YOLO

    model = YOLO(weights)
    images = sorted((DATA_DIR / "images" / "val").glob("*.jpg"))[:TIMING_IMAGES]
    model.predict(str(images[0]), imgsz=imgsz, device="cpu", verbose=False)  # warm-up
    started = time.perf_counter()
    for image in images:
        model.predict(str(image), imgsz=imgsz, device="cpu", verbose=False)
    return (time.perf_counter() - started) / len(images) * 1000


def model_level_table(rows: list[tuple[str, str, str | None]], imgsz: int) -> str:
    from ultralytics import YOLO

    work_dir = Path(tempfile.mkdtemp(prefix="val_"))
    lines = ["| Model | Val frames | Precision | Recall | mAP50 | mAP50-95 | CPU ms per frame |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, weights, camera in rows:
        if not Path(weights).exists() and weights.startswith("outputs/"):
            lines.append(f"| {label} | not trained yet | | | | | |")
            continue
        metrics = YOLO(weights).val(data=str(val_yaml(camera, work_dir)), split="val", imgsz=imgsz, batch=8,
                                    classes=[0], device="cpu", plots=False, verbose=False,
                                    project=str(work_dir), name="val")
        box = metrics.box
        frames = "all cameras" if camera is None else camera
        lines.append(f"| {label} | {frames} | {box.mp:.3f} | {box.mr:.3f} | {box.map50:.3f} | {box.map:.3f} "
                     f"| {ms_per_frame(weights, imgsz):.0f} |")
    return "\n".join(lines)


def review_counts() -> str:
    if not TEACHER_BOXES.exists():
        return "No teacher labels yet."
    splits = images_by_split(DATA_DIR / "images")
    split_of = {image: split for split, images in splits.items() for image in images}
    proposals = read_boxes(TEACHER_BOXES)
    labelled, counts = apply(proposals, read_corrections(CORRECTIONS), split_of)
    teacher_labels = sum(b.source == "teacher" for b in proposals)
    frames = sum(len(images) for images in splits.values())
    return (f"{frames} frames ({', '.join(f'{len(v)} {k}' for k, v in splits.items())}). The teacher proposed "
            f"{teacher_labels} labels (score 0.3 or more) and {len(proposals) - teacher_labels} weaker candidates. "
            f"The review removed {counts['removed']} labels (posters, TV screens, reflections), accepted "
            f"{counts['accepted']} candidates as real people, moved {counts['moved']} and drew {counts['added']} "
            f"new boxes, leaving {len(labelled)} labels.")


def clip_seconds(camera: Camera) -> float:
    import cv2

    capture = cv2.VideoCapture(str(camera.clip_path))
    seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detectors and write outputs/finetune.md.")
    parser.add_argument("--skip-model-level", action="store_true", help="only the business-level tables")
    args = parser.parse_args()
    config = load_config()
    settings = config.finetune

    pretrained_low = Detector("pretrained yolo11n, tracker input 0.10", "yolo11n.pt", 0.10)
    pretrained_high = Detector("pretrained yolo11n, tracker input 0.25 (original setting)", "yolo11n.pt", 0.25)
    baseline = pretrained_low if config.confidence == 0.10 else pretrained_high
    finetuned = Detector("fine-tuned yolo11n", FINETUNED, config.confidence)
    after = [finetuned] if Path(FINETUNED).exists() else []

    entry_from = round(clip_seconds(config.camera("entry")) * (1 - settings.val_fraction), 1)
    sections = [
        "# Detector fine-tuning",
        "",
        "## 1. Tracker input confidence",
        "",
        "ByteTrack's second pass matches boxes scored between 0.1 and 0.25 to existing tracks. Passing conf=0.25",
        "to model.track removes those boxes before the tracker sees them, so the second pass never ran.",
        "Same pretrained weights, whole clips. Zone cells: mean absolute error / bias / exact share.",
        "",
        business_table([pretrained_high, pretrained_low], config, 0.0, 0.0),
        "",
        "## 2. Labels",
        "",
        review_counts(),
        "",
    ]
    if not args.skip_model_level:
        sections += [
            "## 3. Model level (held-out val frames, last 30% of each clip)",
            "",
            "Val labels are the teacher's boxes corrected by review, so the teacher's own score is optimistic by",
            "construction: it is graded mostly against its own answers. Every model is scored at 640 px, the",
            "students' input size; the teacher made its labels at 960 px. CPU time is on this laptop.",
            "",
            model_level_table([
                ("pretrained yolo11n", "yolo11n.pt", None),
                (f"teacher {Path(settings.teacher_model).stem}", settings.teacher_model, None),
                ("fine-tuned yolo11n", FINETUNED, None),
                *[(f"{label} ({camera})", weights, camera)
                  for camera in ("entry", "zone_1", "zone_2", "billing")
                  for label, weights in (("pretrained yolo11n", "yolo11n.pt"), ("fine-tuned yolo11n", FINETUNED))],
            ], settings.imgsz),
            "",
        ]
    sections += [
        "## 4. Business level, held-out tail only",
        "",
        f"zone_2 from {ZONE_TAIL_FROM:.0f} s (18 samples, each about plus or minus 1) and entry from {entry_from} s",
        "(no true crossings, so this only counts false alarms).",
        "",
        business_table([baseline, *after], config, ZONE_TAIL_FROM, entry_from, with_track_quality=False),
        "",
        "## 5. Business level, whole clips (partly seen in training, shown for context)",
        "",
        business_table([baseline, *after], config, 0.0, 0.0),
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(sections) + "\n")
    print("\n".join(sections))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
