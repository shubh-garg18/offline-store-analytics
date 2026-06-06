"""Step 1 of the pipeline: find every person in every frame and give each one a stable track ID.

Writes outputs/tracks/<camera>.csv (one row per person per frame) and, with --save-video,
an annotated video in outputs/videos/ to check the tracking by eye.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import Camera, StoreConfig, load_config
from src.track_table import TrackRow, write_tracks_csv

PERSON_CLASS = 0  # "person" in the COCO dataset the model was trained on
TRACKER = "bytetrack.yaml"


@dataclass
class ClipSummary:
    camera_id: str
    frames: int
    duration_seconds: float
    unique_tracks: int
    detections: int
    mean_confidence: float
    run_seconds: float


def read_fps(clip_path: Path) -> float:
    capture = cv2.VideoCapture(str(clip_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not fps:
        raise ValueError(f"could not read fps from {clip_path}")
    return fps


def open_video_writer(path: Path, fps: float, frame_shape: tuple[int, ...]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def draw_clock(frame, camera: Camera, seconds: float) -> None:
    label = f"{camera.id}  {camera.clock_at(seconds):%H:%M:%S}  t={seconds:6.1f}s"
    cv2.rectangle(frame, (0, frame.shape[0] - 50), (620, frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, label, (12, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def track_clip(
    camera: Camera,
    model_name: str,
    confidence: float,
    video_path: Path | None = None,
) -> tuple[list[TrackRow], int]:
    """Run detection plus ByteTrack over one clip. Returns the rows and the number of frames seen."""
    # A fresh model per clip: the tracker inside it remembers people between frames,
    # and reusing it would carry IDs from one camera into the next.
    model = YOLO(model_name)
    fps = read_fps(camera.clip_path)

    rows: list[TrackRow] = []
    writer = None
    frame_count = 0

    results = model.track(
        source=str(camera.clip_path),
        stream=True,
        persist=True,
        tracker=TRACKER,
        classes=[PERSON_CLASS],
        conf=confidence,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        frame_count += 1
        seconds = frame_index / fps
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            for track_id, (x1, y1, x2, y2), score in zip(
                boxes.id.int().tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()
            ):
                rows.append(
                    TrackRow(
                        frame=frame_index,
                        seconds=round(seconds, 3),
                        track_id=track_id,
                        x1=round(x1, 1),
                        y1=round(y1, 1),
                        x2=round(x2, 1),
                        y2=round(y2, 1),
                        confidence=round(score, 3),
                    )
                )

        if video_path is not None:
            annotated = result.plot()
            draw_clock(annotated, camera, seconds)
            if writer is None:
                writer = open_video_writer(video_path, fps, annotated.shape)
            writer.write(annotated)

    if writer is not None:
        writer.release()
    return rows, frame_count


def summarize(camera: Camera, rows: list[TrackRow], frame_count: int, run_seconds: float) -> ClipSummary:
    fps = read_fps(camera.clip_path)
    mean_confidence = sum(row.confidence for row in rows) / len(rows) if rows else 0.0
    return ClipSummary(
        camera_id=camera.id,
        frames=frame_count,
        duration_seconds=round(frame_count / fps, 1),
        unique_tracks=len({row.track_id for row in rows}),
        detections=len(rows),
        mean_confidence=round(mean_confidence, 3),
        run_seconds=round(run_seconds, 1),
    )


def print_summaries(summaries: list[ClipSummary]) -> None:
    header = f"{'camera':<10}{'frames':>8}{'clip s':>8}{'tracks':>8}{'boxes':>9}{'mean conf':>11}{'run s':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s.camera_id:<10}{s.frames:>8}{s.duration_seconds:>8}{s.unique_tracks:>8}"
            f"{s.detections:>9}{s.mean_confidence:>11}{s.run_seconds:>8}"
        )


def run(config: StoreConfig, camera_ids: list[str], model_name: str, confidence: float, save_video: bool,
        tracks_dir: Path) -> None:
    summaries = []
    for camera_id in camera_ids:
        camera = config.camera(camera_id)
        video_path = config.outputs_dir / "videos" / f"{camera.id}.mp4" if save_video else None

        print(f"tracking {camera.id} ({camera.clip_path.name}) ...", flush=True)
        started = time.perf_counter()
        rows, frame_count = track_clip(camera, model_name, confidence, video_path)
        write_tracks_csv(rows, tracks_dir / f"{camera.id}.csv")
        summaries.append(summarize(camera, rows, frame_count, time.perf_counter() - started))

    print()
    print_summaries(summaries)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Detect and track people in the store clips.")
    parser.add_argument("--camera", action="append", help="camera id to process (repeatable); default: all")
    parser.add_argument("--model", default=config.model, help=f"YOLO weights (default {config.model})")
    parser.add_argument("--conf", type=float, default=config.confidence, help="minimum detection confidence")
    parser.add_argument("--save-video", action="store_true", help="also write annotated videos to outputs/videos/")
    parser.add_argument("--out-dir", type=Path, default=config.outputs_dir / "tracks",
                        help="where to write the tracks CSVs (default outputs/tracks)")
    args = parser.parse_args()

    camera_ids = args.camera or [camera.id for camera in config.cameras]
    run(config, camera_ids, args.model, args.conf, args.save_video, args.out_dir)


if __name__ == "__main__":
    main()
