"""Fine-tuning, step 1: sample still frames from each clip for fine-tuning, split by time into train and val.

Writes data/finetune/images/{train,val}/<camera>_<frame>.jpg. Neighbouring video frames are near copies,
so a random split would test the model on frames it has effectively seen. Instead the first part of
every clip trains and the last part validates, with a gap between them.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import Camera, FinetuneSettings, load_config

DATA_DIR = Path("data/finetune")
SPLITS = ("train", "val")


@dataclass
class PlannedFrame:
    frame: int
    seconds: float
    split: str


def plan_samples(total_frames: int, fps: float, settings: FinetuneSettings) -> list[PlannedFrame]:
    """Evenly spaced frames; the first (1 - val_fraction) of the clip is train, the rest val.

    Frames within `split_gap_seconds` of the boundary are dropped from both sides.
    """
    duration = total_frames / fps
    spacing = duration / settings.frames_per_camera
    if spacing < settings.min_gap_seconds:
        raise ValueError(f"{settings.frames_per_camera} frames over {duration:.0f} s are only {spacing:.2f} s apart")

    boundary = duration * (1 - settings.val_fraction)
    planned = []
    for index in range(settings.frames_per_camera):
        seconds = (index + 0.5) * spacing
        if abs(seconds - boundary) < settings.split_gap_seconds:
            continue
        frame = min(int(round(seconds * fps)), total_frames - 1)
        planned.append(PlannedFrame(frame, round(frame / fps, 3), "train" if seconds < boundary else "val"))
    return planned


def image_name(camera_id: str, frame: int) -> str:
    return f"{camera_id}_{frame:05d}.jpg"


def camera_of(image_name: str) -> str:
    """'zone_2_01234.jpg' -> 'zone_2'."""
    return image_name.rsplit("_", 1)[0]


def write_dataset_yaml(root: Path, train: str | Path, val: str | Path, path: Path) -> Path:
    """An Ultralytics dataset file: where the images are and what the one class is called."""
    path.write_text(
        f"path: {root.resolve()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        "names:\n"
        "  0: person\n"
    )
    return path


def sample_camera(camera: Camera, settings: FinetuneSettings, out_dir: Path) -> dict[str, int]:
    capture = cv2.VideoCapture(str(camera.clip_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    counts = {split: 0 for split in SPLITS}
    for planned in plan_samples(total_frames, fps, settings):
        capture.set(cv2.CAP_PROP_POS_FRAMES, planned.frame)
        ok, image = capture.read()
        if not ok:
            continue
        target = out_dir / "images" / planned.split / image_name(camera.id, planned.frame)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[planned.split] += 1
    capture.release()
    return counts


def main() -> None:
    argparse.ArgumentParser(description="Sample fine-tuning frames from every clip.").parse_args()
    config = load_config()
    for camera in config.cameras:
        counts = sample_camera(camera, config.finetune, DATA_DIR)
        print(f"{camera.id:<9} train {counts['train']:>4}  val {counts['val']:>4}")


if __name__ == "__main__":
    main()
