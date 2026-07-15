"""Loads config/store.yaml, the single place that describes the store and its cameras."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.geometry import Point, Polygon

CONFIG_PATH = Path("config/store.yaml")
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


@dataclass
class Camera:
    id: str
    clip_path: Path
    role: str
    clock_start: datetime
    clock_note: str = ""
    door_line: tuple[Point, Point] | None = None
    inside_point: Point | None = None
    zones: dict[str, Polygon] = field(default_factory=dict)
    queue_area: Polygon | None = None
    staff_area: Polygon | None = None

    def clock_at(self, seconds: float) -> datetime:
        """Store-local time shown by the camera clock, `seconds` into the clip."""
        return self.clock_start + timedelta(seconds=seconds)


@dataclass
class EventSettings:
    min_track_seconds: float
    debounce_seconds: float
    match_tolerance_seconds: float


@dataclass
class PosSettings:
    csv: Path
    conversion_window_minutes: int


@dataclass
class FinetuneSettings:
    student_model: str
    teacher_model: str
    teacher_confidence: float
    teacher_candidate_confidence: float
    teacher_imgsz: int
    frames_per_camera: int
    min_gap_seconds: float
    val_fraction: float
    split_gap_seconds: float
    epochs: int
    imgsz: int
    batch: int
    patience: int


@dataclass
class StoreConfig:
    store_id: str
    store_name: str
    timezone: str
    model: str
    confidence: float
    outputs_dir: Path
    cameras: list[Camera]
    events: EventSettings
    pos: PosSettings | None = None
    finetune: FinetuneSettings | None = None

    def camera(self, camera_id: str) -> Camera:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        known = ", ".join(camera.id for camera in self.cameras)
        raise KeyError(f"unknown camera '{camera_id}', expected one of: {known}")


def to_pixels(normalised: list[float]) -> Point:
    """The YAML stores points as fractions of the frame (0 to 1), so they survive a resolution change."""
    return (normalised[0] * FRAME_WIDTH, normalised[1] * FRAME_HEIGHT)


def to_polygon(points: list[list[float]] | None) -> Polygon | None:
    return [to_pixels(p) for p in points] if points else None


def load_camera(entry: dict, clips_dir: Path) -> Camera:
    door = entry.get("door_line")
    return Camera(
        id=entry["id"],
        clip_path=clips_dir / entry["clip"],
        role=entry["role"],
        clock_start=datetime.strptime(entry["clock_start"], "%Y-%m-%d %H:%M:%S"),
        clock_note=entry.get("clock_note", ""),
        door_line=(to_pixels(door[0]), to_pixels(door[1])) if door else None,
        inside_point=to_pixels(entry["inside"]) if entry.get("inside") else None,
        zones={name: to_polygon(points) for name, points in (entry.get("zones") or {}).items()},
        queue_area=to_polygon(entry.get("queue_area")),
        staff_area=to_polygon(entry.get("staff_area")),
    )


def load_config(path: Path = CONFIG_PATH) -> StoreConfig:
    raw = yaml.safe_load(path.read_text())
    clips_dir = Path(raw["clips_dir"])
    return StoreConfig(
        store_id=raw["store_id"],
        store_name=raw["store_name"],
        timezone=raw["timezone"],
        model=raw["detection"]["model"],
        confidence=raw["detection"]["confidence"],
        outputs_dir=Path(raw["outputs_dir"]),
        cameras=[load_camera(entry, clips_dir) for entry in raw["cameras"]],
        events=EventSettings(**raw["events"]),
        pos=PosSettings(Path(raw["pos"]["csv"]), raw["pos"]["conversion_window_minutes"]) if raw.get("pos") else None,
        finetune=FinetuneSettings(**raw["finetune"]) if raw.get("finetune") else None,
    )
