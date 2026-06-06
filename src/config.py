"""Loads config/store.yaml, the single place that describes the store and its cameras."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

CONFIG_PATH = Path("config/store.yaml")


@dataclass
class Camera:
    id: str
    clip_path: Path
    role: str
    clock_start: datetime
    clock_note: str = ""

    def clock_at(self, seconds: float) -> datetime:
        """Store-local time shown by the camera clock, `seconds` into the clip."""
        return self.clock_start + timedelta(seconds=seconds)


@dataclass
class StoreConfig:
    store_id: str
    store_name: str
    timezone: str
    model: str
    confidence: float
    outputs_dir: Path
    cameras: list[Camera]

    def camera(self, camera_id: str) -> Camera:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        known = ", ".join(camera.id for camera in self.cameras)
        raise KeyError(f"unknown camera '{camera_id}', expected one of: {known}")


def load_camera(entry: dict, clips_dir: Path) -> Camera:
    return Camera(
        id=entry["id"],
        clip_path=clips_dir / entry["clip"],
        role=entry["role"],
        clock_start=datetime.strptime(entry["clock_start"], "%Y-%m-%d %H:%M:%S"),
        clock_note=entry.get("clock_note", ""),
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
    )
