"""The tracks table: one row per person per frame, written by track.py and read by every later step."""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrackRow:
    frame: int
    seconds: float
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def write_tracks_csv(rows: list[TrackRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(TrackRow)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_tracks_csv(path: Path) -> list[TrackRow]:
    with path.open() as file:
        return [
            TrackRow(
                frame=int(row["frame"]),
                seconds=float(row["seconds"]),
                track_id=int(row["track_id"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                confidence=float(row["confidence"]),
            )
            for row in csv.DictReader(file)
        ]
