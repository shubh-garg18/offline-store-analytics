"""Step 5 of the pipeline: cut each tracked person out of the video, for staff labelling and the staff classifiers.

For every track of at least `min_track_seconds`, keeps up to 5 crops: the frames where the person's box is
largest (the clearest view), at least 1 s apart so they are not near copies. Writes
outputs/crops/<camera>/<track>_<frame>.jpg, outputs/crops/index.csv, and labelling sheets in outputs/staff/.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from src.config import Camera, load_config
from src.track_table import TrackRow, read_tracks_csv

CROPS_DIR = Path("outputs/crops")
SHEETS_DIR = Path("outputs/staff")
CROPS_PER_TRACK = 5
MIN_GAP_SECONDS = 1.0
PADDING = 0.1
TILE_HEIGHT = 200
CONTEXT_WIDTH = 320
SHEET_COLUMNS = 3
SHEET_ROWS = 6


@dataclass
class Crop:
    camera: str
    track_id: int
    frame: int
    seconds: float
    track_seconds: float
    confidence: float
    path: str


def box_area(row: TrackRow) -> float:
    return (row.x2 - row.x1) * (row.y2 - row.y1)


def choose_rows(rows: list[TrackRow], fps: float) -> list[TrackRow]:
    """Largest boxes first, skipping any frame within MIN_GAP_SECONDS of one already chosen."""
    chosen: list[TrackRow] = []
    for row in sorted(rows, key=box_area, reverse=True):
        if all(abs(row.frame - kept.frame) >= MIN_GAP_SECONDS * fps for kept in chosen):
            chosen.append(row)
        if len(chosen) == CROPS_PER_TRACK:
            break
    return sorted(chosen, key=lambda row: row.frame)


def cut(frame: np.ndarray, row: TrackRow) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_x, pad_y = (row.x2 - row.x1) * PADDING, (row.y2 - row.y1) * PADDING
    x1, y1 = int(max(0, row.x1 - pad_x)), int(max(0, row.y1 - pad_y))
    x2, y2 = int(min(width, row.x2 + pad_x)), int(min(height, row.y2 + pad_y))
    return frame[y1:y2, x1:x2]


def context_thumbnail(frame: np.ndarray, row: TrackRow) -> np.ndarray:
    """The whole frame, small, with the person's box drawn: where they stand says a lot about who they are."""
    marked = frame.copy()
    cv2.rectangle(marked, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), (0, 0, 255), 8)
    return cv2.resize(marked, (CONTEXT_WIDTH, int(frame.shape[0] * CONTEXT_WIDTH / frame.shape[1])))


def crop_camera(camera: Camera, rows: list[TrackRow], min_track_seconds: float) -> tuple[list[Crop], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(camera.clip_path))
    fps = capture.get(cv2.CAP_PROP_FPS)

    rows_by_track: dict[int, list[TrackRow]] = defaultdict(list)
    for row in rows:
        rows_by_track[row.track_id].append(row)
    wanted_by_frame: dict[int, list[TrackRow]] = defaultdict(list)
    track_seconds: dict[int, float] = {}
    for track_id, track_rows in rows_by_track.items():
        duration = max(r.seconds for r in track_rows) - min(r.seconds for r in track_rows)
        if duration < min_track_seconds:
            continue
        track_seconds[track_id] = duration
        for row in choose_rows(track_rows, fps):
            wanted_by_frame[row.frame].append(row)

    crops: list[Crop] = []
    images_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    context_by_track: dict[int, np.ndarray] = {}
    out_dir = CROPS_DIR / camera.id
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_index, last_wanted = 0, max(wanted_by_frame, default=-1)
    # Reading the clip once, front to back, is much faster than seeking to each wanted frame.
    while frame_index <= last_wanted:
        ok, frame = capture.read()
        if not ok:
            break
        for row in wanted_by_frame.get(frame_index, []):
            image = cut(frame, row)
            path = out_dir / f"{row.track_id}_{row.frame:05d}.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            crops.append(Crop(camera.id, row.track_id, row.frame, row.seconds,
                              round(track_seconds[row.track_id], 2), row.confidence, str(path)))
            images_by_track[row.track_id].append(image)
            best = max(rows_by_track[row.track_id], key=box_area)
            if row.frame == best.frame:
                context_by_track[row.track_id] = context_thumbnail(frame, row)
        frame_index += 1
    capture.release()

    sheet_tiles = [labelling_tile(camera.id, track_id, track_seconds[track_id], context_by_track[track_id],
                                  images_by_track[track_id])
                   for track_id in sorted(images_by_track) if track_id in context_by_track]
    return crops, sheet_tiles


def labelling_tile(camera_id: str, track_id: int, seconds: float, context: np.ndarray,
                   images: list[np.ndarray]) -> np.ndarray:
    tile_width = 1920 // SHEET_COLUMNS
    tile = np.zeros((TILE_HEIGHT + 30, tile_width, 3), np.uint8)
    context = cv2.resize(context, (CONTEXT_WIDTH, TILE_HEIGHT))
    tile[30:, :CONTEXT_WIDTH] = context
    x = CONTEXT_WIDTH + 4
    for image in images[:3]:
        scaled = cv2.resize(image, (max(1, int(image.shape[1] * TILE_HEIGHT / image.shape[0])), TILE_HEIGHT))
        width = min(scaled.shape[1], tile_width - x)
        if width <= 0:
            break
        tile[30:, x:x + width] = scaled[:, :width]
        x += width + 4
    cv2.putText(tile, f"{camera_id} track {track_id}  {seconds:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return tile


def write_sheets(camera_id: str, tiles: list[np.ndarray]) -> int:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [np.zeros_like(tiles[0])] * (per_sheet - len(chunk))
        rows = [np.hstack(chunk[r * SHEET_COLUMNS:(r + 1) * SHEET_COLUMNS]) for r in range(SHEET_ROWS)]
        cv2.imwrite(str(SHEETS_DIR / f"{camera_id}_{start // per_sheet:02d}.jpg"), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (len(tiles) + per_sheet - 1) // per_sheet


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Crop every tracked person for staff labelling.")
    parser.add_argument("--tracks-dir", type=Path, default=config.outputs_dir / "tracks")
    args = parser.parse_args()

    all_crops: list[Crop] = []
    for camera in config.cameras:
        rows = read_tracks_csv(args.tracks_dir / f"{camera.id}.csv")
        crops, tiles = crop_camera(camera, rows, config.events.min_track_seconds)
        all_crops += crops
        sheets = write_sheets(camera.id, tiles) if tiles else 0
        print(f"{camera.id:<9} {len(tiles)} tracks, {len(crops)} crops, {sheets} sheets")

    with (CROPS_DIR / "index.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Crop)])
        writer.writeheader()
        writer.writerows(asdict(crop) for crop in all_crops)


if __name__ == "__main__":
    main()
