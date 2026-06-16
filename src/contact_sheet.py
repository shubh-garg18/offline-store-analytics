"""Builds contact sheets of raw frames (no model output) for labelling ground truth by eye.

Each tile shows one frame, cropped to a region of the camera, with the configured zones outlined
and the clip time printed, so arrivals and departures can be read off at a glance.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.preview import ZONE_COLOUR, draw_polygon


def read_frames_every(clip_path: Path, step_seconds: float) -> list[tuple[float, np.ndarray]]:
    capture = cv2.VideoCapture(str(clip_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_index = 0.0
    while frame_index < total:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            break
        frames.append((int(frame_index) / fps, frame))
        frame_index += step_seconds * fps
    capture.release()
    return frames


def tile(frame: np.ndarray, seconds: float, zones: dict, crop: tuple[float, float, float, float], width: int) -> np.ndarray:
    frame = frame.copy()
    for name, polygon in zones.items():
        draw_polygon(frame, polygon, ZONE_COLOUR, name)
    height, full_width = frame.shape[:2]
    x0, y0, x1, y1 = int(crop[0] * full_width), int(crop[1] * height), int(crop[2] * full_width), int(crop[3] * height)
    region = frame[y0:y1, x0:x1]
    scaled = cv2.resize(region, (width, int(region.shape[0] * width / region.shape[1])))
    cv2.rectangle(scaled, (0, 0), (150, 40), (0, 0, 0), -1)
    cv2.putText(scaled, f"{seconds:6.1f}s", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return scaled


def build_sheets(tiles: list[np.ndarray], columns: int, rows: int) -> list[np.ndarray]:
    per_sheet = columns * rows
    blank = np.zeros_like(tiles[0])
    sheets = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [blank] * (per_sheet - len(chunk))
        grid_rows = [np.hstack(chunk[r * columns:(r + 1) * columns]) for r in range(rows)]
        sheets.append(np.vstack(grid_rows))
    return sheets


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Contact sheets of raw frames for labelling.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--step", type=float, default=1.0, help="seconds between tiles")
    parser.add_argument("--crop", type=float, nargs=4, default=[0.0, 0.0, 1.0, 1.0],
                        metavar=("X0", "Y0", "X1", "Y1"), help="region to keep, in fractions of the frame")
    parser.add_argument("--from-seconds", type=float, default=0.0)
    parser.add_argument("--to-seconds", type=float, default=1e9)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args()

    camera = config.camera(args.camera)
    frames = [(s, f) for s, f in read_frames_every(camera.clip_path, args.step)
              if args.from_seconds <= s <= args.to_seconds]
    tiles = [tile(frame, seconds, camera.zones, tuple(args.crop), args.tile_width) for seconds, frame in frames]

    out_dir = config.outputs_dir / "labelling" / camera.id
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, sheet in enumerate(build_sheets(tiles, args.columns, args.rows)):
        path = out_dir / f"sheet_{args.from_seconds:05.1f}_{index:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {len(tiles)} tiles to {out_dir}")


if __name__ == "__main__":
    main()
