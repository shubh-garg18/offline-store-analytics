"""Draws each camera's configured door line, zones and areas on a real frame, to check calibration by eye.

Writes outputs/frames/<camera>.png. A light 0.1 grid helps read coordinates for store.yaml.
"""

import argparse
import csv
from pathlib import Path

import cv2

from src.config import Camera, load_config
from src.geometry import foot_point

GRID_COLOUR = (90, 90, 90)
DOOR_COLOUR = (0, 0, 255)
ZONE_COLOUR = (0, 200, 255)
QUEUE_COLOUR = (0, 255, 0)
STAFF_COLOUR = (255, 0, 255)
FEET_COLOUR = (255, 255, 0)


def read_frame(camera: Camera, at_fraction: float):
    capture = cv2.VideoCapture(str(camera.clip_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * at_fraction))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {camera.clip_path}")
    return frame


def draw_grid(frame) -> None:
    height, width = frame.shape[:2]
    for step in range(1, 10):
        x, y = int(width * step / 10), int(height * step / 10)
        cv2.line(frame, (x, 0), (x, height), GRID_COLOUR, 1)
        cv2.line(frame, (0, y), (width, y), GRID_COLOUR, 1)
        cv2.putText(frame, f"{step / 10:.1f}", (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"{step / 10:.1f}", (4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_polygon(frame, polygon, colour, label: str) -> None:
    points = [(int(x), int(y)) for x, y in polygon]
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], colour, 3)
    cv2.putText(frame, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)


def draw_geometry(frame, camera: Camera) -> None:
    if camera.door_line:
        a, b = camera.door_line
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), DOOR_COLOUR, 4)
        cv2.putText(frame, "door", (int(a[0]), int(a[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DOOR_COLOUR, 2)
    if camera.inside_point:
        x, y = camera.inside_point
        cv2.circle(frame, (int(x), int(y)), 12, DOOR_COLOUR, -1)
        cv2.putText(frame, "inside", (int(x) + 16, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DOOR_COLOUR, 2)
    for name, polygon in camera.zones.items():
        draw_polygon(frame, polygon, ZONE_COLOUR, name)
    if camera.queue_area:
        draw_polygon(frame, camera.queue_area, QUEUE_COLOUR, "queue")
    if camera.staff_area:
        draw_polygon(frame, camera.staff_area, STAFF_COLOUR, "staff")


def draw_feet(frame, tracks_csv: Path, min_track_seconds: float) -> None:
    """Dot every foot position of real tracks, so zones can be drawn where people actually stand."""
    feet_by_track: dict[int, list] = {}
    with tracks_csv.open() as file:
        for row in csv.DictReader(file):
            feet = foot_point(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            feet_by_track.setdefault(int(row["track_id"]), []).append((float(row["seconds"]), feet))
    for samples in feet_by_track.values():
        if samples[-1][0] - samples[0][0] < min_track_seconds:
            continue
        for _, (x, y) in samples[::5]:
            cv2.circle(frame, (int(x), int(y)), 3, FEET_COLOUR, -1)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Draw configured geometry on a frame of each clip.")
    parser.add_argument("--camera", action="append", help="camera id (repeatable); default: all")
    parser.add_argument("--at", type=float, default=0.5, help="where in the clip to take the frame (0 to 1)")
    parser.add_argument("--feet", action="store_true", help="also dot the foot positions from outputs/tracks/")
    args = parser.parse_args()

    out_dir = config.outputs_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera_id in args.camera or [camera.id for camera in config.cameras]:
        camera = config.camera(camera_id)
        frame = read_frame(camera, args.at)
        draw_grid(frame)
        if args.feet:
            draw_feet(frame, config.outputs_dir / "tracks" / f"{camera.id}.csv", config.events.min_track_seconds)
        draw_geometry(frame, camera)
        path = out_dir / f"{camera.id}.png"
        cv2.imwrite(str(path), frame)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
