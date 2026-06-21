"""Fine-tuning, step 3: contact sheets of the sampled frames with the teacher's boxes numbered on them.

Green boxes are labels, thin orange boxes are candidates (weak guesses), magenta boxes came from the review.
The reviewer reads the sheets and writes data/finetune/corrections.csv. With --corrected the sheets show the
labels after corrections instead, to check the result. With --crops, every box still in doubt (weak labels, and
candidates that are not on a removed region or a copy of a label) is shown as its own numbered crop, which is
faster to judge than a whole frame. Writes outputs/finetune_review/.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.apply_corrections import CORRECTIONS, TEACHER_BOXES, apply, inside, read_corrections
from src.labels import Box, images_by_split, iou, read_boxes
from src.sample_frames import DATA_DIR, camera_of

OUT_DIR = Path("outputs/finetune_review")
TEACHER_COLOUR = (0, 255, 0)
CANDIDATE_COLOUR = (0, 165, 255)
REVIEW_COLOUR = (255, 0, 255)
GRID_COLOUR = (110, 110, 110)
WEAK_LABEL = 0.5
DUPLICATE_IOU = 0.4
CROP_TILE = (200, 300)


def draw_tile(image_path: Path, boxes: list[Box], width: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    height_px, width_px = image.shape[:2]
    for step in range(1, 10):
        x, y = int(width_px * step / 10), int(height_px * step / 10)
        cv2.line(image, (x, 0), (x, height_px), GRID_COLOUR, 1)
        cv2.line(image, (0, y), (width_px, y), GRID_COLOUR, 1)
    for box in boxes:
        colour = {"teacher": TEACHER_COLOUR, "candidate": CANDIDATE_COLOUR}.get(box.source, REVIEW_COLOUR)
        top_left = (int(box.x1 * width_px), int(box.y1 * height_px))
        bottom_right = (int(box.x2 * width_px), int(box.y2 * height_px))
        cv2.rectangle(image, top_left, bottom_right, colour, 2 if box.source == "candidate" else 4)
        label = f"{box.box} {box.confidence:.2f}"
        cv2.rectangle(image, top_left, (top_left[0] + 150, top_left[1] + 44), (0, 0, 0), -1)
        cv2.putText(image, label, (top_left[0] + 4, top_left[1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    tile = cv2.resize(image, (width, int(height_px * width / width_px)))
    cv2.rectangle(tile, (0, tile.shape[0] - 30), (330, tile.shape[0]), (0, 0, 0), -1)
    cv2.putText(tile, image_path.name, (6, tile.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return tile


def sheets_for_camera(camera_id: str, boxes: list[Box], columns: int, rows: int, width: int) -> list[np.ndarray]:
    images = sorted(
        (split.name, image.name)
        for split in (DATA_DIR / "images").iterdir()
        for image in split.glob("*.jpg")
        if camera_of(image.name) == camera_id
    )
    by_image: dict[str, list[Box]] = {}
    for box in boxes:
        by_image.setdefault(box.image, []).append(box)

    tiles = [draw_tile(DATA_DIR / "images" / split / name, by_image.get(name, []), width) for split, name in images]
    per_sheet = columns * rows
    sheets = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [np.zeros_like(tiles[0])] * (per_sheet - len(chunk))
        sheets.append(np.vstack([np.hstack(chunk[r * columns:(r + 1) * columns]) for r in range(rows)]))
    return sheets


def boxes_in_doubt() -> list[Box]:
    """Labels the teacher was unsure of, plus candidates that might be people it under-scored."""
    splits = images_by_split(DATA_DIR / "images")
    split_of = {image: split for split, images in splits.items() for image in images}
    proposals = read_boxes(TEACHER_BOXES)
    corrections = read_corrections(CORRECTIONS)
    labels, _ = apply(proposals, corrections, split_of)
    regions = [c for c in corrections if c.action == "remove_region"]
    labels_by_image: dict[str, list[Box]] = {}
    for label in labels:
        labels_by_image.setdefault(label.image, []).append(label)

    weak_labels = [b for b in labels if b.source == "teacher" and b.confidence < WEAK_LABEL]
    open_candidates = [
        b for b in proposals
        if b.source == "candidate"
        and not any(camera_of(b.image) == r.target and inside(b.centre, r.region) for r in regions)
        and not any(iou(b, label) > DUPLICATE_IOU for label in labels_by_image.get(b.image, []))
    ]
    return sorted(weak_labels + open_candidates, key=lambda b: (b.image, b.box))


def crop_tile(box: Box, number: int) -> np.ndarray:
    image = cv2.imread(str(DATA_DIR / "images" / box.split / box.image))
    height, width = image.shape[:2]
    pad_x, pad_y = (box.x2 - box.x1) * 0.3, (box.y2 - box.y1) * 0.15
    x1, x2 = int(max(0, box.x1 - pad_x) * width), int(min(1, box.x2 + pad_x) * width)
    y1, y2 = int(max(0, box.y1 - pad_y) * height), int(min(1, box.y2 + pad_y) * height)
    crop = image[y1:y2, x1:x2].copy()
    colour = TEACHER_COLOUR if box.source == "teacher" else CANDIDATE_COLOUR
    cv2.rectangle(crop, (int(box.x1 * width) - x1, int(box.y1 * height) - y1),
                  (int(box.x2 * width) - x1, int(box.y2 * height) - y1), colour, 2)
    tile = np.zeros((CROP_TILE[1], CROP_TILE[0], 3), np.uint8)
    scale = min(CROP_TILE[0] / crop.shape[1], (CROP_TILE[1] - 26) / crop.shape[0])
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    tile[26:26 + resized.shape[0], :resized.shape[1]] = resized
    cv2.putText(tile, f"{number} {box.confidence:.2f}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return tile


def write_crop_sheets(columns: int = 10, rows: int = 4) -> None:
    doubtful = boxes_in_doubt()
    out_dir = OUT_DIR / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["number", "image", "box", "source", "confidence"])
        for number, box in enumerate(doubtful):
            writer.writerow([number, box.image, box.box, box.source, box.confidence])
    tiles = [crop_tile(box, number) for number, box in enumerate(doubtful)]
    per_sheet = columns * rows
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [np.zeros_like(tiles[0])] * (per_sheet - len(chunk))
        sheet = np.vstack([np.hstack(chunk[r * columns:(r + 1) * columns]) for r in range(rows)])
        cv2.imwrite(str(out_dir / f"crops_{start // per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {len(tiles)} crops in doubt to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Numbered teacher boxes on contact sheets, for review.")
    parser.add_argument("--corrected", action="store_true", help="show the labels after corrections")
    parser.add_argument("--crops", action="store_true", help="one numbered crop per box still in doubt")
    parser.add_argument("--camera", action="append", help="camera id (repeatable); default: all")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args()
    if args.crops:
        write_crop_sheets()
        return

    source = DATA_DIR / ("labelled_boxes.csv" if args.corrected else "teacher_boxes.csv")
    boxes = read_boxes(source)
    cameras = args.camera or sorted({camera_of(p.name) for p in (DATA_DIR / "images").rglob("*.jpg")})
    out_dir = OUT_DIR / ("corrected" if args.corrected else "teacher")
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera_id in cameras:
        for index, sheet in enumerate(sheets_for_camera(camera_id, boxes, args.columns, args.rows, args.tile_width)):
            cv2.imwrite(str(out_dir / f"{camera_id}_{index:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote sheets to {out_dir}")


if __name__ == "__main__":
    main()
