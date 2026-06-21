"""Fine-tuning, step 4: apply the review's corrections to the teacher's boxes and rebuild the YOLO labels.

The labels are always rebuilt from teacher_boxes.csv plus corrections.csv, so the correction step can be
rerun and audited. corrections.csv columns: target, action, box, x1, y1, x2, y2, note.
  remove         target is an image; drop teacher box number `box`.
  accept         target is an image; candidate number `box` (a weak teacher guess) is a real person, keep it.
  move           target is an image; replace box `box` with the given corners.
  add            target is an image; add a person the teacher missed entirely.
  remove_region  target is a camera id; drop every teacher box whose centre lies in the region, in all of
                 that camera's images. For things that never move, like a poster or a reflection.
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from src.labels import Box, images_by_split, read_boxes, write_boxes, write_yolo_labels
from src.sample_frames import DATA_DIR, camera_of

TEACHER_BOXES = DATA_DIR / "teacher_boxes.csv"
CORRECTIONS = DATA_DIR / "corrections.csv"
LABELLED_BOXES = DATA_DIR / "labelled_boxes.csv"


@dataclass
class Correction:
    target: str
    action: str
    box: int | None
    region: tuple[float, float, float, float] | None
    note: str


def read_corrections(path: Path) -> list[Correction]:
    if not path.exists():
        return []
    corrections = []
    with path.open() as file:
        for row in csv.DictReader(file):
            corners = [row.get(key, "").strip() for key in ("x1", "y1", "x2", "y2")]
            corrections.append(Correction(
                target=row["target"].strip(),
                action=row["action"].strip(),
                box=int(row["box"]) if row.get("box", "").strip() else None,
                region=tuple(float(c) for c in corners) if all(corners) else None,
                note=row.get("note", "").strip(),
            ))
    return corrections


def inside(point: tuple[float, float], region: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = region
    return x1 <= x <= x2 and y1 <= y <= y2


def apply(proposals: list[Box], corrections: list[Correction], split_of: dict[str, str]) -> tuple[list[Box], Counter]:
    """Start from the teacher's labels (not its candidates) and apply every correction in order."""
    counts: Counter = Counter()
    boxes = [b for b in proposals if b.source == "teacher"]
    candidates = {(b.image, b.box): b for b in proposals if b.source == "candidate"}
    highest_number = {}
    for b in proposals:
        highest_number[b.image] = max(highest_number.get(b.image, 0), b.box)

    for c in (c for c in corrections if c.action == "remove_region"):
        kept = [b for b in boxes if not (camera_of(b.image) == c.target and inside(b.centre, c.region))]
        counts["removed"] += len(boxes) - len(kept)
        boxes = kept

    for c in corrections:
        if c.action == "remove":
            before = len(boxes)
            boxes = [b for b in boxes if not (b.image == c.target and b.box == c.box)]
            counts["removed"] += before - len(boxes)
        elif c.action == "accept":
            boxes.append(replace(candidates[(c.target, c.box)], source="review"))
            counts["accepted"] += 1
        elif c.action == "move":
            x1, y1, x2, y2 = c.region
            boxes = [replace(b, x1=x1, y1=y1, x2=x2, y2=y2, source="review")
                     if b.image == c.target and b.box == c.box else b for b in boxes]
            counts["moved"] += 1
        elif c.action == "add":
            x1, y1, x2, y2 = c.region
            highest_number[c.target] = highest_number.get(c.target, 0) + 1
            boxes.append(Box(c.target, split_of[c.target], highest_number[c.target], x1, y1, x2, y2, 1.0, "review"))
            counts["added"] += 1
        elif c.action != "remove_region":
            raise ValueError(f"unknown correction action '{c.action}'")
    return boxes, counts


def main() -> None:
    argparse.ArgumentParser(description="Apply review corrections to the teacher's boxes.").parse_args()
    splits = images_by_split(DATA_DIR / "images")
    split_of = {image: split for split, images in splits.items() for image in images}
    proposals = read_boxes(TEACHER_BOXES)
    boxes, counts = apply(proposals, read_corrections(CORRECTIONS), split_of)

    write_boxes(boxes, LABELLED_BOXES)
    write_yolo_labels(boxes, splits, DATA_DIR / "labels")
    teacher_labels = sum(b.source == "teacher" for b in proposals)
    print(f"teacher labels {teacher_labels}: removed {counts['removed']}, moved {counts['moved']}, "
          f"accepted {counts['accepted']} candidates, added {counts['added']} -> {len(boxes)} labelled boxes")


if __name__ == "__main__":
    main()
