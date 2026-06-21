"""The box table shared by the teacher, the review sheets and the corrections, and the YOLO label files.

Coordinates are fractions of the frame (0 to 1), so they can be read straight off the 0.1 grid on a sheet.
"""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Box:
    image: str
    split: str
    box: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str  # "teacher" (a label), "candidate" (a weak teacher guess, not a label) or "review"

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def iou(a: Box, b: Box) -> float:
    """Intersection over union: how much two boxes overlap, from 0 (apart) to 1 (identical)."""
    overlap_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    overlap_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    overlap = overlap_w * overlap_h
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - overlap
    return overlap / union if union else 0.0


def write_boxes(boxes: list[Box], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Box)])
        writer.writeheader()
        writer.writerows(asdict(box) for box in boxes)


def read_boxes(path: Path) -> list[Box]:
    with path.open() as file:
        return [
            Box(image=row["image"], split=row["split"], box=int(row["box"]),
                x1=float(row["x1"]), y1=float(row["y1"]), x2=float(row["x2"]), y2=float(row["y2"]),
                confidence=float(row["confidence"]), source=row["source"])
            for row in csv.DictReader(file)
        ]


def yolo_line(box: Box) -> str:
    """YOLO format: class, then box centre and size as fractions of the frame."""
    width, height = box.x2 - box.x1, box.y2 - box.y1
    x_centre, y_centre = box.x1 + width / 2, box.y1 + height / 2
    return f"0 {x_centre:.6f} {y_centre:.6f} {width:.6f} {height:.6f}"


def write_yolo_labels(boxes: list[Box], images_by_split: dict[str, list[str]], labels_dir: Path) -> None:
    """One .txt per image. An image with no people gets an empty file: it teaches the model what is not a person."""
    by_image: dict[str, list[Box]] = {}
    for box in boxes:
        by_image.setdefault(box.image, []).append(box)
    for split, images in images_by_split.items():
        split_dir = labels_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            lines = [yolo_line(box) for box in sorted(by_image.get(image, []), key=lambda b: b.box)]
            (split_dir / Path(image).with_suffix(".txt").name).write_text("\n".join(lines) + ("\n" if lines else ""))


def images_by_split(images_dir: Path) -> dict[str, list[str]]:
    return {split.name: sorted(p.name for p in split.glob("*.jpg")) for split in sorted(images_dir.iterdir()) if split.is_dir()}
