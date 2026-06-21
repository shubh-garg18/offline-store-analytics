"""Fine-tuning, step 2: a big, slow, more accurate teacher model proposes person boxes on every sampled frame.

Writes data/finetune/teacher_boxes.csv (the teacher's raw output, never edited by hand), the YOLO label
files built from it, and data/finetune/dataset.yaml. The review corrects these labels in the next step.
"""

import argparse
import time

from ultralytics import YOLO

from src.config import FinetuneSettings, load_config
from src.labels import Box, images_by_split, write_boxes, write_yolo_labels
from src.sample_frames import DATA_DIR, write_dataset_yaml

PERSON_CLASS = 0
TEACHER_BOXES = DATA_DIR / "teacher_boxes.csv"


def label_split(model: YOLO, split: str, images: list[str], settings: FinetuneSettings) -> list[Box]:
    """Boxes at or above teacher_confidence become labels; weaker ones down to the candidate threshold
    are kept as candidates, so the review can accept a half-hidden person instead of drawing a box."""
    boxes = []
    for image in images:
        result = model.predict(str(DATA_DIR / "images" / split / image), classes=[PERSON_CLASS],
                               conf=settings.teacher_candidate_confidence, imgsz=settings.teacher_imgsz,
                               verbose=False)[0]
        ordered = sorted(zip(result.boxes.xyxyn.tolist(), result.boxes.conf.tolist()), key=lambda pair: pair[0][0])
        for number, ((x1, y1, x2, y2), score) in enumerate(ordered, start=1):
            source = "teacher" if score >= settings.teacher_confidence else "candidate"
            boxes.append(Box(image, split, number, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4),
                             round(score, 3), source))
    return boxes


def main() -> None:
    argparse.ArgumentParser(description="Teacher model proposes boxes on the sampled frames.").parse_args()
    settings = load_config().finetune
    model = YOLO(settings.teacher_model)
    splits = images_by_split(DATA_DIR / "images")

    boxes = []
    for split, images in splits.items():
        started = time.perf_counter()
        boxes += label_split(model, split, images, settings)
        per_image = (time.perf_counter() - started) / max(len(images), 1)
        print(f"{split:<6} {len(images)} images, {per_image:.2f} s per image on this machine")

    write_boxes(boxes, TEACHER_BOXES)
    labels = [box for box in boxes if box.source == "teacher"]
    write_yolo_labels(labels, splits, DATA_DIR / "labels")
    write_dataset_yaml(DATA_DIR, "images/train", "images/val", DATA_DIR / "dataset.yaml")
    print(f"teacher proposed {len(labels)} labels and {len(boxes) - len(labels)} candidates -> {TEACHER_BOXES}")


if __name__ == "__main__":
    main()
