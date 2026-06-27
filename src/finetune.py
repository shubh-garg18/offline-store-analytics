"""Fine-tuning, step 5: fine-tune the small pretrained detector on this store's labelled frames.

Runs unchanged on a GPU (Kaggle) or a CPU. Writes outputs/models/store_yolo11n.pt, the training history
and the training curves.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

from src.config import FinetuneSettings, load_config
from src.sample_frames import DATA_DIR, write_dataset_yaml

MODELS_DIR = Path("outputs/models")


def pick_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def train(settings: FinetuneSettings, data_dir: Path, out_dir: Path, epochs: int) -> Path:
    device = pick_device()
    work_dir = Path(tempfile.mkdtemp(prefix="finetune_"))
    # Ultralytics finds each image's labels by swapping images/ for labels/ in its path.
    dataset_yaml = write_dataset_yaml(data_dir, "images/train", "images/val", work_dir / "dataset.yaml")
    run_name = "store"

    model = YOLO(settings.student_model)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        patience=settings.patience,
        imgsz=settings.imgsz,
        batch=settings.batch if device != "cpu" else min(settings.batch, 8),
        device=device,
        project=str(work_dir / "runs"),
        name=run_name,
        seed=0,
        deterministic=True,
        workers=2,
        # Augmentation for ceiling CCTV: vary brightness and colour, mirror left-right, never flip upside
        # down (people are never upside down) and never rotate (the cameras are fixed).
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        mosaic=1.0,
        plots=True,
        verbose=False,
    )

    run_dir = work_dir / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "store_yolo11n.pt"
    shutil.copy(run_dir / "weights" / "best.pt", weights)
    shutil.copy(run_dir / "results.csv", out_dir / "training_history.csv")
    if (run_dir / "results.png").exists():
        shutil.copy(run_dir / "results.png", out_dir / "training_curves.png")
    return weights


def main() -> None:
    settings = load_config().finetune
    parser = argparse.ArgumentParser(description="Fine-tune yolo11n on the labelled store frames.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    weights = train(settings, args.data_dir, args.out, args.epochs)
    print(f"saved {weights}")


if __name__ == "__main__":
    main()
