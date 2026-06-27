"""Runs a training job on a free Kaggle GPU and brings the results back.

Uploads a folder as a private Kaggle dataset, pushes a private GPU notebook that copies this repo's src/
and config/ from that dataset and runs the given commands, waits, then downloads what the job wrote.
The code travels inside the upload (not a GitHub clone), so the GPU runs exactly the local code, including
changes that are not pushed yet. Only prepared training sets are uploaded, never full clips or the POS file.

The Kaggle CLI is run through uvx, so it is not a project dependency. Set KAGGLE_USERNAME to your Kaggle username;
the CLI reads its token from ~/.kaggle as usual.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2

KAGGLE = ["uvx", "-q", "kaggle@2.2.4"]
WORK_DIR = Path("outputs/kaggle")
GPU_SHAPE = "NvidiaTeslaT4"
POLL_SECONDS = 30
# Training runs at 640 px, so full-HD frames only make the upload slower. Labels are fractions of the
# frame, so they stay valid after resizing.
UPLOAD_MAX_WIDTH = 1280

# Runs on Kaggle. It finds the uploaded code and data wherever Kaggle mounted and unpacked them, copies
# both to /tmp (the input mount is read-only, and Ultralytics writes label caches next to the labels),
# installs the pinned packages and runs each command. {data} in a command becomes the data folder.
RUNNER_TEMPLATE = '''import glob, os, shutil, subprocess, sys

code = os.path.dirname(glob.glob("/kaggle/input/**/code/requirements.txt", recursive=True)[0])
images = [d for d in glob.glob("/kaggle/input/**/images/train", recursive=True)
          if os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(d)), "labels", "train"))][0]
shutil.copytree(code, "/tmp/project")
shutil.copytree(os.path.dirname(os.path.dirname(images)), "/tmp/data",
                ignore=shutil.ignore_patterns("code", "*.json"))
os.chdir("/tmp/project")
pins = [line.strip() for line in open("requirements.txt") if line.split("==")[0] in {packages!r}]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pins])
for command in {commands!r}:
    command = [part.replace("{{data}}", "/tmp/data") for part in command]
    print("running:", " ".join(command), flush=True)
    subprocess.check_call([sys.executable, "-m", *command])
'''


class KaggleUnavailable(RuntimeError):
    """Raised when the CLI, credentials, phone verification or GPU quota stops the job; fall back to CPU."""


def kaggle_username() -> str:
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        raise KaggleUnavailable("set KAGGLE_USERNAME to the Kaggle account that will run the job")
    return username


def kaggle(*args: str) -> str:
    result = subprocess.run([*KAGGLE, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise KaggleUnavailable(f"kaggle {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def bundle_code(bundle_dir: Path) -> None:
    """Copy the code the job needs next to the data, so the GPU runs exactly the local, uncommitted code."""
    code_dir = bundle_dir / "code"
    if code_dir.exists():
        shutil.rmtree(code_dir)
    shutil.copytree("src", code_dir / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree("config", code_dir / "config")
    shutil.copy("requirements.txt", code_dir / "requirements.txt")


def upload_dataset(bundle_dir: Path, slug: str, title: str) -> None:
    (bundle_dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{kaggle_username()}/{slug}", "licenses": [{"name": "other"}]}, indent=2))
    try:
        kaggle("datasets", "status", f"{kaggle_username()}/{slug}")
        exists = True
    except KaggleUnavailable:
        exists = False
    if exists:
        kaggle("datasets", "version", "-p", str(bundle_dir), "-m", "updated training set", "-r", "zip", "-q")
    else:
        kaggle("datasets", "create", "-p", str(bundle_dir), "-r", "zip", "-q")  # private unless --public
    wait_for_dataset(slug)


def wait_for_dataset(slug: str, timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if "ready" in kaggle("datasets", "status", f"{kaggle_username()}/{slug}").lower():
                return
        except KaggleUnavailable:
            pass  # a just-created dataset answers 403 for a short while before it exists for the API
        time.sleep(15)
    raise KaggleUnavailable(f"dataset {slug} was not ready after {timeout_seconds} s")


def push_kernel(kernel_slug: str, dataset_slug: str, commands: list[list[str]], packages: set[str]) -> Path:
    kernel_dir = WORK_DIR / kernel_slug
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "runner.py").write_text(RUNNER_TEMPLATE.format(commands=commands, packages=sorted(packages)))
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{kaggle_username()}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{kaggle_username()}/{dataset_slug}"],
        "kernel_sources": [],
        "competition_sources": [],
        "machine_shape": GPU_SHAPE,
    }, indent=2))
    kaggle("kernels", "push", "-p", str(kernel_dir))
    return kernel_dir


def wait_for_kernel(kernel_slug: str, timeout_seconds: int = 4 * 3600) -> str:
    deadline = time.time() + timeout_seconds
    status = ""
    while time.time() < deadline:
        status = kaggle("kernels", "status", f"{kaggle_username()}/{kernel_slug}").lower()
        if "complete" in status:
            return "complete"
        if "error" in status or "cancel" in status:
            return status.strip()
        time.sleep(POLL_SECONDS)
    raise KaggleUnavailable(f"kernel {kernel_slug} still running after {timeout_seconds} s: {status}")


def download_outputs(kernel_slug: str) -> Path:
    out_dir = WORK_DIR / kernel_slug / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{kaggle_username()}/{kernel_slug}", "-p", str(out_dir), "-o", "-q")
    return out_dir


def run_job(bundle_dir: Path, dataset_slug: str, kernel_slug: str, commands: list[list[str]],
            packages: set[str]) -> Path:
    """Upload the bundle, run the commands on a GPU, and return the folder holding the job's outputs."""
    bundle_code(bundle_dir)
    print(f"uploading {bundle_dir} as private dataset {dataset_slug} ...", flush=True)
    upload_dataset(bundle_dir, dataset_slug, dataset_slug)
    print(f"pushing private GPU job {kernel_slug} ...", flush=True)
    push_kernel(kernel_slug, dataset_slug, commands, packages)
    status = wait_for_kernel(kernel_slug)
    out_dir = download_outputs(kernel_slug)
    if status != "complete":
        raise RuntimeError(f"kernel {kernel_slug} ended with {status}; log in {out_dir}")
    return out_dir


def copy_images_downsized(source: Path, target: Path) -> None:
    for image_path in source.rglob("*.jpg"):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        if width > UPLOAD_MAX_WIDTH:
            image = cv2.resize(image, (UPLOAD_MAX_WIDTH, round(height * UPLOAD_MAX_WIDTH / width)),
                               interpolation=cv2.INTER_AREA)
        out_path = target / image_path.relative_to(source)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def prepare_finetune_bundle() -> Path:
    """Sampled frames, their corrected labels and the code: everything the GPU needs, nothing more."""
    bundle_dir = WORK_DIR / "finetune_bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    copy_images_downsized(Path("data/finetune/images"), bundle_dir / "images")
    shutil.copytree(Path("data/finetune/labels"), bundle_dir / "labels", ignore=shutil.ignore_patterns("*.cache"))
    bundle_code(bundle_dir)
    return bundle_dir


def finetune_job() -> None:
    bundle_dir = prepare_finetune_bundle()
    commands = [["src.finetune", "--data-dir", "{data}", "--out", "/kaggle/working/models"]]
    out_dir = run_job(bundle_dir, "cctv-store-finetune", "cctv-store-finetune-run", commands, {"ultralytics"})

    models_dir = Path("outputs/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.rglob("*"):
        if path.suffix in {".pt", ".csv", ".png"}:
            shutil.copy(path, models_dir / path.name)
            print(f"downloaded {models_dir / path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the detector on a Kaggle GPU.")
    parser.add_argument("--prepare-only", action="store_true", help="build the upload folder and stop (Colab fallback)")
    args = parser.parse_args()
    if args.prepare_only:
        print(f"bundle ready in {prepare_finetune_bundle()}")
        return
    finetune_job()


if __name__ == "__main__":
    main()
