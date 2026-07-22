"""One command for the whole analysis: tracks -> events -> evaluations -> crops -> staff -> database -> report.

Each step is the same module you can run on its own (python -m src.<step>); this file only runs them in order
and stops at the first failure. Fine-tuning the detector (src/finetune.py) is a separate, one-off job.
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("track", "find and track people in every clip"),
    ("events", "turn tracks into door, zone and queue events"),
    ("evaluate", "score door crossings against the ground truth"),
    ("evaluate_zones", "score zone occupancy against the ground truth"),
    ("crops", "cut out every tracked person"),
    ("staff", "label tracks as staff or not"),
    ("database", "load everything into outputs/store.db"),
    ("report", "write outputs/report.md"),
]


def run_step(module: str, description: str) -> None:
    print(f"==> {module}: {description}", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", f"src.{module}"], check=True)
    print(f"    done in {time.perf_counter() - started:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole analysis pipeline.")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="reuse outputs/tracks/ (tracking is the slow step, about 20 minutes on a CPU)")
    args = parser.parse_args()

    for module, description in STEPS:
        if module == "track" and args.skip_tracking:
            print("==> track: skipped, reusing outputs/tracks/")
            continue
        try:
            run_step(module, description)
        except subprocess.CalledProcessError:
            sys.exit(f"pipeline stopped: step '{module}' failed")


if __name__ == "__main__":
    main()
