# Offline Store Analytics

An online store knows exactly what every visitor looked at and where they gave up. A physical store usually knows how much it sold and not much else. I wanted to see how much of the online picture you can get back from the CCTV a store already has.

This repo takes four ceiling-camera clips from a beauty store (Purplle, Brigade Road, Bangalore) plus that day's POS bills, and turns them into the numbers a store manager would ask for: how many people came in, which shelves held their attention, how the billing queue behaved, and when the money actually came in.

## What it found

- **The evening carries the day.** 6 PM to 8 PM brought in 38% of the day's revenue from a third of the bills, and the top 20% of bills made up 56% of revenue. The average basket is 1.7x the median, so a few big baskets do a lot of the work.
- **The Lakme wall pulls people in.** Shoppers spent about twice as long at the Lakme wall as at the summer display next to it (2.0x in labelled frames). The pipeline measures 1.6x, and 1.4x before the detector was fine-tuned. It undercounts because the display hides the feet of people standing at the wall behind it.
- **A store-specific detector matters.** Fine-tuning YOLO11 nano on 328 frames from these cameras lifted person recall from 0.69 to 0.87 (mAP50-95 0.59 to 0.77) at the same speed.
- **Door counting holds up against street traffic.** About 20 to 25 people walked past the glass frontage and one stood at the door on his phone; the door line counted none of them as entries. A carelessly drawn line counted 36.
- **Conversion can't be measured on this footage, and the report says so.** The clips cover about three minutes around 8:10 PM and no bill falls in that window, so conversion is reported as undefined rather than as a misleading 0%.

The full numbers, and how each one was checked, are in [docs/results.md](docs/results.md).

## How it works

```text
clips --> track people --> door, zone and queue events --> staff or customer --> SQLite --> report and API
          (YOLO11n +         (geometry on foot points)        (CLIP embeddings +      (SQL       (FastAPI,
           ByteTrack)                                           logistic regression)   metrics)   Docker)
```

1. **Track.** YOLO11 nano finds people in every frame and ByteTrack keeps one ID per person across frames.
2. **Events.** Each person's position is their feet (the bottom-centre of their box). Crossing the door line is an entry or exit, standing inside a shelf area is a zone visit with a dwell time, and standing in the queue area is a queue visit.
3. **Staff or customer.** Staff and many customers both wear black, so colour is useless. A small classifier on top of CLIP image embeddings decides instead.
4. **SQL.** Events, staff labels and bills go into SQLite, and every metric is a SQL query: footfall, zone attention, queue depth, a time-window join from queue to bill for conversion, and window functions for the sales patterns.
5. **Report and API.** The same queries feed a markdown report and a small read-only FastAPI service.

Why it's built this way is in [docs/design.md](docs/design.md).

## Running it

Python 3.11 on a CPU is enough.

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-dev.txt
python -m pytest                          # 39 tests, no footage needed
```

With the clips in `data/clips/` and the bills in `data/pos/` (paths, camera clocks and all zone shapes live in `config/store.yaml`):

```bash
python -m src.pipeline                    # video to report, about 25 minutes on a laptop CPU
python -m src.pipeline --skip-tracking    # reuse existing tracks, about 3 minutes
```

Every step also runs on its own, for example `python -m src.track --save-video` for annotated videos, or `python -m src.preview --feet` to draw the zones over the spots where people actually stood.

The API reads the finished database, so it needs neither PyTorch nor OpenCV:

```bash
docker build -t offline-store-analytics .
docker run -p 8000:8000 -v "$PWD/outputs:/app/outputs:ro" offline-store-analytics
```

It serves `/health`, `/metrics`, `/zones`, `/queue`, `/funnel` and `/bills`, with docs at `/docs`.

Fine-tuning the detector is a separate one-off job. `python -m src.kaggle_job` runs it on a free Kaggle GPU (set `KAGGLE_USERNAME`), and `python -m src.finetune` runs it locally.

## Layout

| Path | What's there |
| --- | --- |
| `src/track.py`, `src/events.py`, `src/geometry.py` | Tracking and the door, zone and queue logic |
| `src/evaluate.py`, `src/evaluate_zones.py` | Checking door counts and zone occupancy against labelled frames |
| `src/sample_frames.py` to `src/compare_detectors.py` | Building the training set, fine-tuning, and comparing detectors |
| `src/crops.py`, `src/staff.py` | Staff vs customer |
| `src/database.py`, `src/metrics.py`, `src/report.py` | SQLite, the SQL metrics and the report |
| `src/pipeline.py`, `src/api.py` | One-command run, and the API |
| `config/store.yaml` | Cameras, clocks, zone shapes and settings |
| `docs/` | Design notes, results and limitations |

## Data

The footage and the POS file aren't in this repo. They belong to the retailer and were shared for this analysis only, so frames, crops, labels and trained weights stay out too. The POS file contains customer names and phone numbers; the code only ever reads the bill id, date, time, store and amount.
