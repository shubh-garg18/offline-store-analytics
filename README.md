# Offline Store Analytics

Turning a store's CCTV into the numbers a store manager would ask for: how many people came in, which shelves
held their attention, and how the billing queue behaved.

Footage from four ceiling cameras in a beauty store (entrance, two shelf areas, billing counter). The footage
isn't in this repo; put the clips in `data/clips/`.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

## Tracking

```bash
python -m src.track --save-video
```

YOLO11 nano finds people in every frame and ByteTrack keeps one ID per person. Each camera gets
`outputs/tracks/<camera>.csv` (one row per person per frame) and an annotated video.
