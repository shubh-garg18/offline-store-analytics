# Design notes

These are the decisions that shaped the pipeline, and why I made them.

## Tracking

**A fresh model per clip.** Ultralytics keeps the tracker inside the model object. Reusing one model across cameras would carry track IDs from the entrance camera into the shelf cameras.

**Tracker input at 0.10, not 0.25.** ByteTrack matches confident boxes first, then gives weaker boxes (0.1 to 0.25) a second chance to continue an existing track. That second pass is how it keeps hold of someone half hidden behind a shelf. My first version passed `conf=0.25` to the tracker, which threw those weak boxes away before ByteTrack ever saw them. Dropping it to 0.10 cut flicker tracks on the busiest camera from 65 to 57, and cut visits cut off by a broken track from 69% to 64%.

**Reproducible.** Tracking the same clip twice gives byte-identical output. That matters because the staff labels are tied to track IDs.

## Turning boxes into events

**Feet, not the box centre.** The bottom-centre of a person's box is where they touch the floor, and the floor is what the zones are drawn on. The box centre floats around with height and camera angle.

**Shapes drawn where people actually stand.** The zone polygons and the door line are drawn on real frames, with every tracked foot position dotted on top (`python -m src.preview --feet`). An earlier attempt traced zones from the floor plan, and because the cameras see the floor in perspective, the door line never fired.

**Three guards against noise:**
- a track has to last a second, because most shorter tracks are reflections and flickers;
- a change of side or zone has to hold for 0.33 s, because feet jitter by a few pixels every frame;
- an entry has to cross the door segment itself, not the line extended past it, because people walk along the street right beside the door.

**Cut-off visits.** If a visit is already under way when a track starts, or still going when it ends, its real start or end was never seen. Those visits are flagged and left out of average dwell, though they still count towards total time.

## Checking it against reality

Numbers from a pipeline like this mean nothing until they're checked against what actually happened, so ground truth was labelled from raw frames, never from model output.
- **Door.** Nobody actually walked in during the clip. That makes the useful test "does it stay at zero while 20 to 25 people pass outside", and I compare against deliberately careless door lines.
- **Shelves.** Every 2 seconds, the people at the Lakme wall and at the summer display were counted, and the pipeline's count is compared at the same moments. This catches errors that cancel out in a total.

## Fine-tuning the detector

COCO, which YOLO learned "person" from, is mostly eye-level photos. These cameras look down through glass at people in black, surrounded by posters and TV screens full of faces.
- **Teacher, then review.** A large model (yolo11x) drafted boxes on 464 frames, and every box was reviewed. Posters, screens and reflections lost their boxes, because those are exactly what the small model has to learn to ignore. Half-hidden people that the teacher scored too low were added back.
- **Split by time.** Neighbouring frames are near copies, so a random split would test on frames the model had effectively seen. The first 70% of each clip trains and the last 30% validates, with a gap between them.
- **Same model, new data.** yolo11n is trained further rather than swapped for a bigger model, so the pipeline keeps its CPU speed.

## Staff or customer

Staff wear a black uniform, and so do plenty of customers, so a colour rule was never going to work. Three approaches were compared on labelled tracks:
- **A rule:** feet behind the billing counter means staff. It only works on the billing camera.
- **Zero-shot CLIP:** compare each crop with sentences describing staff and customers. No training at all.
- **A linear probe:** logistic regression on frozen CLIP embeddings.

The probe won. CLIP itself isn't fine-tuned, because with around 40 staff examples it would just memorise those people. The probe keeps CLIP frozen and learns about 500 weights. Scores are cross-validated by track, so one person's crops never sit in both training and test, and the decision threshold is chosen without looking at the test fold.

## SQL and serving

- **SQLite, one file.** That's plenty for one store-day. Staff labels live in their own table and a `customer_events` view joins them in, so the raw events are never rewritten.
- **One place for the queries.** All metrics are SQL in `src/metrics.py`, and both the report and the API call the same functions, so they can't disagree.
- **Precompute, then serve.** The heavy work runs once, offline. The API only reads the database, which keeps the Docker image small (229 MB) and every request fast.
