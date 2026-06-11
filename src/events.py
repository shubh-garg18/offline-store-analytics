"""Step 2 of the pipeline: turn each person's foot path into business events.

Entry camera: ENTRY / EXIT through the door. Zone cameras: ZONE_ENTER / ZONE_EXIT with dwell time.
Billing camera: QUEUE_JOIN / QUEUE_LEAVE, plus how many people are in the queue each second.
Writes outputs/events.csv and outputs/queue_depth.csv.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Hashable

from src.config import Camera, EventSettings, StoreConfig, load_config
from src.geometry import Point, crossing_direction, foot_point, point_in_polygon, side_of_line
from src.track_table import TrackRow, read_tracks_csv


@dataclass
class Sample:
    seconds: float
    frame: int
    foot: Point
    confidence: float


@dataclass
class Visit:
    """A stretch of samples where the person stayed in one zone (or the queue)."""

    zone: str
    start: int
    end: int
    censored: bool


@dataclass
class Event:
    event_id: str
    camera: str
    track_id: int
    event_type: str
    seconds: float
    time: str
    zone: str
    dwell_seconds: float | None
    confidence: float
    censored: bool


@dataclass
class QueueDepth:
    camera: str
    second: int
    time: str
    depth: int


def group_by_track(rows: list[TrackRow]) -> dict[int, list[Sample]]:
    tracks: dict[int, list[Sample]] = defaultdict(list)
    for row in rows:
        foot = foot_point(row.x1, row.y1, row.x2, row.y2)
        tracks[row.track_id].append(Sample(row.seconds, row.frame, foot, row.confidence))
    for samples in tracks.values():
        samples.sort(key=lambda sample: sample.frame)
    return dict(tracks)


def drop_short_tracks(tracks: dict[int, list[Sample]], min_seconds: float) -> dict[int, list[Sample]]:
    """Most sub-second tracks are flickers (reflections, posters, broken IDs), not people."""
    return {
        track_id: samples
        for track_id, samples in tracks.items()
        if samples[-1].seconds - samples[0].seconds >= min_seconds
    }


def debounced_changes(
    samples: list[Sample], state_of: Callable[[Sample], Hashable], debounce_seconds: float
) -> list[tuple[int, Hashable]]:
    """(sample index, state) for the starting state and every change that lasted at least `debounce_seconds`.

    Feet jitter by a few pixels every frame, so a person standing on a boundary would otherwise
    flip in and out many times a second.
    """
    stable = state_of(samples[0])
    changes = [(0, stable)]
    candidate, candidate_start = None, 0
    for index in range(1, len(samples)):
        state = state_of(samples[index])
        if state == stable:
            candidate = None
            continue
        if state != candidate:
            candidate, candidate_start = state, index
        if samples[index].seconds - samples[candidate_start].seconds >= debounce_seconds:
            stable = candidate
            changes.append((candidate_start, stable))
            candidate = None
    return changes


def visits_from_changes(changes: list[tuple[int, Hashable]], last_index: int) -> list[Visit]:
    """Runs of a non-empty state. A run touching the start or end of the track is censored:
    we never saw the person arrive, or never saw them leave, so its dwell is a lower bound."""
    visits = []
    for position, (start, state) in enumerate(changes):
        if not state:
            continue
        is_last_run = position == len(changes) - 1
        # The visit ends at the first sample seen outside the zone, the moment the person is seen leaving.
        end = last_index if is_last_run else changes[position + 1][0]
        visits.append(Visit(zone=str(state), start=start, end=end, censored=start == 0 or is_last_run))
    return visits


def make_event(camera: Camera, track_id: int, event_type: str, sample: Sample, zone: str = "",
               dwell_seconds: float | None = None, censored: bool = False) -> Event:
    return Event(
        event_id=f"{camera.id}-{track_id}-{sample.frame}-{event_type}",
        camera=camera.id,
        track_id=track_id,
        event_type=event_type,
        seconds=sample.seconds,
        time=camera.clock_at(sample.seconds).isoformat(sep=" ", timespec="milliseconds"),
        zone=zone,
        dwell_seconds=round(dwell_seconds, 2) if dwell_seconds is not None else None,
        confidence=sample.confidence,
        censored=censored,
    )


def side_sign(point: Point, a: Point, b: Point) -> int:
    value = side_of_line(point, a, b)
    return (value > 0) - (value < 0)


def door_events(camera: Camera, track_id: int, samples: list[Sample], debounce_seconds: float) -> list[Event]:
    door_a, door_b = camera.door_line
    inside_sign = side_sign(camera.inside_point, door_a, door_b)

    def side(sample: Sample) -> str:
        return "inside" if side_sign(sample.foot, door_a, door_b) == inside_sign else "outside"

    events = []
    for index, _ in debounced_changes(samples, side, debounce_seconds)[1:]:
        direction = crossing_direction(samples[index - 1].foot, samples[index].foot, door_a, door_b,
                                       camera.inside_point)
        if direction == "in":
            events.append(make_event(camera, track_id, "ENTRY", samples[index]))
        elif direction == "out":
            events.append(make_event(camera, track_id, "EXIT", samples[index]))
    return events


def area_events(camera: Camera, track_id: int, samples: list[Sample], areas: dict[str, list[Point]],
                debounce_seconds: float, enter_type: str, exit_type: str) -> tuple[list[Event], list[Visit]]:
    def area_of(sample: Sample) -> str:
        for name, polygon in areas.items():
            if point_in_polygon(sample.foot, polygon):
                return name
        return ""

    changes = debounced_changes(samples, area_of, debounce_seconds)
    visits = visits_from_changes(changes, last_index=len(samples) - 1)
    events = []
    for visit in visits:
        start, end = samples[visit.start], samples[visit.end]
        dwell = end.seconds - start.seconds
        events.append(make_event(camera, track_id, enter_type, start, visit.zone, censored=visit.censored))
        events.append(make_event(camera, track_id, exit_type, end, visit.zone, dwell, visit.censored))
    return events, visits


def queue_depth_per_second(camera: Camera, queue_visits: list[tuple[float, float]], duration: float) -> list[QueueDepth]:
    depths = []
    for second in range(int(duration) + 1):
        depth = sum(1 for start, end in queue_visits if start <= second <= end)
        depths.append(QueueDepth(camera.id, second, camera.clock_at(second).isoformat(sep=" "), depth))
    return depths


def camera_events(camera: Camera, rows: list[TrackRow], settings: EventSettings) -> tuple[list[Event], list[QueueDepth]]:
    tracks = drop_short_tracks(group_by_track(rows), settings.min_track_seconds)
    events: list[Event] = []
    queue_visits: list[tuple[float, float]] = []

    for track_id, samples in sorted(tracks.items()):
        if camera.door_line:
            events += door_events(camera, track_id, samples, settings.debounce_seconds)
        if camera.zones:
            zone_events, _ = area_events(camera, track_id, samples, camera.zones, settings.debounce_seconds,
                                         "ZONE_ENTER", "ZONE_EXIT")
            events += zone_events
        if camera.queue_area:
            queue_events, visits = area_events(camera, track_id, samples, {"queue": camera.queue_area},
                                               settings.debounce_seconds, "QUEUE_JOIN", "QUEUE_LEAVE")
            events += queue_events
            queue_visits += [(samples[v.start].seconds, samples[v.end].seconds) for v in visits]

    depths = []
    if camera.queue_area and rows:
        depths = queue_depth_per_second(camera, queue_visits, max(row.seconds for row in rows))
    events.sort(key=lambda event: (event.seconds, event.track_id))
    return events, depths


def write_rows(rows: list, row_type: type, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run(config: StoreConfig) -> None:
    all_events: list[Event] = []
    all_depths: list[QueueDepth] = []
    for camera in config.cameras:
        rows = read_tracks_csv(config.outputs_dir / "tracks" / f"{camera.id}.csv")
        events, depths = camera_events(camera, rows, config.events)
        all_events += events
        all_depths += depths
        counts = defaultdict(int)
        for event in events:
            counts[event.event_type] += 1
        summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "no events"
        print(f"{camera.id:<9} {summary}")

    write_rows(all_events, Event, config.outputs_dir / "events.csv")
    write_rows(all_depths, QueueDepth, config.outputs_dir / "queue_depth.csv")
    print(f"wrote {len(all_events)} events and {len(all_depths)} queue-depth rows")


def main() -> None:
    argparse.ArgumentParser(description="Turn tracks into entry, zone and queue events.").parse_args()
    run(load_config())


if __name__ == "__main__":
    main()
