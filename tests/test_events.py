from datetime import datetime
from pathlib import Path

from src.config import Camera, EventSettings
from src.events import camera_events
from src.track_table import TrackRow

FPS = 30
SETTINGS = EventSettings(min_track_seconds=1.0, debounce_seconds=0.33, match_tolerance_seconds=2.0)
START = datetime(2026, 4, 10, 20, 10, 0)


def walk(track_id: int, feet: list[tuple[float, float]], start_frame: int = 0) -> list[TrackRow]:
    """One row per frame, with a 20x40 px box whose bottom-centre is the given foot point."""
    return [
        TrackRow(frame=start_frame + i, seconds=(start_frame + i) / FPS, track_id=track_id,
                 x1=x - 10, y1=y - 40, x2=x + 10, y2=y, confidence=0.8)
        for i, (x, y) in enumerate(feet)
    ]


def line(start: tuple[float, float], end: tuple[float, float], frames: int) -> list[tuple[float, float]]:
    return [(start[0] + (end[0] - start[0]) * i / (frames - 1), start[1] + (end[1] - start[1]) * i / (frames - 1))
            for i in range(frames)]


def door_camera() -> Camera:
    return Camera(id="entry", clip_path=Path("x.mp4"), role="entry", clock_start=START,
                  door_line=((100, 500), (300, 500)), inside_point=(200, 100))


def zone_camera() -> Camera:
    return Camera(id="zone", clip_path=Path("x.mp4"), role="zone", clock_start=START,
                  zones={"shelf": [(0, 0), (100, 0), (100, 100), (0, 100)]})


def event_types(camera: Camera, rows: list[TrackRow]) -> list[str]:
    events, _ = camera_events(camera, rows, SETTINGS)
    return [event.event_type for event in events]


def test_walking_through_the_door_is_one_entry():
    rows = walk(1, line((200, 700), (200, 300), 60))
    assert event_types(door_camera(), rows) == ["ENTRY"]


def test_a_group_of_three_gives_three_entries():
    rows = []
    for track_id, x in enumerate([150, 200, 250], start=1):
        rows += walk(track_id, line((x, 700), (x, 300), 60))
    assert event_types(door_camera(), rows) == ["ENTRY", "ENTRY", "ENTRY"]


def test_walking_past_beside_the_door_is_not_an_entry():
    rows = walk(1, line((600, 700), (600, 300), 60))
    assert event_types(door_camera(), rows) == []


def test_jitter_on_the_door_line_does_not_spam_events():
    feet = [(200, 495 if i % 2 else 505) for i in range(60)]
    assert event_types(door_camera(), walk(1, feet)) == []


def test_tracks_shorter_than_the_minimum_are_ignored():
    rows = walk(1, line((200, 700), (200, 300), 15))  # half a second
    assert event_types(door_camera(), rows) == []


def test_zone_visit_has_dwell_and_is_uncensored_when_seen_arriving_and_leaving():
    feet = [(300, 50)] * 30 + [(50, 50)] * 90 + [(300, 50)] * 30
    events, _ = camera_events(zone_camera(), walk(1, feet), SETTINGS)
    assert [e.event_type for e in events] == ["ZONE_ENTER", "ZONE_EXIT"]
    assert events[1].dwell_seconds == 3.0
    assert not events[1].censored


def test_visit_already_in_progress_at_track_start_is_censored():
    feet = [(50, 50)] * 90 + [(300, 50)] * 30
    events, _ = camera_events(zone_camera(), walk(1, feet), SETTINGS)
    assert events[-1].event_type == "ZONE_EXIT"
    assert events[-1].censored


def test_event_ids_are_stable_across_runs():
    rows = walk(1, line((200, 700), (200, 300), 60))
    first, _ = camera_events(door_camera(), rows, SETTINGS)
    second, _ = camera_events(door_camera(), rows, SETTINGS)
    assert [e.event_id for e in first] == [e.event_id for e in second]
