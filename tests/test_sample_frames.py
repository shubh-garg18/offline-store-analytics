import pytest

from src.config import FinetuneSettings
from src.sample_frames import camera_of, plan_samples

SETTINGS = FinetuneSettings(
    student_model="yolo11n.pt", teacher_model="yolo11x.pt", teacher_confidence=0.3,
    teacher_candidate_confidence=0.1, teacher_imgsz=960, frames_per_camera=120, min_gap_seconds=0.5,
    val_fraction=0.3, split_gap_seconds=2.0, epochs=1, imgsz=640, batch=8, patience=5,
)
FPS = 29.97
TOTAL_FRAMES = 4436  # the entry clip, 148 s
BOUNDARY = TOTAL_FRAMES / FPS * 0.7


def test_val_frames_all_come_from_the_last_part_of_the_clip():
    val = [p for p in plan_samples(TOTAL_FRAMES, FPS, SETTINGS) if p.split == "val"]
    assert val and all(p.seconds >= BOUNDARY for p in val)


def test_train_frames_all_come_from_the_first_part_of_the_clip():
    train = [p for p in plan_samples(TOTAL_FRAMES, FPS, SETTINGS) if p.split == "train"]
    assert train and all(p.seconds < BOUNDARY for p in train)


def test_no_frame_sits_close_to_the_split_boundary():
    planned = plan_samples(TOTAL_FRAMES, FPS, SETTINGS)
    assert all(abs(p.seconds - BOUNDARY) >= SETTINGS.split_gap_seconds - 1 / FPS for p in planned)


def test_frames_are_far_enough_apart_not_to_be_copies():
    planned = plan_samples(TOTAL_FRAMES, FPS, SETTINGS)
    gaps = [b.seconds - a.seconds for a, b in zip(planned, planned[1:])]
    assert min(gaps) >= SETTINGS.min_gap_seconds


def test_too_many_frames_for_a_short_clip_is_refused():
    with pytest.raises(ValueError):
        plan_samples(300, FPS, SETTINGS)  # 10 s cannot hold 120 frames 0.5 s apart


def test_camera_name_is_recovered_from_the_image_name():
    assert camera_of("zone_2_01234.jpg") == "zone_2"
    assert camera_of("entry_00010.jpg") == "entry"
