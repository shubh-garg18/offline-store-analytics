from src.apply_corrections import Correction, apply
from src.labels import Box, yolo_line

SPLIT_OF = {"entry_00010.jpg": "train", "entry_00020.jpg": "train"}


def box(image: str, number: int, source: str = "teacher", x1: float = 0.1, y1: float = 0.1,
        x2: float = 0.2, y2: float = 0.5) -> Box:
    return Box(image, "train", number, x1, y1, x2, y2, 0.9 if source == "teacher" else 0.2, source)


def test_candidates_are_not_labels_until_accepted():
    proposals = [box("entry_00010.jpg", 1), box("entry_00010.jpg", 2, "candidate")]
    labels, _ = apply(proposals, [], SPLIT_OF)
    assert [b.box for b in labels] == [1]

    accepted, counts = apply(proposals, [Correction("entry_00010.jpg", "accept", 2, None, "")], SPLIT_OF)
    assert sorted(b.box for b in accepted) == [1, 2] and counts["accepted"] == 1


def test_remove_and_move_one_box():
    proposals = [box("entry_00010.jpg", 1), box("entry_00010.jpg", 2)]
    corrections = [Correction("entry_00010.jpg", "remove", 1, None, "reflection"),
                   Correction("entry_00010.jpg", "move", 2, (0.3, 0.3, 0.4, 0.9), "box too short")]
    labels, counts = apply(proposals, corrections, SPLIT_OF)
    assert len(labels) == 1 and (labels[0].x1, labels[0].y2) == (0.3, 0.9)
    assert counts["removed"] == 1 and counts["moved"] == 1


def test_region_removal_applies_to_every_frame_of_that_camera():
    proposals = [box("entry_00010.jpg", 1), box("entry_00020.jpg", 1), box("entry_00020.jpg", 2, x1=0.7, x2=0.8)]
    region = Correction("entry", "remove_region", None, (0.0, 0.0, 0.5, 1.0), "poster")
    labels, counts = apply(proposals, [region], SPLIT_OF)
    assert [(b.image, b.box) for b in labels] == [("entry_00020.jpg", 2)] and counts["removed"] == 2


def test_added_box_gets_a_number_no_proposal_uses():
    proposals = [box("entry_00010.jpg", 1), box("entry_00010.jpg", 2, "candidate")]
    labels, _ = apply(proposals, [Correction("entry_00010.jpg", "add", None, (0.5, 0.1, 0.6, 0.6), "")], SPLIT_OF)
    added = [b for b in labels if b.source == "review"]
    assert added[0].box == 3


def test_yolo_line_is_centre_and_size():
    assert yolo_line(box("x.jpg", 1, x1=0.1, y1=0.2, x2=0.3, y2=0.6)) == "0 0.200000 0.400000 0.200000 0.400000"
