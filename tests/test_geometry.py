from src.geometry import crossing_direction, foot_point, point_in_polygon, segments_intersect, side_of_line

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
DOOR = ((0, 5), (10, 5))
INSIDE = (5, 0)


def test_foot_point_is_bottom_centre():
    assert foot_point(10, 20, 30, 100) == (20, 100)


def test_point_in_polygon():
    assert point_in_polygon((5, 5), SQUARE)
    assert not point_in_polygon((15, 5), SQUARE)
    assert not point_in_polygon((5, -1), SQUARE)


def test_point_in_concave_polygon():
    l_shape = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    assert point_in_polygon((2, 8), l_shape)
    assert not point_in_polygon((8, 8), l_shape)


def test_side_of_line_sign_flips_across_the_line():
    a, b = DOOR
    assert side_of_line((5, 0), a, b) * side_of_line((5, 10), a, b) < 0


def test_segments_intersect():
    assert segments_intersect((5, 0), (5, 10), *DOOR)
    assert not segments_intersect((15, 0), (15, 10), *DOOR)


def test_walking_in_and_out_through_the_door():
    assert crossing_direction((5, 8), (5, 2), *DOOR, INSIDE) == "in"
    assert crossing_direction((5, 2), (5, 8), *DOOR, INSIDE) == "out"


def test_crossing_the_extended_line_beside_the_door_is_not_an_entry():
    assert crossing_direction((15, 8), (15, 2), *DOOR, INSIDE) is None


def test_moving_parallel_to_the_door_is_not_a_crossing():
    assert crossing_direction((2, 8), (8, 8), *DOOR, INSIDE) is None
