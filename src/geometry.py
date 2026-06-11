"""Pure 2D geometry used to turn a person's foot position into events. Coordinates are in pixels."""

Point = tuple[float, float]
Polygon = list[Point]


def foot_point(x1: float, y1: float, x2: float, y2: float) -> Point:
    """Bottom-centre of a box: where the person touches the floor."""
    return ((x1 + x2) / 2, y2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray casting: shoot a ray to the right and count how many edges it crosses; odd means inside."""
    x, y = point
    inside = False
    for i in range(len(polygon)):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % len(polygon)]
        edge_spans_y = (ay > y) != (by > y)
        if edge_spans_y:
            x_where_edge_meets_ray = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < x_where_edge_meets_ray:
                inside = not inside
    return inside


def side_of_line(point: Point, a: Point, b: Point) -> float:
    """Cross product of (b - a) and (point - a): positive on one side, negative on the other, zero on the line."""
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def segments_intersect(p: Point, q: Point, a: Point, b: Point) -> bool:
    """True if segment p-q crosses segment a-b (touching at an end counts)."""
    d1 = side_of_line(p, a, b)
    d2 = side_of_line(q, a, b)
    d3 = side_of_line(a, p, q)
    d4 = side_of_line(b, p, q)
    return d1 * d2 <= 0 and d3 * d4 <= 0 and (d1 != 0 or d2 != 0)


def crossing_direction(previous: Point, current: Point, door_a: Point, door_b: Point, inside: Point) -> str | None:
    """'in' or 'out' if the step from previous to current crosses the door segment itself, else None.

    The infinite line through the door is not enough: someone walking past outside, beyond the door's
    end, would cross the extended line without ever going through the door.
    """
    if not segments_intersect(previous, current, door_a, door_b):
        return None
    inside_sign = side_of_line(inside, door_a, door_b)
    now_inside = side_of_line(current, door_a, door_b) * inside_sign > 0
    was_inside = side_of_line(previous, door_a, door_b) * inside_sign > 0
    if now_inside and not was_inside:
        return "in"
    if was_inside and not now_inside:
        return "out"
    return None
