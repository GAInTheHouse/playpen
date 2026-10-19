"""The shape families Player10 can build, as face lengths plus the turn at each corner.

Every face has an integer length and every heading is a multiple of 45 degrees, so a
loop closes exactly when, along each of the four axes (0, 45, 90, 135 degrees), the
length walked forward equals the length walked back: sqrt(2) is irrational, so diagonal
faces can only cancel each other. The length rules below come straight from that.
"""

import math
from dataclasses import dataclass

from src.pieces import Connector, ConnectorType

SQRT2 = math.sqrt(2.0)

RECTANGLE = "rectangle"
OCTAGON = "octagon"
L_SHAPE = "L"
U_SHAPE = "U"
HEXAGON = "hexagon"
FREE = "free-form"

# how many corner connectors of each type a family needs
CORNERS_NEEDED = {
    RECTANGLE: {ConnectorType.RIGHT: 4},
    OCTAGON: {ConnectorType.DIAGONAL: 8},
    L_SHAPE: {ConnectorType.RIGHT: 6},
    U_SHAPE: {ConnectorType.RIGHT: 8},
    HEXAGON: {ConnectorType.RIGHT: 2, ConnectorType.DIAGONAL: 4},
}

# a shape looks the same again after turning this far, so placements only need
# to try headings below it (an L has no such symmetry)
SYMMETRY_DEG = {RECTANGLE: 180, OCTAGON: 180, L_SHAPE: 360, U_SHAPE: 360, HEXAGON: 180}
# the step between the headings worth trying, relative to a room edge
HEADING_STEP_DEG = {RECTANGLE: 90, OCTAGON: 45, L_SHAPE: 90, U_SHAPE: 90, HEXAGON: 45}


@dataclass(frozen=True)
class Shape:
    family: str
    params: tuple[int, ...]
    faces: tuple[int, ...]  # face lengths, in walking order
    turns: tuple[float, ...]  # turns[i] is the turn at the start of face i
    area: float
    perimeter: float

    def corner(self, face: int) -> Connector:
        """The connector at the start of `face`."""
        turn = self.turns[face]
        kind = ConnectorType.RIGHT if abs(turn) == 90 else ConnectorType.DIAGONAL
        return Connector(kind, reflex=turn < 0)

    def vertices(self) -> list[tuple[float, float]]:
        """The corners when face 0 starts at the origin heading along +x."""
        points = [(0.0, 0.0)]
        heading = 0.0
        for i, length in enumerate(self.faces[:-1]):
            if i:
                heading += self.turns[i]
            t = math.radians(heading)
            x, y = points[-1]
            points.append((x + length * math.cos(t), y + length * math.sin(t)))
        return points

    def describe(self) -> str:
        if self.family == FREE:
            return f"{FREE}, {len(self.faces)} faces"
        if self.family == "polish":
            return f"polish, {len(self.faces)} faces"
        return f"{self.family} {'x'.join(str(p) for p in self.params)}"


def rectangle_measures(a: int, b: int) -> tuple[float, float]:
    return float(a * b), 2.0 * (a + b)


def octagon_measures(p: int, r: int, q: int, t: int) -> tuple[float, float]:
    """An octagon walked east p, north-east r, north q, north-west t, and back."""
    cut = (r + t) / SQRT2
    return (p + cut) * (q + cut) - (r * r + t * t) / 2.0, 2.0 * (p + q + r + t)


def l_measures(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """An `a` wide, `b + d` tall rectangle with a `c` by `d` notch out of one corner."""
    return float(a * (b + d) - c * d), 2.0 * (a + b + d)


def u_measures(left: int, gap: int, right: int, height: int, leg: int) -> tuple:
    """A `left + gap + right` wide, `height` tall rectangle with a `gap` by `leg`
    notch out of the middle of one side."""
    width = left + gap + right
    return float(width * height - gap * leg), 2.0 * (width + height + leg)


def hexagon_measures(a: int, d1: int, d2: int) -> tuple[float, float]:
    """A hexagon walked east a, north-east d1, north-west d2, and back: a d1 by d2
    rectangle standing on its corner, stretched sideways by a. It needs only two
    right-angle connectors, with four diagonal ones."""
    return d1 * d2 + a * (d1 + d2) / SQRT2, 2.0 * (a + d1 + d2)


MEASURES = {
    RECTANGLE: rectangle_measures,
    OCTAGON: octagon_measures,
    L_SHAPE: l_measures,
    U_SHAPE: u_measures,
    HEXAGON: hexagon_measures,
}


def make_shape(family: str, params: tuple[int, ...]) -> Shape:
    area, perimeter = MEASURES[family](*params)
    if family == RECTANGLE:
        a, b = params
        faces, turns = (a, b, a, b), (90.0,) * 4
    elif family == OCTAGON:
        p, r, q, t = params
        faces, turns = (p, r, q, t, p, r, q, t), (45.0,) * 8
    elif family == L_SHAPE:
        a, b, c, d = params
        faces = (a, b, c, d, a - c, b + d)
        turns = (90.0, 90.0, 90.0, -90.0, 90.0, 90.0)
    elif family == HEXAGON:
        a, d1, d2 = params
        faces = (a, d1, d2, a, d1, d2)
        turns = (45.0, 45.0, 90.0, 45.0, 45.0, 90.0)
    else:
        left, gap, right, height, leg = params
        faces = (left + gap + right, height, right, leg, gap, leg, left, height)
        turns = (90.0, 90.0, 90.0, 90.0, -90.0, -90.0, 90.0, 90.0)
    return Shape(family, params, faces, turns, area, perimeter)


def free_shape(headed_faces: tuple[tuple[int, int], ...]) -> Shape:
    """A shape from faces given as (heading in 45-degree steps, length)."""
    lengths = tuple(length for _, length in headed_faces)
    headings = [h for h, _ in headed_faces]
    turns = tuple(
        ((h - headings[i - 1] + 3) % 8 - 3) * 45.0 for i, h in enumerate(headings)
    )
    return Shape(FREE, lengths, lengths, turns, 0.0, float(sum(lengths)))


def headed_faces(shape: Shape) -> tuple[tuple[int, int], ...]:
    """A shape's faces as (heading in 45-degree steps from face 0, length)."""
    out = []
    heading = 0
    for i, length in enumerate(shape.faces):
        if i:
            heading = (heading + round(shape.turns[i] / 45.0)) % 8
        out.append((heading, length))
    return tuple(out)


def contains_smaller(family: str, big: tuple[int, ...], small: tuple[int, ...]) -> bool:
    """Whether the `big` shape can hold a copy of `small`. If `small` fits nowhere in
    the room, neither does `big`."""
    if family == RECTANGLE:
        return big[0] >= small[0] and big[1] >= small[1]
    if family == OCTAGON:
        # an octagon is a sum of four segments, so longer segments contain shorter;
        # the four cyclic shifts are the same octagon turned by 45 degrees
        return any(
            all(big[(i + shift) % 4] >= small[i] for i in range(4))
            for shift in range(4)
        )
    if family == U_SHAPE:
        return False  # no simple rule; every U is tried
    if family == HEXAGON:
        # a sum of three segments, like the octagon
        return all(b >= s for b, s in zip(big, small))
    a, b, c, d = big
    sa, sb, sc, sd = small
    return a >= sa and b + d >= sb + sd and a - c >= sa - sc and b >= sb


def inner_rectangles(family: str, params: tuple[int, ...]) -> list[tuple[int, int]]:
    """Rectangles that lie inside an L or a U. If one of them fits nowhere in the
    room, neither does the shape."""
    if family == L_SHAPE:
        a, b, c, d = params
        return [(a, b), (a - c, b + d)]
    if family == U_SHAPE:
        left, gap, right, height, leg = params
        return [(left + gap + right, height - leg), (left, height), (right, height)]
    return []


def inner_radius(family: str, params: tuple[int, ...]) -> float:
    """The radius of a circle that certainly fits inside the shape. A room with no
    room for that circle has no room for the shape."""
    if family == RECTANGLE:
        return min(params) / 2.0
    if family == OCTAGON:
        p, r, q, t = params
        widths = (
            p + (r + t) / SQRT2,
            q + (r + t) / SQRT2,
            r + (p + q) / SQRT2,
            t + (p + q) / SQRT2,
        )
        return min(widths) / 2.0
    if family == HEXAGON:
        a, d1, d2 = params
        return min((d1 + d2) / SQRT2, a / SQRT2 + d1, a / SQRT2 + d2) / 2.0
    return max(min(rect) for rect in inner_rectangles(family, params)) / 2.0
