"""Represent closed shapes and place them inside the complete room."""

import math
import signal
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import pairwise

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep

from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType

Point = tuple[float, float]
Pose = tuple[Point, float]  # Start position and rotation in degrees.

# Experiments can install a recorder. Normal runs retain no event data.
RECORDER = ContextVar("player2_recorder", default=None)


def record(stage, event, **details):
    recorder = RECORDER.get()
    if recorder is not None:
        recorder(stage, event, details)


# x = axial_x + diagonal_x / sqrt(2), and likewise for y.
# Four integer sums let us check closure without floating-point rounding.
DIRECTIONS = (
    (1, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 1, 0, 0),
    (0, 0, -1, 1),
    (-1, 0, 0, 0),
    (0, 0, -1, -1),
    (0, -1, 0, 0),
    (0, 0, 1, -1),
)


class SearchStopped(Exception):
    """The search is incomplete because its allowance has ended."""


@dataclass(frozen=True)
class Budget:
    deadline: float

    @classmethod
    def start(cls, seconds: float) -> "Budget":
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("The CPU allowance must be finite and nonnegative")
        if hasattr(signal, "getitimer"):
            remaining, _ = signal.getitimer(signal.ITIMER_PROF)
            if remaining > 0:
                seconds = min(seconds, remaining * 0.9)
        return cls(time.process_time() + seconds)

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.process_time())

    def slice(self, seconds: float) -> "Budget":
        return Budget(min(self.deadline, time.process_time() + max(0, seconds)))

    def check(self) -> None:
        if not self.remaining:
            raise SearchStopped


def offset(direction: int, length: float) -> Point:
    x, y, dx, dy = DIRECTIONS[direction % 8]
    return length * (x + dx / math.sqrt(2)), length * (y + dy / math.sqrt(2))


def turn(previous: int, current: int) -> int:
    return (current - previous + 4) % 8 - 4


@dataclass(frozen=True)
class Shape:
    """An ordered boundary. Directions are multiples of 45 degrees."""

    directions: tuple[int, ...]
    lengths: tuple[int, ...]
    polygon: Polygon = field(init=False, compare=False, hash=False)

    def __post_init__(self):
        if len(self.directions) != len(self.lengths) or len(self.lengths) < 4:
            raise ValueError("Each face needs one direction and one length")
        if any(d not in range(8) for d in self.directions) or any(
            not isinstance(n, int) or not MIN_WALL_LENGTH <= n <= MAX_FACE_LENGTH
            for n in self.lengths
        ):
            raise ValueError("Illegal face direction or length")
        if any(abs(t) not in (1, 2) for t in self.turns) or sum(self.turns) != 8:
            raise ValueError("Corners must make one counterclockwise loop")
        position = (0, 0, 0, 0)
        vertices = []
        for direction, length in zip(self.directions, self.lengths):
            x, y, dx, dy = position
            vertices.append((x + dx / math.sqrt(2), y + dy / math.sqrt(2)))
            position = tuple(
                a + length * b for a, b in zip(position, DIRECTIONS[direction])
            )
        polygon = Polygon(vertices)
        if any(position) or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("The boundary must close without crossings")
        object.__setattr__(self, "polygon", polygon)

    @property
    def turns(self) -> tuple[int, ...]:
        return tuple(
            turn(self.directions[i - 1], d) for i, d in enumerate(self.directions)
        )

    def placed(self, pose: Pose) -> Polygon:
        start, angle = pose
        return translate(rotate(self.polygon, angle, origin=(0, 0)), *start)

    def construction(self, faces, pose: Pose) -> Construction:
        if tuple(sum(p.length for p in face) for face in faces) != self.lengths:
            raise ValueError("Allocated pieces do not match the faces")
        pieces, connectors = [], []
        for corner, face in zip(self.turns, faces):
            kind = ConnectorType.RIGHT if abs(corner) == 2 else ConnectorType.DIAGONAL
            for index, piece in enumerate(face):
                pieces.append(piece)
                connectors.append(
                    Connector(
                        kind if index == 0 else ConnectorType.STRAIGHT,
                        reflex=index == 0 and corner < 0,
                    )
                )
        start, angle = pose
        return Construction(start, angle + self.directions[0] * 45, pieces, connectors)


def vertices(polygon: Polygon) -> tuple[Point, ...]:
    """Remove straight intermediate vertices and use counterclockwise order."""
    points = tuple(orient(polygon, sign=1).exterior.coords)[:-1]
    return tuple(
        b
        for a, b, c in zip(points[-1:] + points[:-1], points, points[1:] + points[:1])
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-7
    )


def boundary_shape(polygon: Polygon, angle: float) -> tuple[Shape, Pose]:
    points = vertices(polygon)
    directions, lengths = [], []
    for a, b in pairwise((*points, points[0])):
        direction = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) / 45
        length = math.dist(a, b)
        if (
            abs(direction - round(direction)) > 1e-7
            or abs(length - round(length)) > 1e-7
        ):
            raise ValueError("Boundary faces need legal directions and integer lengths")
        directions.append(round(direction) % 8)
        lengths.append(round(length))
    x, y = points[0]
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return Shape(tuple(directions), tuple(lengths)), (
        (co * x - si * y, si * x + co * y),
        angle,
    )


def convex_start(room: Polygon, points, budget: Budget) -> Point | None:
    """Clip the possible translations against each convex room edge."""
    xmin, ymin, xmax, ymax = room.bounds
    left, right = xmin - min(x for x, y in points), xmax - max(x for x, y in points)
    bottom, top = ymin - min(y for x, y in points), ymax - max(y for x, y in points)
    if left > right + 1e-9 or bottom > top + 1e-9:
        return None
    feasible = [(left, bottom), (right, bottom), (right, top), (left, top)]
    for a, b in pairwise(orient(room, sign=1).exterior.coords):
        budget.check()
        nx, ny = b[1] - a[1], a[0] - b[0]
        limit = nx * a[0] + ny * a[1] - max(nx * x + ny * y for x, y in points) + 1e-9
        clipped = []
        for p, q in zip(feasible, feasible[1:] + feasible[:1]):
            dp, dq = nx * p[0] + ny * p[1] - limit, nx * q[0] + ny * q[1] - limit
            if (dp <= 0) != (dq <= 0):
                fraction = dp / (dp - dq)
                clipped.append(
                    (p[0] + fraction * (q[0] - p[0]), p[1] + fraction * (q[1] - p[1]))
                )
            if dq <= 0:
                clipped.append(q)
        if not clipped:
            return None
        feasible = clipped
    return tuple(sum(p[i] for p in feasible) / len(feasible) for i in (0, 1))


class RoomGeometry:
    def __init__(self, room):
        self.polygon = room.polygon
        self.allowed = prep(self.polygon.buffer(TOL))
        self.convex = self.polygon.equals(self.polygon.convex_hull)
        self.corners = vertices(self.polygon)
        axes = {}
        for a, b in pairwise((*self.corners, self.corners[0])):
            angle = round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90, 8)
            axes[angle] = axes.get(angle, 0) + math.dist(a, b)
        self.axes = tuple(sorted(axes, key=lambda a: -axes[a]))

    def place(self, shape: Shape, budget: Budget) -> Pose | None:
        angles = dict.fromkeys(
            [a + delta for a in (0, *self.axes) for delta in (0, 45, 90, 135)]
            + list(range(0, 360, 15))
        )
        for angle in angles:
            budget.check()
            polygon = rotate(shape.polygon, angle, origin=(0, 0))
            points = tuple(polygon.exterior.coords)[:-1]
            if self.convex:
                start = convex_start(self.polygon, points, budget)
                if start is not None:
                    return start, angle
                continue
            center = polygon.centroid
            anchors = (self.polygon.centroid, self.polygon.representative_point())
            starts = [(p.x - center.x, p.y - center.y) for p in anchors]
            starts.extend(
                (rx - x, ry - y) for rx, ry in self.corners for x, y in points
            )
            for start in starts:
                budget.check()
                if self.allowed.covers(translate(polygon, *start)):
                    return start, angle
        return None
