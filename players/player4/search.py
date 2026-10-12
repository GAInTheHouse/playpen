"""Beam search for a closed loop from one anchor.

The loop is grown one linear face at a time in a local frame whose first face
points along heading 0; the anchor maps that frame into the room. Every turn is
a multiple of 45 degrees, so positions are kept exact (see lattice.py). From
every state the search also tries to close the loop with one or two more faces,
solved exactly rather than searched for.
"""

import math
import time
from dataclasses import dataclass
from itertools import product

import shapely
from shapely.geometry import LineString

from players.player4.faces import FaceCatalog, FaceOption
from players.player4.lattice import (
    ORIGIN,
    length_to_origin,
    lengths_to_origin,
    step,
    to_xy,
    turn_between,
)
from src.constants import BASELINE_SCORE
from src.pieces import Connector, ConnectorType
from src.weights import Weights

TURNS = (2, 1, -1, -2)  # 45-degree steps: +-2 is a RIGHT connector, +-1 a DIAGONAL
FULL_TURN = 8  # a counterclockwise simple loop turns through exactly 360 degrees
MIN_TOTAL_TURN = -4
MAX_TOTAL_TURN = 10
EPS = 1e-7
CLOSED_KEPT = 12


class Anchor:
    """Where the local frame sits in the room: its origin and the room-frame
    direction of local heading 0."""

    def __init__(self, start: tuple[float, float], heading_deg: float):
        self.start = start
        self.heading_deg = heading_deg
        t = math.radians(heading_deg)
        self._cos, self._sin = math.cos(t), math.sin(t)

    def place(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.start[0] + x * self._cos - y * self._sin,
            self.start[1] + x * self._sin + y * self._cos,
        )


@dataclass(frozen=True)
class Face:
    heading: int
    turn: int  # the turn at its first vertex; for face 0 that's the closing turn
    option: FaceOption


@dataclass(frozen=True)
class Closed:
    score: float
    faces: tuple[Face, ...]


@dataclass(frozen=True)
class State:
    pos: tuple
    heading: int
    total_turn: int
    remaining: tuple[int, ...]
    connectors: tuple[int, int, int]  # straight, right, diagonal left
    faces: tuple[Face, ...]
    points: tuple[tuple[float, float], ...]  # room frame; points[0] is the anchor
    twice_area: float  # shoelace sum over the faces laid so far
    length: int
    gate_faces: int
    estimate: float


def connector_for(turn: int) -> Connector:
    if turn == 0:
        return Connector(ConnectorType.STRAIGHT)
    kind = ConnectorType.RIGHT if abs(turn) == 2 else ConnectorType.DIAGONAL
    return Connector(kind, reflex=turn < 0)


def _spend(connectors, turns, straights):
    s, r, d = connectors
    s -= straights
    for t in turns:
        if abs(t) == 2:
            r -= 1
        else:
            d -= 1
    if s < 0 or r < 0 or d < 0:
        return None
    return (s, r, d)


def _home_distance(pos) -> float:
    """The shortest path back to the origin using only 45-degree headings."""
    x, y = to_xy(pos)
    dx, dy = abs(x), abs(y)
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c) -> bool:
    return (
        min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    )


def _touch(p1, p2, q1, q2) -> bool:
    """Whether two segments share any point (crossing or touching)."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True
    return (
        (abs(d1) <= EPS and _on_segment(q1, q2, p1))
        or (abs(d2) <= EPS and _on_segment(q1, q2, p2))
        or (abs(d3) <= EPS and _on_segment(p1, p2, q1))
        or (abs(d4) <= EPS and _on_segment(p1, p2, q2))
    )


def _clear(points, end, closing: bool) -> bool:
    """Whether the segment from points[-1] to `end` avoids every earlier segment
    of the path. It shares an endpoint with the last segment, and when it is
    the closing segment, also with the first."""
    start = points[-1]
    for i in range(1 if closing else 0, len(points) - 2):
        if _touch(start, end, points[i], points[i + 1]):
            return False
    return True


class BeamSearch:
    def __init__(
        self,
        anchor: Anchor,
        catalog: FaceCatalog,
        connectors: tuple[int, int, int],
        weights: Weights,
        room,  # prepared shapely polygon, buffered by the containment tolerance
        beam_width: int,
        max_faces: int,
        deadline: float,
    ):
        self.anchor = anchor
        self.catalog = catalog
        self.connectors = connectors
        self.weights = weights
        self.room = room
        self.beam_width = beam_width
        self.max_faces = max_faces
        self.deadline = deadline
        self.closed: list[Closed] = []

    def run(self) -> list[Closed]:
        root = State(
            pos=ORIGIN,
            heading=0,
            total_turn=0,
            remaining=self.catalog.stock,
            connectors=self.connectors,
            faces=(),
            points=(self.anchor.start,),
            twice_area=0.0,
            length=0,
            gate_faces=0,
            estimate=0.0,
        )
        beam = [root]
        for _ in range(self.max_faces):
            children = []
            for state in beam:
                if time.process_time() > self.deadline:
                    return self._best()
                self._close(state)
                children.extend(self._expand(state))
            beam = self._select(children)
            if not beam:
                break
        return self._best()

    def _best(self) -> list[Closed]:
        return sorted(self.closed, key=lambda c: c.score, reverse=True)[:CLOSED_KEPT]

    def _select(self, children: list[State]) -> list[State]:
        best: dict[tuple, State] = {}
        for child in children:
            key = (child.pos, child.heading, child.total_turn, min(child.gate_faces, 2))
            if key not in best or child.estimate > best[key].estimate:
                best[key] = child
        ranked = sorted(best.values(), key=lambda s: s.estimate, reverse=True)
        return ranked[: self.beam_width]

    def _expand(self, st: State) -> list[State]:
        turns = TURNS if st.faces else (0,)
        moves = []
        for t in turns:
            heading = (st.heading + t) % 8
            total = st.total_turn + t
            if not MIN_TOTAL_TURN <= total <= MAX_TOTAL_TURN:
                continue
            connectors = _spend(st.connectors, (t,) if t else (), 0)
            if connectors is None:
                continue
            for length in self.catalog.lengths:
                pos = step(st.pos, heading, length)
                if pos == ORIGIN:
                    continue  # closing is _close's job
                moves.append((t, heading, total, connectors, length, pos))
        if not moves:
            return []

        last = st.points[-1]
        ends = [self.anchor.place(*to_xy(m[5])) for m in moves]
        inside = shapely.contains_xy(
            self.room, [e[0] for e in ends], [e[1] for e in ends]
        )
        kept = [i for i, ok in enumerate(inside) if ok]
        if not kept:
            return []
        lines = shapely.linestrings([[last, ends[i]] for i in kept])
        fits = shapely.contains(self.room, lines)

        children = []
        start = st.points[0]
        for i, ok in zip(kept, fits):
            if not ok or not _clear(st.points, ends[i], closing=False):
                continue
            t, heading, total, connectors, length, pos = moves[i]
            end = ends[i]
            twice_area = st.twice_area + _cross(last, end)
            home = _home_distance(pos)
            for with_gate in (False, True):
                option = self.catalog.pick(st.remaining, length, with_gate)
                if option is None:
                    continue
                left = _spend(connectors, (), option.straights)
                if left is None:
                    continue
                need = FULL_TURN - total
                if abs(need) > 2 * left[1] + left[2]:
                    continue
                remaining = FaceCatalog.take(st.remaining, option)
                if self.catalog.total_length(remaining) < home - EPS:
                    continue
                gate_faces = st.gate_faces + (1 if with_gate else 0)
                if gate_faces == 0 and not self.catalog.has_gate(remaining):
                    continue
                chord = math.dist(end, start)
                estimate = (
                    self.weights.A * (twice_area + _cross(end, start)) / 2
                    + self.weights.C * (st.length + length + chord)
                    + (self.weights.G if gate_faces >= 2 else 0.0)
                )
                children.append(
                    State(
                        pos=pos,
                        heading=heading,
                        total_turn=total,
                        remaining=remaining,
                        connectors=left,
                        faces=st.faces + (Face(heading, t, option),),
                        points=st.points + (end,),
                        twice_area=twice_area,
                        length=st.length + length,
                        gate_faces=gate_faces,
                        estimate=estimate,
                    )
                )
        return children

    def _close(self, st: State) -> None:
        if not st.faces:
            return
        need = FULL_TURN - st.total_turn
        lengths = self.catalog.length_set
        for h1 in range(8):
            t1 = turn_between(st.heading, h1)
            if t1 not in TURNS:
                continue
            t0 = turn_between(h1, 0)
            if len(st.faces) >= 2 and t0 in TURNS and t1 + t0 == need:
                length = length_to_origin(st.pos, h1)
                if length in lengths:
                    self._finish(st, ((t1, h1, length),), t0)
            for h2 in range(8):
                t2 = turn_between(h1, h2)
                t0 = turn_between(h2, 0)
                if t2 not in TURNS or t0 not in TURNS or t1 + t2 + t0 != need:
                    continue
                found = lengths_to_origin(st.pos, h1, h2)
                if found and found[0] in lengths and found[1] in lengths:
                    self._finish(st, ((t1, h1, found[0]), (t2, h2, found[1])), t0)

    def _finish(self, st: State, legs, closing_turn: int) -> None:
        connectors = _spend(st.connectors, [t for t, _, _ in legs] + [closing_turn], 0)
        if connectors is None:
            return

        points = list(st.points)
        pos = st.pos
        twice_area = st.twice_area
        for j, (_, heading, length) in enumerate(legs):
            pos = step(pos, heading, length)
            end = st.points[0] if j == len(legs) - 1 else self.anchor.place(*to_xy(pos))
            closing = j == len(legs) - 1
            if not self.room.contains(LineString([points[-1], end])):
                return
            if not _clear(points, end, closing=closing):
                return
            twice_area += _cross(points[-1], end)
            points.append(end)
        area = twice_area / 2
        if area <= 0:
            return
        perimeter = st.length + sum(length for _, _, length in legs)

        first = st.faces[0]
        head = (Face(first.heading, closing_turn, first.option),) + st.faces[1:]
        for gates in product((False, True), repeat=len(legs)):
            gate_faces = st.gate_faces + sum(gates)
            if gate_faces == 0:
                continue
            remaining = st.remaining
            straights = 0
            new_faces = []
            for (t, heading, length), with_gate in zip(legs, gates):
                option = self.catalog.pick(remaining, length, with_gate)
                if option is None:
                    break
                remaining = FaceCatalog.take(remaining, option)
                straights += option.straights
                new_faces.append(Face(heading, t, option))
            else:
                if _spend(connectors, (), straights) is None:
                    continue
                score = (
                    BASELINE_SCORE
                    + self.weights.A * area
                    + self.weights.C * perimeter
                    + (self.weights.G if gate_faces >= 2 else 0.0)
                )
                self.closed.append(Closed(score, head + tuple(new_faces)))
