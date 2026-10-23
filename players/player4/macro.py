"""Coarse beam search over long lattice edges (lengths from ray casts)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from shapely.geometry import LineString

from players.player4.faces import FaceCatalog
from players.player4.lattice import (
    ORIGIN,
    length_to_origin,
    lengths_to_origin,
    step,
    to_xy,
    turn_between,
)
from src.constants import MAX_FACE_LENGTH
from players.player4.search import (
    Anchor,
    TURNS,
    _clear,
    _cross,
    _home_distance,
    _spend,
)
from src.constants import BASELINE_SCORE, MIN_WALL_LENGTH
from src.weights import Weights

FULL_TURN = 8
MIN_TOTAL_TURN = -8
MAX_TOTAL_TURN = 16
MAX_MACRO_EDGES = 80
CLOSED_KEPT = 12
RAY_FRACTIONS = (1.0, 0.75, 0.5)
EPS = 1e-7

MACRO_INVENTORY_LENGTH = 840  # about 2 * 14 faces * 30
MIN_RIGHTS_FOR_MACRO = 16


def buildable_lengths(face_lengths: tuple[int, ...], max_length: int = 6000) -> frozenset[int]:
    if not face_lengths:
        return frozenset()
    reachable = {0}
    out: set[int] = set()
    min_len = min(face_lengths)
    for total in range(min_len, max_length + 1):
        for fl in face_lengths:
            if fl <= total and total - fl in reachable:
                reachable.add(total)
                out.add(total)
                break
    return frozenset(out)


def expected_notch_perimeter(edge_length: int, min_piece: int, notch_run: int) -> float:
    if edge_length <= MAX_FACE_LENGTH:
        return 0.0
    n = (edge_length + MAX_FACE_LENGTH - 1) // MAX_FACE_LENGTH - 1
    return n * (2 * min_piece + 2 * notch_run)


@dataclass(frozen=True)
class MacroEdge:
    turn: int
    heading: int
    length: int


@dataclass(frozen=True)
class MacroClosed:
    score: float
    closing_turn: int
    edges: tuple[MacroEdge, ...]


@dataclass(frozen=True)
class MacroState:
    pos: tuple
    heading: int
    total_turn: int
    connectors: tuple[int, int, int]
    edges: tuple[MacroEdge, ...]
    points: tuple[tuple[float, float], ...]
    twice_area: float
    length: int
    estimate: float


def should_use_macro(inventory) -> bool:
    from src.pieces import ConnectorType

    total = sum(n * length for length, n in inventory.walls.items())
    total += sum(n * length for length, n in inventory.gates.items())
    rights = inventory.connectors.get(ConnectorType.RIGHT, 0)
    return total > MACRO_INVENTORY_LENGTH and rights >= MIN_RIGHTS_FOR_MACRO


class MacroBeamSearch:
    def __init__(
        self,
        anchor: Anchor,
        catalog: FaceCatalog,
        connectors: tuple[int, int, int],
        weights: Weights,
        room,
        beam_width: int,
        deadline: float,
        realizable: frozenset[int],
        min_piece: int,
        notch_run: int,
    ):
        self.anchor = anchor
        self.catalog = catalog
        self.connectors = connectors
        self.weights = weights
        self.room = room
        self.beam_width = beam_width
        self.deadline = deadline
        self.realizable = realizable
        self.min_piece = min_piece
        self.notch_run = notch_run
        self.closed: list[MacroClosed] = []
        self._realizable_sorted = sorted(realizable)

    def run(self) -> list[MacroClosed]:
        root = MacroState(
            pos=ORIGIN,
            heading=0,
            total_turn=0,
            connectors=self.connectors,
            edges=(),
            points=(self.anchor.start,),
            twice_area=0.0,
            length=0,
            estimate=0.0,
        )
        beam = [root]
        for _ in range(MAX_MACRO_EDGES):
            children: list[MacroState] = []
            for state in beam:
                if time.process_time() > self.deadline:
                    return self._best()
                self._close(state)
                children.extend(self._expand(state))
            beam = self._select(children)
            if not beam:
                break
        return self._best()

    def _best(self) -> list[MacroClosed]:
        return sorted(self.closed, key=lambda c: c.score, reverse=True)[:CLOSED_KEPT]

    def _select(self, children: list[MacroState]) -> list[MacroState]:
        best: dict[tuple, MacroState] = {}
        for child in children:
            key = (child.pos, child.heading, child.total_turn)
            if key not in best or child.estimate > best[key].estimate:
                best[key] = child
        ranked = sorted(best.values(), key=lambda s: s.estimate, reverse=True)
        return ranked[: self.beam_width]

    def _snap_length(self, run: int) -> int:
        lo, hi = 0, len(self._realizable_sorted) - 1
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._realizable_sorted[mid] <= run:
                best = self._realizable_sorted[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def _segment_ok(self, st_points: tuple, from_xy: tuple[float, float], heading: int, pos, length: int) -> bool:
        if length < MIN_WALL_LENGTH:
            return False
        end = self.anchor.place(*to_xy(step(pos, heading, length)))
        if not self.room.contains(LineString([from_xy, end])):
            return False
        return _clear(st_points, end, closing=False)

    def _ray_lengths(
        self, st_points: tuple, from_xy: tuple[float, float], heading: int, pos
    ) -> list[int]:
        lo, hi = MIN_WALL_LENGTH, 6000
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._segment_ok(st_points, from_xy, heading, pos, mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best < MIN_WALL_LENGTH:
            return []
        out: set[int] = set()
        for frac in RAY_FRACTIONS:
            snapped = self._snap_length(int(best * frac))
            if snapped >= MIN_WALL_LENGTH:
                out.add(snapped)
        return sorted(out)

    def _expand(self, st: MacroState) -> list[MacroState]:
        turns = TURNS if st.edges else (0,)
        children: list[MacroState] = []
        last = st.points[-1]
        for t in turns:
            heading = (st.heading + t) % 8
            total = st.total_turn + t
            if not MIN_TOTAL_TURN <= total <= MAX_TOTAL_TURN:
                continue
            connectors = _spend(st.connectors, (t,) if t else (), 0)
            if connectors is None:
                continue
            lengths = self._ray_lengths(st.points, last, heading, st.pos)
            for length in lengths:
                if length not in self.realizable:
                    continue
                pos = step(st.pos, heading, length)
                if pos == ORIGIN:
                    continue
                end = self.anchor.place(*to_xy(pos))
                if not self.room.contains(LineString([last, end])):
                    continue
                if not _clear(st.points, end, closing=False):
                    continue
                twice_area = st.twice_area + _cross(last, end)
                home = _home_distance(pos)
                need = FULL_TURN - total
                if abs(need) > 2 * connectors[1] + connectors[2]:
                    continue
                notch_p = expected_notch_perimeter(length, self.min_piece, self.notch_run)
                start = st.points[0]
                chord = math.dist(end, start)
                estimate = (
                    self.weights.A * (twice_area + _cross(end, start)) / 2
                    + self.weights.C * (st.length + length + chord + notch_p)
                )
                children.append(
                    MacroState(
                        pos=pos,
                        heading=heading,
                        total_turn=total,
                        connectors=connectors,
                        edges=st.edges + (MacroEdge(t, heading, length),),
                        points=st.points + (end,),
                        twice_area=twice_area,
                        length=st.length + length,
                        estimate=estimate,
                    )
                )
        return children

    def _close(self, st: MacroState) -> None:
        if not st.edges:
            return
        need = FULL_TURN - st.total_turn
        lengths = self.realizable
        for h1 in range(8):
            t1 = turn_between(st.heading, h1)
            if t1 not in TURNS:
                continue
            t0 = turn_between(h1, 0)
            if len(st.edges) >= 2 and t0 in TURNS and t1 + t0 == need:
                length = length_to_origin(st.pos, h1)
                if length is not None and length in lengths:
                    self._finish(st, ((t1, h1, length),), t0)
            for h2 in range(8):
                t2 = turn_between(h1, h2)
                t0 = turn_between(h2, 0)
                if t2 not in TURNS or t0 not in TURNS or t1 + t2 + t0 != need:
                    continue
                found = lengths_to_origin(st.pos, h1, h2)
                if found and found[0] in lengths and found[1] in lengths:
                    self._finish(st, ((t1, h1, found[0]), (t2, h2, found[1])), t0)

    def _finish(self, st: MacroState, legs, closing_turn: int) -> None:
        connectors = _spend(st.connectors, [t for t, _, _ in legs] + [closing_turn], 0)
        if connectors is None:
            return
        points = list(st.points)
        pos = st.pos
        twice_area = st.twice_area
        perimeter = st.length
        for j, (_, heading, length) in enumerate(legs):
            pos = step(pos, heading, length)
            end = st.points[0] if j == len(legs) - 1 else self.anchor.place(*to_xy(pos))
            closing = j == len(legs) - 1
            if not self.room.contains(LineString([points[-1], end])):
                return
            if not _clear(points, end, closing=closing):
                return
            twice_area += _cross(points[-1], end)
            perimeter += length
            points.append(end)
        area = twice_area / 2
        if area <= 0:
            return
        notch_extra = sum(
            expected_notch_perimeter(l, self.min_piece, self.notch_run)
            for _, _, l in legs
        )
        notch_extra += sum(
            expected_notch_perimeter(e.length, self.min_piece, self.notch_run)
            for e in st.edges
        )
        score = BASELINE_SCORE + self.weights.A * area + self.weights.C * (
            perimeter + notch_extra
        )
        head = st.edges
        new_edges = tuple(MacroEdge(t, h, l) for t, h, l in legs)
        self.closed.append(MacroClosed(score, closing_turn, head + new_edges))
