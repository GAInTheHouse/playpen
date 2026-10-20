"""Turn a coarse macro loop into a valid Construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from players.player4.faces import FaceCatalog
from players.player4.lattice import ORIGIN, step, to_xy
from players.player4.macro import MacroClosed, MacroEdge
from players.player4.search import Anchor, Face, _clear, connector_for
from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType
from shapely.geometry import LineString

from src.weights import Weights

INWARD = 2


@dataclass(frozen=True)
class MicroLeg:
    turn: int
    heading: int
    length: int


def notch_legs(heading: int, min_piece: int, run: int) -> tuple[MicroLeg, ...]:
    h = heading
    h_in = (h + INWARD) % 8
    h_back = (h + 6) % 8
    return (
        MicroLeg(INWARD, h_in, min_piece),
        MicroLeg(-INWARD, h, run),
        MicroLeg(-INWARD, h_back, min_piece),
        MicroLeg(INWARD, h, 0),
    )


def _partition(
    total: int,
    face_lengths: tuple[int, ...],
    max_notches: int,
    min_piece: int,
    notch_run: int,
) -> tuple[tuple[int, ...], int] | None:
    lengths = tuple(sorted({l for l in face_lengths if MIN_WALL_LENGTH <= l <= MAX_FACE_LENGTH}))
    if not lengths or total < min(lengths):
        return None

    @lru_cache(maxsize=None)
    def dp(rem: int, notches_left: int) -> tuple[tuple[int, ...], int] | None:
        if rem == 0:
            return ((), 0)
        best: tuple[tuple[int, ...], int] | None = None
        for fl in lengths:
            if fl > rem:
                continue
            sub = dp(rem - fl, notches_left)
            if sub is None:
                continue
            parts, n = sub
            candidate = ((fl,) + parts, n)
            if best is None or len(candidate[0]) < len(best[0]):
                best = candidate
        if notches_left > 0:
            for fl in lengths:
                inner = rem - fl - notch_run
                if inner <= 0:
                    continue
                sub = dp(inner, notches_left - 1)
                if sub is None:
                    continue
                parts, n = sub
                candidate = ((fl,) + parts, n + 1)
                if best is None or candidate[1] < best[1]:
                    best = candidate
        return best

    return dp(total, max_notches)


def expand_macro_edge(
    edge: MacroEdge,
    face_lengths: tuple[int, ...],
    max_notches: int,
    min_piece: int,
    notch_run: int,
) -> tuple[MicroLeg, ...] | None:
    plan = _partition(edge.length, face_lengths, max_notches, min_piece, notch_run)
    if plan is None:
        return None
    parts, _ = plan
    legs: list[MicroLeg] = []
    for i, plen in enumerate(parts):
        if plen <= 0:
            return None
        if i > 0:
            for leg in notch_legs(edge.heading, min_piece, notch_run):
                if leg.length > 0:
                    legs.append(leg)
        turn = edge.turn if i == 0 else 0
        legs.append(MicroLeg(turn, edge.heading, plen))
    return tuple(legs)


def realize_closed(
    anchor: Anchor,
    closed: MacroClosed,
    catalog: FaceCatalog,
    connectors: tuple[int, int, int],
    weights: Weights,
    room,
) -> Construction | None:
    from players.player4.search import _spend

    face_lengths = tuple(catalog.lengths)
    if not face_lengths:
        return None
    min_piece = min(face_lengths)
    notch_run = min_piece
    rights = connectors[1]

    legs: list[MicroLeg] = []
    for edge in closed.edges:
        expanded = expand_macro_edge(edge, face_lengths, rights, min_piece, notch_run)
        if expanded is None:
            return None
        legs.extend(expanded)
    if not legs:
        return None

    # Replace turn on the very first leg with the closing turn.
    legs[0] = MicroLeg(closed.closing_turn, legs[0].heading, legs[0].length)

    positive = [i for i, leg in enumerate(legs) if leg.length > 0]
    gate_patterns: list[tuple[bool, ...]] = [(False,) * len(legs)]
    for i in positive[:8]:
        for j in positive:
            if j <= i:
                continue
            pat = [False] * len(legs)
            pat[i] = pat[j] = True
            gate_patterns.append(tuple(pat))

    best: Construction | None = None
    best_score = -1.0

    for gates in gate_patterns:
        remaining = catalog.stock
        budget = connectors
        faces: list[Face] = []
        points: list[tuple[float, float]] = [anchor.start]
        pos = ORIGIN
        prev_heading = 0

        ok = True
        for idx, leg in enumerate(legs):
            if leg.length == 0:
                continue
            turn = leg.turn
            if idx > 0 and turn == 0:
                from players.player4.lattice import turn_between

                turn = turn_between(prev_heading, leg.heading)
            pos = step(pos, leg.heading, leg.length)
            end = anchor.place(*to_xy(pos))
            closing = idx == len(legs) - 1
            if not room.contains(LineString([points[-1], end])):
                ok = False
                break
            if not _clear(tuple(points), end, closing=closing):
                ok = False
                break
            points.append(end)

            with_gate = gates[idx] if idx < len(gates) else False
            option = catalog.pick(remaining, leg.length, with_gate)
            if option is None:
                ok = False
                break
            budget = _spend(budget, (turn,) if turn else (), option.straights)
            if budget is None:
                ok = False
                break
            remaining = FaceCatalog.take(remaining, option)
            faces.append(Face(leg.heading, turn, option))
            prev_heading = leg.heading

        if not ok or not faces:
            continue
        gate_faces = sum(f.option.gates for f in faces)
        if gate_faces == 0:
            continue

        from src.constants import BASELINE_SCORE

        twice = 0.0
        for i in range(len(points) - 1):
            twice += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
        area = twice / 2
        if area <= 0:
            continue
        perimeter = sum(f.option.length for f in faces)
        score = (
            BASELINE_SCORE
            + weights.A * area
            + weights.C * perimeter
            + (weights.G if gate_faces >= 2 else 0.0)
        )
        if score <= best_score:
            continue
        pieces, conn = [], []
        for face in faces:
            for j, piece in enumerate(face.option.pieces):
                pieces.append(piece)
                conn.append(
                    connector_for(face.turn)
                    if j == 0
                    else Connector(ConnectorType.STRAIGHT)
                )
        best = Construction(
            start=anchor.start,
            start_heading=anchor.heading_deg,
            pieces=pieces,
            connectors=conn,
        )
        best_score = score

    return best
