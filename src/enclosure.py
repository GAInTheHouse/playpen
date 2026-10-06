"""Turns a player's proposed construction into geometry, validates it against
every rule in the spec, and scores it.

A Construction is a closed loop of N pieces (walls/gates) joined by N
connectors -- one connector per vertex. Vertex i's connector is the turn
applied at that vertex, between the incoming piece (i-1) and the outgoing
piece (i). Given a starting point + heading for piece 0, walking the N
pieces/turns must arrive back at the starting point and heading.
"""

import math
from dataclasses import dataclass, field

from shapely.geometry import Polygon

import src.constants as c
from src.inventory import Inventory
from src.pieces import Connector, Piece
from src.room import Room
from src.weights import Weights


@dataclass
class Construction:
    start: tuple[float, float]
    start_heading: float  # degrees, direction of piece 0 from `start`
    pieces: list[Piece]
    connectors: list[
        Connector
    ]  # len(connectors) == len(pieces); connectors[i] is the turn at vertex i


@dataclass
class Face:
    piece_indices: list[int]
    length: float
    has_gate: bool


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    polygon: Polygon | None = None
    vertices: list[tuple[float, float]] = field(default_factory=list)
    area: float = 0.0
    perimeter: float = 0.0
    gate_count: int = 0
    faces: list[Face] = field(default_factory=list)
    second_gate_bonus: bool = False


class EnclosureError(Exception):
    pass


def _angle_gap(a: float, b: float) -> float:
    """Smallest difference between two angles (degrees), in [0, 180]."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def _walk(
    construction: Construction, lenient: bool = False
) -> tuple[list[tuple[float, float]], list[float], list[int]]:
    """Walk the pieces/turns. Returns (points, headings, missing): n+1 points
    (the last is where the final piece actually ends), each piece's heading,
    and the vertices that have no connector.

    Strict (the default) raises EnclosureError for structural problems that
    make a real walk impossible. `lenient` is only for *drawing* a
    construction that has such problems: a vertex with no connector repeats
    the previous turn, so the wall after it still gets a plausible direction
    to be drawn in (see `sketch_points`).
    """
    n = len(construction.pieces)
    m = len(construction.connectors)
    if not lenient:
        if n != m:
            raise EnclosureError(
                f"need exactly one connector per vertex: got {n} pieces but "
                f"{m} connectors"
            )
        if n < c.MIN_ENCLOSURE_PIECES:
            raise EnclosureError(
                f"enclosure needs at least {c.MIN_ENCLOSURE_PIECES} pieces, got {n}"
            )

    headings = [0.0] * n
    if n:
        headings[0] = construction.start_heading
    last_turn = construction.connectors[0].turn_angle() if m else 90.0
    for i in range(1, n):
        if i < m:
            last_turn = construction.connectors[i].turn_angle()
        headings[i] = headings[i - 1] + last_turn

    points = [construction.start]
    for i in range(n):
        theta = math.radians(headings[i])
        px, py = points[-1]
        dx = construction.pieces[i].length * math.cos(theta)
        dy = construction.pieces[i].length * math.sin(theta)
        points.append((px + dx, py + dy))

    return points, headings, list(range(m, n))


def walk_points(construction: Construction) -> list[tuple[float, float]]:
    """All n+1 points of the walk. The last one is where the final piece
    really ends; it equals the first only if the loop actually closes, so
    this is what to draw when showing a construction that might not."""
    return _walk(construction)[0]


def sketch_points(
    construction: Construction,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Like `walk_points`, but never raises: for drawing a construction even
    when it's structurally broken (e.g. fewer connectors than vertices).
    Returns (points, missing) where `missing` lists the vertices that have no
    connector. The wall starting at such a vertex has no real direction; it is
    laid out as if the previous turn were repeated, so a forgotten corner of a
    rectangle still shows up where the rectangle wanted it."""
    points, _, missing = _walk(construction, lenient=True)
    return points, missing


def build_geometry(
    construction: Construction,
) -> tuple[list[tuple[float, float]], float, float]:
    """Walk the construction's pieces/connectors.

    Returns (vertices, position_gap, heading_gap) where the gaps measure how
    far the walk is from closing perfectly (position in units, heading in
    degrees). Raises EnclosureError for structural problems that make the
    walk impossible to even attempt.
    """
    points, headings, _ = _walk(construction)

    closing_turn = construction.connectors[0].turn_angle()
    expected_final_heading = headings[-1] + closing_turn
    heading_gap = _angle_gap(expected_final_heading, headings[0])

    position_gap = math.hypot(
        points[-1][0] - points[0][0], points[-1][1] - points[0][1]
    )

    return points[:-1], position_gap, heading_gap


def compute_faces(construction: Construction) -> list[Face]:
    """Group pieces into linear faces: consecutive runs joined by STRAIGHT
    (180 degree) connectors.

    `connectors[i]` is the turn at vertex i, between incoming piece (i-1) and
    outgoing piece i. So piece i starts a *new* face (relative to piece i-1)
    exactly when connectors[i] is not straight.
    """
    n = len(construction.pieces)
    starts = [i for i in range(n) if not construction.connectors[i].is_straight()]

    if not starts:
        # every connector is straight -- degenerate (shouldn't actually
        # close), but treat the whole loop as a single face for reporting
        total = sum(p.length for p in construction.pieces)
        has_gate = any(p.is_gate for p in construction.pieces)
        return [Face(piece_indices=list(range(n)), length=total, has_gate=has_gate)]

    faces = []
    for k, start in enumerate(starts):
        end = starts[(k + 1) % len(starts)]  # exclusive: where the next face begins
        indices = []
        i = start
        while True:
            indices.append(i)
            i = (i + 1) % n
            if i == end:
                break
        length = sum(construction.pieces[j].length for j in indices)
        has_gate = any(construction.pieces[j].is_gate for j in indices)
        faces.append(Face(piece_indices=indices, length=length, has_gate=has_gate))
    return faces


def unavailable_items(
    construction: Construction, inventory: Inventory
) -> tuple[list[int], list[int]]:
    """Which pieces and connectors (as indices into the construction) the
    inventory can't supply.

    Everything is taken in construction order -- all the pieces, then all the
    connectors -- so an item is listed when the inventory has already run out
    by the time its turn comes: a construction that needs two 13-unit walls
    from a stock of one lists the *second* one. The caller's inventory is not
    changed. This is also the order `validate_construction` checks in.
    """
    scratch = inventory.copy()
    short_pieces = [
        i
        for i, piece in enumerate(construction.pieces)
        if not scratch.take_piece(piece)
    ]
    short_connectors = [
        i
        for i, connector in enumerate(construction.connectors)
        if not scratch.take_connector(connector.connector_type)
    ]
    return short_pieces, short_connectors


def validate_construction(
    construction: Construction,
    room: Room,
    inventory: Inventory,
    tol: float = c.TOL,
) -> ValidationResult:
    """Check a construction against every rule in the spec:

    - uses only pieces/connectors available in `inventory`
    - every piece is at least MIN_WALL_LENGTH units
    - the loop closes (position and heading) within tolerance
    - the resulting shape is a simple (non-self-intersecting) polygon
    - the shape fits entirely within `room`
    - has at least one gate
    - every linear face is at most MAX_FACE_LENGTH units
    """
    try:
        vertices, position_gap, heading_gap = build_geometry(construction)
    except EnclosureError as e:
        return ValidationResult(valid=False, reason=str(e))

    if position_gap > tol:
        return ValidationResult(
            valid=False,
            reason=f"enclosure does not close: last piece ends {position_gap:.4f} units from the start",
        )
    if heading_gap > c.ANGLE_TOL:
        return ValidationResult(
            valid=False,
            reason=(
                "closing connector's angle is inconsistent with the geometry "
                f"(off by {heading_gap:.4f} degrees)"
            ),
        )

    short_pieces, short_connectors = unavailable_items(construction, inventory)
    if short_pieces:
        piece = construction.pieces[short_pieces[0]]
        return ValidationResult(
            valid=False,
            reason=f"not enough {piece} in inventory to build this construction",
        )
    if short_connectors:
        kind = construction.connectors[short_connectors[0]].connector_type.value
        return ValidationResult(
            valid=False,
            reason=f"not enough '{kind}' connectors in inventory",
        )

    gate_count = sum(1 for p in construction.pieces if p.is_gate)
    if gate_count < 1:
        return ValidationResult(
            valid=False, reason="enclosure must have at least one gate"
        )

    polygon = Polygon(vertices)
    if not polygon.is_valid or not polygon.exterior.is_simple:
        return ValidationResult(valid=False, reason="enclosure walls cross themselves")
    if polygon.area <= 0:
        return ValidationResult(valid=False, reason="enclosure has zero area")

    if not room.contains_enclosure(polygon, tol=tol):
        return ValidationResult(
            valid=False, reason="enclosure does not fit within the room"
        )

    faces = compute_faces(construction)
    overlong = [f for f in faces if f.length > c.MAX_FACE_LENGTH + tol]
    if overlong:
        return ValidationResult(
            valid=False,
            reason=(
                f"a linear face is {overlong[0].length:.1f} units long, "
                f"exceeds the {c.MAX_FACE_LENGTH}-unit limit"
            ),
        )

    gate_faces = {id(f) for f in faces if f.has_gate}
    second_gate_bonus = len(gate_faces) >= 2

    return ValidationResult(
        valid=True,
        reason="valid",
        polygon=polygon,
        vertices=vertices,
        area=polygon.area,
        perimeter=polygon.length,
        gate_count=gate_count,
        faces=faces,
        second_gate_bonus=second_gate_bonus,
    )


@dataclass(frozen=True)
class ScoreBreakdown:
    """The terms a score is made of; `total` is their sum.

    For a valid enclosure: baseline + A*area + C*perimeter (+ G if a second
    gate sits on a different face than the first). For an invalid one there
    is just the penalty, and every other term is zero."""

    valid: bool
    baseline: float
    area_term: float
    perimeter_term: float
    gate_bonus: float
    total: float


def explain_score(result: ValidationResult, weights: Weights) -> ScoreBreakdown:
    if not result.valid:
        return ScoreBreakdown(
            valid=False,
            baseline=0.0,
            area_term=0.0,
            perimeter_term=0.0,
            gate_bonus=0.0,
            total=c.INVALID_SCORE,
        )
    area_term = weights.A * result.area
    perimeter_term = weights.C * result.perimeter  # C is <= 0, so this subtracts
    gate_bonus = weights.G if result.second_gate_bonus else 0.0
    return ScoreBreakdown(
        valid=True,
        baseline=c.BASELINE_SCORE,
        area_term=area_term,
        perimeter_term=perimeter_term,
        gate_bonus=gate_bonus,
        total=c.BASELINE_SCORE + area_term + perimeter_term + gate_bonus,
    )


def score_construction(result: ValidationResult, weights: Weights) -> float:
    """A correct solution starts at 1000 points, +A*area, +C*perimeter (C is
    negative), +G if a second gate sits on a different face than the first.
    An incorrect one scores -1000. See `explain_score` for the terms."""
    return explain_score(result, weights).total
