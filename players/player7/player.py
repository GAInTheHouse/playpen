import math
from collections import Counter

from shapely.geometry import Point

import src.constants as c
from players.player0 import Player
from src.enclosure import (
    Construction,
    score_construction,
    validate_construction,
    walk_points,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

GRID_STEPS = 8
MAX_SHAPES = 500
RIGHT = [Connector(ConnectorType.RIGHT)] * 4
DIAGONAL = [Connector(ConnectorType.DIAGONAL)] * 8


def _vertices(pieces, connectors):
    base = Construction((0.0, 0.0), 0.0, pieces, connectors)
    return walk_points(base)[:-1]


def _area(points):
    n = len(points)
    s = sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
        for i in range(n)
    )
    return abs(s) / 2.0


def _centroid(points):
    n = len(points)
    a = cx = cy = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)
    return (cx / (3.0 * a), cy / (3.0 * a))


class Player7(Player):
    """Build the largest valid rectangle and the largest valid octagon, then
    keep whichever scores higher (or None)."""

    def build_enclosure(self) -> Construction | None:
        best, best_score = None, None
        for name, construction in (
            ("rectangle", self._best(self._rectangles(), range(0, 180, 15))),
            ("octagon", self._best(self._octagons(), range(0, 45, 5))),
        ):
            if construction is None:
                continue
            construction = self._add_second_gate(construction)
            result = validate_construction(construction, self.room, self.inventory)
            score = score_construction(result, self.weights)
            if best_score is None or score > best_score:
                best, best_score, best_name = construction, score, name
        self.note = f"{best_name} ({best_score:.0f})" if best else "nothing fits"
        return best

    def _add_second_gate(self, construction):
        """If G rewards it, turn one wall into a second gate (a different face,
        since every edge is its own face here) when the inventory has one spare."""
        if self.weights.G <= 0 or sum(p.is_gate for p in construction.pieces) >= 2:
            return construction
        for i, piece in enumerate(construction.pieces):
            if piece.is_gate:
                continue
            pieces = list(construction.pieces)
            pieces[i] = Piece(PieceType.GATE, piece.length)
            trial = Construction(
                construction.start,
                construction.start_heading,
                pieces,
                construction.connectors,
            )
            if validate_construction(trial, self.room, self.inventory).valid:
                return trial
        return construction

    def _take(self, lengths, gate_index):
        """Reserve these pieces from a copy of the inventory (one a gate), or
        None if it can't supply them."""
        inv = self.inventory.copy()
        pieces = []
        for i, length in enumerate(lengths):
            piece = Piece(PieceType.GATE if i == gate_index else PieceType.WALL, length)
            if not inv.take_piece(piece):
                return None
            pieces.append(piece)
        return pieces

    def _rectangles(self):
        if self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
            return []
        out = []
        for gate in self.inventory.gates:
            for wall in self.inventory.walls:
                pieces = self._take((gate, wall, gate, wall), 0)
                if pieces:
                    out.append((self._value(pieces, RIGHT), pieces, RIGHT))
        return self._biggest(out)

    def _octagons(self):
        if self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 8:
            return []
        # an octagon closes only if opposite sides are equal, so it is four
        # matched pair-lengths, laid out [h, d1, v, d2, h, d1, v, d2]
        pairs = sorted(
            length
            for length in set(self.inventory.walls) | set(self.inventory.gates)
            if length <= c.MAX_FACE_LENGTH
            and self.inventory.walls.get(length, 0)
            + self.inventory.gates.get(length, 0)
            >= 2
        )
        out = []
        for h in pairs:
            for d1 in pairs:
                for v in pairs:
                    for d2 in pairs:
                        lengths = (h, d1, v, d2, h, d1, v, d2)
                        for gate in range(4):
                            pieces = self._take(lengths, gate)
                            if pieces:
                                out.append(
                                    (self._value(pieces, DIAGONAL), pieces, DIAGONAL)
                                )
                                break
        return self._biggest(out)

    def _value(self, pieces, connectors):
        """Estimated score of this shape: area and perimeter terms, plus the
        gate bonus if the inventory could supply a second gate for it."""
        area = _area(_vertices(pieces, connectors))
        perimeter = sum(p.length for p in pieces)
        counts = Counter(p.length for p in pieces)
        max_gates = sum(
            min(n, self.inventory.gates.get(length, 0)) for length, n in counts.items()
        )
        bonus = self.weights.G if self.weights.G > 0 and max_gates >= 2 else 0.0
        return self.weights.A * area + self.weights.C * perimeter + bonus

    def _biggest(self, candidates):
        candidates.sort(key=lambda s: s[0], reverse=True)
        return candidates[:MAX_SHAPES]

    def _best(self, candidates, orientations):
        """First candidate (biggest first) that the simulator accepts somewhere
        in the room, placed by its centroid."""
        for _area_, pieces, connectors in candidates:
            cx, cy = _centroid(_vertices(pieces, connectors))
            for center in self._anchors():
                for heading in orientations:
                    theta = math.radians(heading)
                    start = (
                        center[0] - (cx * math.cos(theta) - cy * math.sin(theta)),
                        center[1] - (cx * math.sin(theta) + cy * math.cos(theta)),
                    )
                    construction = Construction(start, heading, pieces, connectors)
                    if validate_construction(
                        construction, self.room, self.inventory
                    ).valid:
                        return construction
        return None

    def _anchors(self):
        polygon = self.room.polygon
        yield (polygon.centroid.x, polygon.centroid.y)
        minx, miny, maxx, maxy = polygon.bounds
        for i in range(GRID_STEPS):
            for j in range(GRID_STEPS):
                x = minx + (maxx - minx) * i / (GRID_STEPS - 1)
                y = miny + (maxy - miny) * j / (GRID_STEPS - 1)
                if polygon.contains(Point(x, y)):
                    yield (x, y)
