"""Approach (b): a small library of pre-validated buildable shapes
(rectangle, regular octagon), each grown as large as fits and shrunk when the
inventory is short. Every candidate is scored with the real objective and the
best-scoring valid placement wins. Robust: never returns an invalid loop, and
returns None (0 points) rather than a sub-zero build.
"""

import math

from shapely.geometry import Point

from players.player0 import Player
from src.enclosure import Construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

HEADINGS = list(range(0, 180, 15))
GRID_STEPS = 8
MAX_TRY = 400


class Player9(Player):
    def build_enclosure(self) -> Construction | None:
        # primary shapes (rectangles, tiled rectangles, octagons): proven, fast,
        # score-driven. if any places, it wins -- this pass is unchanged so it
        # can never regress.
        cands = self._rectangles() + self._tiled_rectangles() + self._octagons()
        cands.sort(key=lambda c: c[0], reverse=True)
        for score, pieces, conns in cands[:MAX_TRY]:
            if score <= 0:
                break
            cons = self._place(pieces, conns)
            if cons is not None:
                self.note = f"(b) {len(pieces)}-piece shape score {score:.1f}"
                return cons

        # fallback ONLY if nothing above placed: room-aware L-shapes for
        # non-convex rooms. runs last, so it can only turn a 0 into something.
        cons = self._l_shape_fallback()
        if cons is not None:
            return cons
        return None

    def _score_poly(self, area: float, perim: float, two_gates: bool) -> float:
        g = self.weights.G if two_gates else 0.0
        return 1000.0 + self.weights.A * area + self.weights.C * perim + g

    # ---- room-aware L-shape fallback (non-convex rooms) ----
    def _l_shape_fallback(self) -> Construction | None:
        """Try L-shaped enclosures oriented to the room's concave corners. Only
        called when no primary shape placed, so it can only improve a 0. An L
        has sides [X, Y, nx, ny, X-nx, Y-ny] with a reflex (270) corner at
        vertex 4; it fills an L / notched room better than a rectangle."""
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        if not gl:
            return None

        anchors = self._concave_anchors()
        if not anchors:
            return None

        # a bounded set of L outer-box / notch sizes, built from inventory via
        # tiling. coarse steps keep this fast.
        best: tuple[float, Construction] | None = None
        for X in range(10, 31, 3):
            for Y in range(10, 31, 3):
                for nx in range(5, X - 4, 3):
                    for ny in range(5, Y - 4, 3):
                        sides = [X, Y, nx, ny, X - nx, Y - ny]
                        if any(s < 5 or s > 30 for s in sides):
                            continue
                        for two in (True, False):
                            built = self._l_pieces(sides, two)
                            if built is None:
                                continue
                            pieces, conns = built
                            area = X * Y - nx * ny
                            score = self._score_poly(area, sum(sides), two)
                            if score <= 0:
                                continue
                            cons = self._place_at_anchors(pieces, conns, anchors)
                            if cons is not None and (best is None or score > best[0]):
                                best = (score, cons)
                                self.note = f"(b) L-shape {sides} score {score:.1f}"
        return best[1] if best else None

    def _concave_anchors(self) -> list[tuple[float, float]]:
        """Points near the room's reflex (concave) corners -- where an L's
        notch wants to sit -- plus the centroid as a fallback."""
        coords = list(self.room.polygon.exterior.coords)[:-1]
        n = len(coords)
        pts = []
        for i in range(n):
            ax, ay = coords[i]
            bx, by = coords[(i + 1) % n]
            cx, cy = coords[(i - 1) % n]
            # cross product of incoming->outgoing; sign flags a reflex corner
            cross = (ax - cx) * (by - ay) - (ay - cy) * (bx - ax)
            if cross < 0:  # reflex for a CCW polygon
                pts.append((ax, ay))
        cen = self.room.polygon.centroid
        pts.append((cen.x, cen.y))
        return pts

    def _place_at_anchors(self, pieces, conns, anchors) -> Construction | None:
        """Place a shape near each concave anchor over several headings."""
        for ax, ay in anchors:
            for hd in range(0, 360, 30):
                cons = self._at(pieces, conns, (ax, ay), hd)
                if validate_construction(cons, self.room, self.inventory).valid:
                    return cons
        return None

    def _l_pieces(
        self, sides: list[int], two: bool
    ) -> tuple[list[Piece], list[Connector]] | None:
        """Tile the six L sides from inventory. Reflex (270) corner at vertex 4,
        convex (90) elsewhere; straights within a side. Gate on side 0, second
        gate on side 2 (a different face) when `two`."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        gate_sides = {0, 2} if two else {0}
        pieces: list[Piece] = []
        conns: list[Connector] = []
        for si, side_len in enumerate(sides):
            corner = Connector(ConnectorType.RIGHT, reflex=(si == 4))
            seg_lengths: list[int] = []
            first_is_gate = False
            if si in gate_sides:
                gchoice = None
                for gl in sorted(
                    (L for L, n in gates.items() if n > 0 and 5 <= L <= 30),
                    reverse=True,
                ):
                    if gl <= side_len and (side_len - gl == 0 or side_len - gl >= 5):
                        gchoice = gl
                        break
                if gchoice is None:
                    return None
                gates[gchoice] -= 1
                seg_lengths.append(gchoice)
                first_is_gate = True
                rest = side_len - gchoice
                if rest > 0:
                    tail = self._tile(rest, walls)
                    if tail is None:
                        return None
                    seg_lengths.extend(tail)
            else:
                tiled = self._tile(side_len, walls)
                if tiled is None:
                    return None
                seg_lengths = tiled
            for k, L in enumerate(seg_lengths):
                if k == 0:
                    conns.append(corner)
                    is_gate = first_is_gate
                else:
                    conns.append(Connector(ConnectorType.STRAIGHT))
                    is_gate = False
                ptype = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(ptype, L))
        return pieces, conns

    @staticmethod
    def _tile(target: int, avail: dict[int, int]) -> list[int] | None:
        """Break `target` into a run of piece lengths available in `avail`
        (greedy largest-first), leaving no stub shorter than 5. Only lengths in
        [5, 30] are usable. Mutates `avail`. Returns the lengths, or None."""
        sizes = sorted(
            (L for L, n in avail.items() if n > 0 and 5 <= L <= 30), reverse=True
        )
        out: list[int] = []
        rem = target
        while rem > 0:
            for L in sizes:
                if L <= rem and avail.get(L, 0) > 0 and (rem - L == 0 or rem - L >= 5):
                    avail[L] -= 1
                    out.append(L)
                    rem -= L
                    break
            else:
                return None
        return out

    # ---- rectangles (4 right connectors) ----
    def _rectangles(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        wl = sorted(
            L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30
        )
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        out = []
        seen = set()
        for w in gl:
            for h in wl:
                for two in (True, False):
                    key = (w, h, two)
                    if key in seen:
                        continue
                    seen.add(key)
                    pieces = self._rect_pieces(w, h, two)
                    if pieces is None:
                        continue
                    score = self._score_poly(w * h, 2 * (w + h), two)
                    conns = [Connector(ConnectorType.RIGHT)] * 4
                    out.append((score, pieces, conns))
        return out

    def _rect_pieces(self, w, h, two) -> list[Piece] | None:
        scratch = self.inventory.copy()
        layout = [(True, w), (False, h), (two, w), (False, h)]
        pieces = []
        for gate, L in layout:
            p = Piece(PieceType.GATE if gate else PieceType.WALL, L)
            if not scratch.take_piece(p):
                return None
            pieces.append(p)
        return pieces

    # ---- tiled rectangles: each side is a run of collinear pieces joined by
    # straight connectors, so odd inventories (one piece per length) can still
    # build a big rectangle. each side is one linear face, so <= 30 units. ----
    def _tiled_rectangles(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        out = []
        # candidate side lengths: 5..30, both dimensions. keep it bounded so we
        # stay within the cpu budget. only emit when inventory can tile all four
        # sides and supply a gate.
        for w in range(5, 31):
            for h in range(w, 31):  # w <= h avoids duplicate orientations
                for two in (True, False):
                    built = self._tiled_rect_pieces(w, h, two)
                    if built is None:
                        continue
                    pieces, conns = built
                    score = self._score_poly(w * h, 2 * (w + h), two)
                    out.append((score, pieces, conns))
        return out

    def _tiled_rect_pieces(
        self, w: int, h: int, two: bool
    ) -> tuple[list[Piece], list[Connector]] | None:
        """A w x h rectangle whose four sides are tiled from inventory. A real
        corner (right connector) starts each side; straight connectors join the
        pieces within a side. Gate on side 0's first piece, and (if `two`) on
        the opposite side's first piece for the G bonus."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        sides = [w, h, w, h]
        gate_sides = {0, 2} if two else {0}

        pieces: list[Piece] = []
        conns: list[Connector] = []
        for si, side_len in enumerate(sides):
            want_gate = si in gate_sides
            # if this side carries a gate, spend one gate on its first segment,
            # then tile the rest from walls.
            seg_lengths: list[int] = []
            first_is_gate = False
            if want_gate:
                # pick the largest gate <= side_len that leaves a tileable rest
                gchoice = None
                for gl in sorted(
                    (L for L, n in gates.items() if n > 0 and 5 <= L <= 30),
                    reverse=True,
                ):
                    if gl <= side_len and (side_len - gl == 0 or side_len - gl >= 5):
                        gchoice = gl
                        break
                if gchoice is None:
                    return None
                gates[gchoice] -= 1
                seg_lengths.append(gchoice)
                first_is_gate = True
                rest = side_len - gchoice
                if rest > 0:
                    tail = self._tile(rest, walls)
                    if tail is None:
                        return None
                    seg_lengths.extend(tail)
            else:
                tiled = self._tile(side_len, walls)
                if tiled is None:
                    return None
                seg_lengths = tiled

            for k, L in enumerate(seg_lengths):
                if k == 0:
                    conns.append(Connector(ConnectorType.RIGHT))  # the real corner
                    is_gate = first_is_gate
                else:
                    conns.append(Connector(ConnectorType.STRAIGHT))  # within the side
                    is_gate = False
                ptype = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(ptype, L))

        return pieces, conns

    # ---- regular octagons (8 diagonal connectors; all sides equal length s) ----
    def _octagons(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        if not gl:
            return []
        out = []
        for s in sorted(
            {L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30}
            | set(gl)
        ):
            # need 1 or 2 gates of length s + remaining walls of length s
            for two in (True, False):
                pieces = self._oct_pieces(s, two)
                if pieces is None:
                    continue
                # regular octagon with side s: area = 2(1+sqrt2) s^2, perim = 8s
                area = 2 * (1 + math.sqrt(2)) * s * s
                perim = 8 * s
                score = self._score_poly(area, perim, two)
                conns = [Connector(ConnectorType.DIAGONAL)] * 8
                out.append((score, pieces, conns))
        return out

    def _oct_pieces(self, s, two) -> list[Piece] | None:
        scratch = self.inventory.copy()
        pieces = []
        for i in range(8):
            # gates on side 0 and side 4 (opposite -> different face) when two
            gate = (i == 0) or (two and i == 4)
            p = Piece(PieceType.GATE if gate else PieceType.WALL, s)
            if not scratch.take_piece(p):
                return None
            pieces.append(p)
        return pieces

    # ---- placement: centroid then grid of anchors, several headings ----
    def _place(self, pieces, conns) -> Construction | None:
        cen = self.room.polygon.centroid
        for hd in HEADINGS:
            cons = self._at(pieces, conns, (cen.x, cen.y), hd)
            if validate_construction(cons, self.room, self.inventory).valid:
                return cons
        minx, miny, maxx, maxy = self.room.polygon.bounds
        xs = [minx + (maxx - minx) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
        ys = [miny + (maxy - miny) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
        for x in xs:
            for y in ys:
                if not self.room.polygon.contains(Point(x, y)):
                    continue
                for hd in HEADINGS:
                    cons = self._at(pieces, conns, (x, y), hd)
                    if validate_construction(cons, self.room, self.inventory).valid:
                        return cons
        return None

    def _at(self, pieces, conns, center, heading_deg) -> Construction:
        # place so the shape's centroid lands near `center`: walk from origin,
        # find centroid, then offset start. cheap + good enough for search.
        pts = [(0.0, 0.0)]
        hd = heading_deg
        # headings per piece from the turn sequence
        headings = [heading_deg]
        for i in range(1, len(pieces)):
            hd = hd + conns[i].turn_angle()
            headings.append(hd)
        for i, p in enumerate(pieces):
            th = math.radians(headings[i])
            x, y = pts[-1]
            pts.append((x + p.length * math.cos(th), y + p.length * math.sin(th)))
        cx = sum(x for x, _ in pts[:-1]) / len(pieces)
        cy = sum(y for _, y in pts[:-1]) / len(pieces)
        start = (center[0] - cx, center[1] - cy)
        return Construction(start, heading_deg, pieces, conns)
