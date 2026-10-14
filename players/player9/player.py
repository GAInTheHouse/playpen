"""Player 9 — build the biggest, cheapest enclosure that fits in the room.

The score rewards area and punishes perimeter, so we try a few strategies and
keep whichever one actually scores best (checked with the real scorer):

1. Shape search: try rectangles, hexagons, octagons and L-shapes at many
   integer sizes. For each one, work out how to build its sides from the
   pieces we have (without running out of straight connectors), then try to
   place it in the room at lots of angles and positions. Octagons usually win
   in open rooms because they pack in a lot of area.

2. Boundary hug: for odd / concave rooms, follow the room's own outline just
   inside the walls. The room shape itself holds the most area, so hugging it
   beats any fixed shape when we have enough pieces to go all the way around.

3. Safety net: a small library of known-good shapes, so we never do worse than
   before even when the fancier strategies come up empty.

We spend the available CPU time searching and return the best valid result.
Some ideas here were inspired by other groups' write-ups (group 2's side
allocation and group 10's edge hugging) and a teammate's wall-alignment idea.
"""

import math
import time
from functools import lru_cache
from itertools import product

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, TOL
from src.enclosure import (
    Construction,
    score_construction,
    validate_construction,
    walk_points,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

HEADINGS = list(range(0, 180, 15))
GRID_STEPS = 8
MAX_TRY = 400
# stay well under the simulator's 300s CPU limit; the search returns the best
# found so far when the budget runs out, so this is a soft cap on effort.
SEARCH_CPU_SECONDS = 55.0
ALLOCATION_NODE_LIMIT = 20000


class Player9(Player):
    def build_enclosure(self) -> Construction | None:
        deadline = time.process_time() + SEARCH_CPU_SECONDS
        best = None
        best_score = 0.0  # a no-build (0) beats any negative-scoring loop
        straight = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonal = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)

        # ---- piece stock as parallel (types, counts) lists ----
        types: list[Piece] = []
        stock: list[int] = []
        for kind, pool in (
            (PieceType.WALL, self.inventory.walls),
            (PieceType.GATE, self.inventory.gates),
        ):
            for length, count in sorted(pool.items()):
                if count > 0 and length <= MAX_FACE_LENGTH:
                    types.append(Piece(kind, length))
                    stock.append(count)
        if not any(p.is_gate for p in types):
            self.note = "No usable gate (length <= 30) -- no valid enclosure."
            return None

        # ---- enumerate faces: tiled runs of pieces (<= straight+1 pieces) ----
        # faces[length] = list of (usage_tuple, joins, has_gate)
        faces: dict[int, list] = {}
        counts = [0] * len(types)

        def enumerate_faces(first, length, number, has_gate):
            if time.process_time() >= deadline:
                return
            if number:
                faces.setdefault(length, []).append(
                    (tuple(counts), number - 1, has_gate)
                )
            if number >= min(6, straight + 1):
                return
            for i in range(first, len(types)):
                nl = length + types[i].length
                if nl > MAX_FACE_LENGTH or counts[i] >= stock[i]:
                    continue
                counts[i] += 1
                enumerate_faces(i, nl, number + 1, has_gate or types[i].is_gate)
                counts[i] -= 1

        enumerate_faces(0, 0, 0, False)
        if not faces:
            self.note = "No face can be built from the inventory."
            return None
        # prefer fewer joins (fewer straight connectors) and gate-bearing faces
        for options in faces.values():
            options.sort(key=lambda o: (o[1], not o[2]))

        # ---- work out how to build each side from the pieces we have ----
        def allocate(lengths, required_gate_faces):
            nodes = 0

            @lru_cache(maxsize=ALLOCATION_NODE_LIMIT)
            def search(side, remaining, avail_straight, gate_faces):
                nonlocal nodes
                nodes += 1
                if nodes > ALLOCATION_NODE_LIMIT or time.process_time() >= deadline:
                    return None
                if side == len(lengths):
                    return () if gate_faces >= required_gate_faces else None
                if gate_faces + len(lengths) - side < required_gate_faces:
                    return None
                for usage, joins, has_gate in faces.get(lengths[side], ()):
                    if joins > avail_straight:
                        continue
                    if any(u > r for u, r in zip(usage, remaining)):
                        continue
                    rest = search(
                        side + 1,
                        tuple(r - u for u, r in zip(usage, remaining)),
                        avail_straight - joins,
                        min(required_gate_faces, gate_faces + int(has_gate)),
                    )
                    if rest is not None:
                        return (usage,) + rest
                return None

            return search(0, tuple(stock), straight, 0)

        # ---- room geometry + placement anchors ----
        room = self.room.polygon
        buffered = prep(room.buffer(TOL))
        corners = list(room.exterior.coords)[:-1]
        minx, miny, maxx, maxy = room.bounds
        room_diag = math.hypot(maxx - minx, maxy - miny)

        headings = set(range(0, 180, 15))
        for p, q in zip(corners, corners[1:] + corners[:1]):
            ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            headings.add(round(ang % 180, 9))
            headings.add(round((ang + 90) % 180, 9))

        centers = [(room.centroid.x, room.centroid.y)]
        rp = room.representative_point()
        centers.append((rp.x, rp.y))
        for i in range(1, 10):
            for j in range(1, 10):
                centers.append((minx + (maxx - minx) * i / 10,
                                miny + (maxy - miny) * j / 10))

        def placement(vertices):
            local = Polygon(vertices)
            center = local.centroid
            angles = set(range(0, 360, 15))
            for h in headings:
                for off in range(0, 360, 45):
                    angles.add(round((h + off) % 360, 9))
            for h in sorted(angles):
                if time.process_time() >= deadline:
                    return None
                th = math.radians(h)
                co, si = math.cos(th), math.sin(th)
                offs = [(x * co - y * si, x * si + y * co) for x, y in vertices]
                cx = center.x * co - center.y * si
                cy = center.x * si + center.y * co
                # corner-start placements first (notes: corner starts give room)
                starts = [(px - ox, py - oy) for px, py in corners for ox, oy in offs]
                starts.extend((x - cx, y - cy) for x, y in centers)
                for x, y in starts:
                    if time.process_time() >= deadline:
                        return None
                    poly = Polygon([(x + dx, y + dy) for dx, dy in offs])
                    if buffered.contains(poly):
                        return (x, y), h
            return None

        # ---- generate template families, round-robin, ranked by true score ----
        lengths_avail = sorted(faces)

        def templates():
            gens = []
            if right >= 4:
                gens.append(("rectangle", (a, b, a, b))
                            for a, b in product(lengths_avail, repeat=2) if a <= b)
            if right >= 2 and diagonal >= 4:
                gens.append(("hexagon", (a, b, c, a, b, c))
                            for a, b, c in product(lengths_avail, repeat=3))
            if diagonal >= 8:
                gens.append(("octagon", (a, b, c, d, a, b, c, d))
                            for a, b, c, d in product(lengths_avail, repeat=4)
                            if a <= c)
            if right >= 6:
                gens.append(("L-shape", (a + b, c, b, d, a, c + d))
                            for a, b, c, d in product(lengths_avail, repeat=4)
                            if a + b in faces and c + d in faces)
            gens = [g for g in gens]
            while gens:
                for g in gens[:]:
                    try:
                        yield next(g)
                    except StopIteration:
                        gens.remove(g)

        available_perimeter = sum(p.length * n for p, n in zip(types, stock))
        rw, rh = maxx - minx, maxy - miny
        room_bb_lo, room_bb_hi = min(rw, rh), max(rw, rh)
        candidates = []
        gen_deadline = time.process_time() + max(0.5, (deadline - time.process_time()) * 0.2)
        for family, sides in templates():
            if time.process_time() >= gen_deadline:
                break
            if sum(sides) > available_perimeter:
                continue
            area = self._template_area(family, sides)
            if area > room.area + TOL * room.length:
                continue
            # skip shapes too big to ever fit the room (even rotated), so we
            # don't waste time trying to place giant octagons that can't fit.
            verts0 = self._integer_vertices(sides, self._template_turns(family))
            if verts0 is None:
                continue
            xs = [v[0] for v in verts0]
            ys = [v[1] for v in verts0]
            sw, sh = max(xs) - min(xs), max(ys) - min(ys)
            s_lo, s_hi = min(sw, sh), max(sw, sh)
            if s_lo > room_bb_lo + TOL or s_hi > room_bb_hi + TOL:
                continue
            base = BASELINE_SCORE + self.weights.A * area + self.weights.C * sum(sides)
            if base + self.weights.G > 0:
                candidates.append((base + self.weights.G, base, family, sides))
        candidates.sort(reverse=True)

        tried = 0
        for upper, base, family, sides in candidates:
            if time.process_time() >= deadline or upper <= best_score:
                break
            tried += 1
            turns = self._template_turns(family)
            vertices = self._integer_vertices(sides, turns)
            if vertices is None:
                continue
            shape = Polygon(vertices)
            if not shape.is_valid or shape.area <= 0:
                continue
            if any(math.hypot(x - xx, y - yy) > room_diag + TOL
                   for x, y in vertices for xx, yy in vertices):
                continue
            assemblies = None
            if self.weights.G > 0:
                assemblies = allocate(sides, 2)
            if assemblies is None and base > best_score:
                assemblies = allocate(sides, 1)
            if assemblies is None:
                continue
            bonus = sum(any(types[i].is_gate and n for i, n in enumerate(face))
                        for face in assemblies) >= 2
            if base + (self.weights.G if bonus else 0) <= best_score:
                continue
            found = placement(vertices)
            if found is None:
                continue
            pieces, conns = [], []
            for usage, turn in zip(assemblies, turns):
                face = [p for p, n in zip(types, usage) for _ in range(n)]
                pieces.extend(face)
                conns.append(Connector(
                    ConnectorType.RIGHT if abs(turn) == 90 else ConnectorType.DIAGONAL,
                    reflex=turn < 0))
                conns.extend(Connector(ConnectorType.STRAIGHT) for _ in face[1:])
            start, heading = found
            cons = Construction(start, heading, pieces, conns)
            res = validate_construction(cons, self.room, self.inventory)
            if res.valid:
                sc = score_construction(res, self.weights)
                if sc > best_score:
                    best, best_score = cons, sc
                    self.note = f"(a) {family} score {sc:.1f}"

        # ---- boundary hug: inset the room and trace its actual outline. this
        # follows arbitrary non-convex rooms (the room's own shape is the most
        # area-efficient enclosure), so it is usually the big winner. ----
        hug = self._boundary_hug()
        if hug is not None:
            hs = self._score_cons(hug)
            if hs > best_score:
                best, best_score = hug, hs
                self.note = f"(a) boundary-hug score {hs:.1f}"

        # ---- extra family: trace the room outline (helps concave rooms) ----
        traced = self._room_traced()
        if traced is not None:
            ts = self._score_cons(traced)
            if ts > best_score:
                best, best_score = traced, ts
                self.note = f"(a) room-traced score {ts:.1f}"

        # ---- safety net: the proven template library + wall-aligned placement.
        # covers rooms the parameterized search misses (odd concave shapes). it
        # competes on true score, so it can only raise the result, never lower.
        legacy = self._legacy_search(best_score)
        if legacy is not None:
            lcons, lscore = legacy
            if lscore > best_score:
                best, best_score = lcons, lscore
                self.note = f"(b) legacy shape score {lscore:.1f}"

        if best is None:
            self.note = "No valid positive-scoring enclosure found."
        return best

    def _legacy_search(self, best_score: float):
        """Proven shape library (rectangles, tiled rectangles, octagons) placed
        with the wall-aligned `_place`, then the room-aware L fallback. Returns
        (construction, true_score) for the best that beats `best_score`, or None."""
        cands = self._rectangles() + self._tiled_rectangles() + self._octagons()
        cands.sort(key=lambda c: c[0], reverse=True)
        best = None
        bscore = best_score
        for score, pieces, conns in cands[:MAX_TRY]:
            if score <= bscore:
                break
            cons = self._place(pieces, conns)
            if cons is not None:
                sc = self._score_cons(cons)
                if sc > bscore:
                    best, bscore = cons, sc
                    break
        if best is None:
            fb = self._l_shape_fallback()
            if fb is not None:
                sc = self._score_cons(fb)
                if sc > best_score:
                    best, bscore = fb, sc
        return (best, bscore) if best is not None else None

    @staticmethod
    def _template_turns(family):
        return {
            "rectangle": (90, 90, 90, 90),
            "hexagon": (90, 45, 45, 90, 45, 45),
            "octagon": (45,) * 8,
            "L-shape": (90, 90, 90, -90, 90, 90),
        }[family]

    @staticmethod
    def _template_area(family, sides):
        if family == "rectangle":
            return sides[0] * sides[1]
        if family == "hexagon":
            a, b, c = sides[:3]
            return a * c + (a + c) * b / math.sqrt(2)
        if family == "octagon":
            a, b, c, d = sides[:4]
            return a * c + (a + c) * (b + d) / math.sqrt(2) + b * d
        width, c, b, d, a, _ = sides  # L-shape
        return width * c + a * d

    @staticmethod
    def _integer_vertices(lengths, turns):
        """Check the shape actually closes back to its start, then return its
        corner points. Returns None if the sides don't form a closed loop."""
        if (len(lengths) != len(turns) or sum(turns) != 360
                or any(not isinstance(n, int) or not 5 <= n <= MAX_FACE_LENGTH
                       for n in lengths)):
            return None
        ax = ay = dx_ = dy_ = 0
        direction = 0
        verts = []
        dirs = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        for i, L in enumerate(lengths):
            if i:
                direction = (direction + turns[i] // 45) % 8
            verts.append((ax + dx_ / math.sqrt(2), ay + dy_ / math.sqrt(2)))
            ddx, ddy = dirs[direction]
            if direction % 2:
                dx_ += L * ddx
                dy_ += L * ddy
            else:
                ax += L * ddx
                ay += L * ddy
        if (ax, ay, dx_, dy_) != (0, 0, 0, 0):
            return None
        return verts

    def _score_cons(self, cons: Construction) -> float:
        res = validate_construction(cons, self.room, self.inventory)
        if not res.valid:
            return float("-inf")
        return score_construction(res, self.weights)

    def _score_poly(self, area: float, perim: float, two_gates: bool) -> float:
        g = self.weights.G if two_gates else 0.0
        return 1000.0 + self.weights.A * area + self.weights.C * perim + g


    def _connector_for_angle(self, interior: float):
        """Map a corner interior angle (deg) to the connector that reproduces it."""
        tol = 2.0
        table = [
            (90.0, Connector(ConnectorType.RIGHT, False)),
            (270.0, Connector(ConnectorType.RIGHT, True)),
            (135.0, Connector(ConnectorType.DIAGONAL, False)),
            (225.0, Connector(ConnectorType.DIAGONAL, True)),
            (180.0, Connector(ConnectorType.STRAIGHT, False)),
        ]
        for ang, conn in table:
            if abs(interior - ang) <= tol:
                return conn
        return None

    def _room_traced(self) -> Construction | None:
        """Follow the room's own boundary. The room outline is the most
        area-efficient enclosure, so for any room whose corners are 45-degree
        multiples we trace it (optionally inset inward so it sits strictly
        inside), tiling each edge from a shared inventory and matching each
        corner's connector. Material is usually plentiful, so we try the FULL
        outline first, then progressively larger insets only if needed."""
        from shapely.geometry import Polygon as _Poly

        base_poly = self.room.polygon
        # try the exact outline first, then inward buffers (keeps the shape,
        # unlike scaling toward the centroid which distorts a concave room).
        for inset in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            poly = base_poly if inset == 0.0 else base_poly.buffer(-inset,
                                                                   join_style=2)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            built = self._trace_polygon(poly)
            if built is None:
                continue
            start, heading, pieces, conns = built
            cons = Construction(start, heading, pieces, conns)
            if validate_construction(cons, self.room, self.inventory).valid:
                return cons
        return None

    def _trace_polygon(self, poly) -> tuple | None:
        """Build pieces+connectors that trace `poly`'s boundary exactly. Returns
        (start, heading, pieces, connectors) or None if it can't be tiled from
        inventory within the connector budgets."""
        pts = list(poly.exterior.coords)[:-1]
        n = len(pts)
        if n < 3:
            return None
        # CCW winding so turns sum to +360
        signed = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                     for i in range(n))
        if signed < 0:
            pts = pts[::-1]

        hd = [math.degrees(math.atan2(pts[(i + 1) % n][1] - pts[i][1],
                                      pts[(i + 1) % n][0] - pts[i][0])) for i in range(n)]
        # corner connector = turn INTO edge i (at vertex i)
        corner_conns = []
        corner_budget = {
            ConnectorType.RIGHT: self.inventory.connectors.get(ConnectorType.RIGHT, 0),
            ConnectorType.DIAGONAL: self.inventory.connectors.get(ConnectorType.DIAGONAL, 0),
            ConnectorType.STRAIGHT: self.inventory.connectors.get(ConnectorType.STRAIGHT, 0),
        }
        for i in range(n):
            turn = (hd[i] - hd[(i - 1) % n] + 180.0) % 360.0 - 180.0
            internal = round((180.0 - turn) % 360.0)
            conn = self._connector_for_angle(internal)
            if conn is None:
                return None  # non-45 corner
            corner_conns.append(conn)
            corner_budget[conn.connector_type] -= 1
            if corner_budget[conn.connector_type] < 0:
                return None  # not enough corner connectors of this type

        # each edge's buildable PIECE length: axis edge = geometric length;
        # diagonal (45-deg) edge = geometric length / sqrt(2) (its integer leg).
        sq = math.sqrt(2)
        edges = []
        for i in range(n):
            dx = pts[(i + 1) % n][0] - pts[i][0]
            dy = pts[(i + 1) % n][1] - pts[i][1]
            glen = math.hypot(dx, dy)
            h = round(math.degrees(math.atan2(dy, dx)) % 90, 3)
            is_diag = abs(h - 45) < 1.0
            piece_len = glen / sq if is_diag else glen
            edges.append(piece_len)
        # reject if any edge isn't close to an integer buildable length
        if any(abs(e - round(e)) > 0.05 for e in edges):
            return None
        # remaining STRAIGHT connectors available to JOIN pieces within edges
        straight_budget = corner_budget[ConnectorType.STRAIGHT]

        built = self._tile_outline_global(edges, corner_conns, straight_budget)
        if built is None:
            return None
        pieces, conns = built
        return (pts[0][0], pts[0][1]), hd[0], pieces, conns

    def _boundary_hug(self) -> Construction | None:
        """Follow the room's own outline, just inside the walls. The room shape
        holds the most area, so hugging it beats any fixed shape on odd rooms.
        We pull the walls in by a few amounts (so the fence sits inside), break
        up any wall run longer than 30 with a small inward notch, then build and
        place it. We keep the best size that actually fits our pieces."""
        import time as _t
        pts0 = list(self.room.polygon.exterior.coords)[:-1]
        n0 = len(pts0)
        if n0 < 3:
            return None
        if sum(pts0[i][0] * pts0[(i + 1) % n0][1] - pts0[(i + 1) % n0][0] * pts0[i][1]
               for i in range(n0)) < 0:
            pts0 = pts0[::-1]

        best = None
        best_score = 0.0
        deadline = _t.process_time() + 30.0
        sq = math.sqrt(2)
        for d in (1, 2, 3, 4, 5, 6, 7, 8, 2.5, 3.5, 4.5, 1.5):  # inset depths
            if _t.process_time() >= deadline:
                break
            verts = self._inset_integer(pts0, d, d / sq)
            if verts is None:
                continue
            verts = self._split_long_faces(verts, d)
            if verts is None or len(verts) < 3:
                continue
            try:
                hp = Polygon(verts)
            except Exception:
                continue
            if not hp.is_valid or hp.area <= 0:
                continue
            if not self.room.polygon.buffer(1e-6).contains(hp):
                continue
            built = self._trace_polygon(hp)
            if built is None:
                continue
            start, heading, pieces, conns = built
            cons = Construction(start, heading, pieces, conns)
            res = validate_construction(cons, self.room, self.inventory)
            if res.valid:
                sc = score_construction(res, self.weights)
                if sc > best_score:
                    best, best_score = cons, sc
        return best

    @staticmethod
    def _inset_integer(pts, d_axis, d_diag):
        """Pull every wall inward along its own normal (straight walls by
        d_axis, diagonal walls by d_diag) and reconnect the corners. The amounts
        are chosen so the new corners still land on buildable points."""
        n = len(pts)
        lines = []
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-9:
                return None
            nx, ny = -dy / L, dx / L  # inward normal for CCW
            h = round(math.degrees(math.atan2(dy, dx)) % 90)
            off = d_diag if abs(h - 45) < 1 else d_axis
            lines.append(((ax + nx * off, ay + ny * off), (dx, dy)))

        def inter(p1, d1, p2, d2):
            x1, y1 = p1
            x2, y2 = p2
            dx1, dy1 = d1
            dx2, dy2 = d2
            den = dx1 * dy2 - dy1 * dx2
            if abs(den) < 1e-9:
                return None
            t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / den
            return (x1 + dx1 * t, y1 + dy1 * t)

        verts = []
        for i in range(n):
            v = inter(lines[(i - 1) % n][0], lines[(i - 1) % n][1],
                      lines[i][0], lines[i][1])
            if v is None:
                return None
            verts.append(v)
        return verts

    @staticmethod
    def _snap_lattice(x, y):
        """Snap a point to the nearest Z + Z/sqrt(2) lattice point (coordinates
        of a 45-degree integer-length walk). Keeps the enclosure buildable."""
        sq = math.sqrt(2)
        best = None
        for dx in range(-60, 61):
            ax = round(x - dx / sq)
            px = ax + dx / sq
            for dy in range(-60, 61):
                ay = round(y - dy / sq)
                py = ay + dy / sq
                e = (px - x) ** 2 + (py - y) ** 2
                if best is None or e < best[0]:
                    best = (e, px, py)
                    if e < 1e-9:
                        return (px, py)
        return (best[1], best[2])

    def _split_long_faces(self, verts, inset):
        """Insert an inward notch wherever a straight run exceeds 30 units so no
        single face is too long. The notch steps 5 units inward, runs forward a
        little, then 5 units back out -- all legs >= the 5-unit min piece length
        and at 90-degree (right) corners, so every piece stays buildable."""
        n = len(verts)
        if n < 3:
            return verts
        DEPTH = 5.0
        out = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            out.append((ax, ay))
            L = math.hypot(bx - ax, by - ay)
            if L <= 30 + 1e-6:
                continue
            ux, uy = (bx - ax) / L, (by - ay) / L
            nx, ny = -uy, ux  # inward normal (interior on left, CCW)
            segs = int(math.ceil(L / 28.0))
            step = L / segs
            if step < 2 * DEPTH + 5:
                continue  # too short to notch cleanly; leave as-is
            # at each cut point insert a rectangular inward bite: up DEPTH,
            # forward DEPTH along the edge, back down DEPTH -- all 90-deg legs.
            for k in range(1, segs):
                cx, cy = ax + ux * step * k, ay + uy * step * k
                a1 = (round(cx + nx * DEPTH), round(cy + ny * DEPTH))
                a2 = (round(cx + ux * DEPTH + nx * DEPTH),
                      round(cy + uy * DEPTH + ny * DEPTH))
                out.append((round(cx), round(cy)))          # arrive on the edge
                out.append(a1)                               # 90-deg turn inward
                out.append(a2)                               # run forward inside
                # next appended edge-point drops back out at (cx+DEPTH*u) on edge
                out.append((round(cx + ux * DEPTH), round(cy + uy * DEPTH)))
        return out

    def _tile_outline_global(self, edge_lengths, corner_conns, straight_budget):
        """Tile all edges from one shared inventory, hardest edge first, each
        edge tiled with the FEWEST pieces (to conserve straight connectors).
        Keeps a running straight-connector budget. connectors[i] is the turn
        INTO piece i, so each edge's first piece carries its corner connector and
        the rest carry STRAIGHT. Places two gates on different faces if it can."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        n = len(edge_lengths)
        targets = [int(round(e)) for e in edge_lengths]
        if any(t < 5 for t in targets):
            return None
        gate_edges = {0}
        if sum(gates.values()) >= 2 and n >= 3:
            gate_edges.add(n // 2)

        order = sorted(range(n), key=lambda i: -targets[i])
        edge_runs = [None] * n
        used_straight = 0
        for ei in order:
            segs = self._tile_run(targets[ei], walls, gates, ei in gate_edges)
            if segs is None:
                return None
            used_straight += len(segs) - 1  # joins within this edge
            if used_straight > straight_budget:
                return None
            edge_runs[ei] = segs

        pieces: list[Piece] = []
        conns: list[Connector] = []
        for ei in range(n):
            for si, (ptype, L) in enumerate(edge_runs[ei]):
                pieces.append(Piece(ptype, L))
                conns.append(corner_conns[ei] if si == 0
                             else Connector(ConnectorType.STRAIGHT, False))
        if len(conns) != len(pieces):
            return None
        return pieces, conns

    def _tile_outline(self, edge_lengths, corner_conns):
        """Legacy per-edge tiler kept for the old L-fallback path."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        straights = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        n = len(edge_lengths)
        targets = [int(round(e)) for e in edge_lengths]
        if any(t < 5 for t in targets):
            return None
        gate_edges = {0}
        if sum(gates.values()) >= 2 and n >= 3:
            gate_edges.add(n // 2)
        # tile the MOST-CONSTRAINED edges first. short edges (5, 10) have the
        # fewest tilings, so allocate them before long edges grab all the small
        # walls. order: gate edges first, then ascending length.
        order = sorted(range(n), key=lambda i: (i not in gate_edges, targets[i]))
        edge_runs = [None] * n
        for ei in order:
            segs = self._tile_run(targets[ei], walls, gates, ei in gate_edges)
            if segs is None:
                return None
            edge_runs[ei] = segs
        pieces: list[Piece] = []
        conns: list[Connector] = []
        for ei in range(n):
            for si, (ptype, L) in enumerate(edge_runs[ei]):
                pieces.append(Piece(ptype, L))
                if si == 0:
                    conns.append(corner_conns[ei])
                else:
                    if straights <= 0:
                        return None
                    straights -= 1
                    conns.append(Connector(ConnectorType.STRAIGHT, False))
        if len(conns) != len(pieces):
            return None
        return pieces, conns

    def _tile_run(self, target, walls, gates, want_gate):
        """Tile a run of total length `target` using the FEWEST pieces possible,
        consuming from the walls/gates dicts (mutates them only on success).
        Fewest pieces == fewest STRAIGHT connectors spent joining them, which is
        the real scarce resource when tracing a whole room. Places one gate first
        if `want_gate`. Returns list of (PieceType, length) or None."""
        out = []
        remaining = target
        if want_gate:
            # prefer the gate that leaves the easiest-to-tile remainder; longest
            # first to keep piece count low.
            gchoice = next((L for L in sorted(gates, reverse=True)
                            if gates[L] > 0 and 5 <= L <= remaining
                            and (remaining - L == 0 or remaining - L >= 5)), None)
            if gchoice is None:
                return None
            gates[gchoice] -= 1
            out.append((PieceType.GATE, gchoice))
            remaining -= gchoice

        # fill the remainder with the FEWEST pieces, drawing from walls AND any
        # spare gates (a gate can stand in as a wall segment). track the source
        # so inventory is consumed correctly.
        pool = {}  # length -> [wall_count, gate_count]
        for L, c in walls.items():
            if c > 0:
                pool.setdefault(L, [0, 0])[0] += c
        for L, c in gates.items():
            if c > 0:
                pool.setdefault(L, [0, 0])[1] += c
        avail = sorted(pool, reverse=True)
        counts = {L: pool[L][0] + pool[L][1] for L in avail}
        best = [None]

        def solve(rem, acc):
            if best[0] is not None and len(acc) >= len(best[0]):
                return
            if rem == 0:
                best[0] = list(acc)
                return
            for L in avail:
                if counts[L] > 0 and L <= rem and (rem - L == 0 or rem - L >= 5):
                    counts[L] -= 1
                    acc.append(L)
                    solve(rem - L, acc)
                    acc.pop()
                    counts[L] += 1

        solve(remaining, [])
        if best[0] is None:
            return None
        for L in best[0]:
            # prefer spending a wall; use a gate only if no wall of that length
            if walls.get(L, 0) > 0:
                walls[L] -= 1
                out.append((PieceType.WALL, L))
            else:
                gates[L] -= 1
                out.append((PieceType.GATE, L))
        return out





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

    def _place(self, pieces, conns) -> Construction | None:
        # (Yuan) room-aware pass: snap an enclosure corner flush onto a room-edge
        # endpoint with its outgoing edge parallel to that wall. Wins on concave /
        # cornered / angled rooms where centering can't find a valid spot.
        pts = walk_points(Construction((0.0, 0.0), 0, pieces, conns))
        corners = [i for i, conn in enumerate(conns) if not conn.is_straight()]
        if corners:
            room_pts = list(self.room.polygon.exterior.coords)[:-1]
            # Sample <=16 room edges and <=8 enclosure corners: <= 256 tries.
            corners = corners[:: max(1, math.ceil(len(corners) / 8))]
            step = max(1, math.ceil(len(room_pts) / 16))
            for j in range(0, len(room_pts), step):
                a, b = room_pts[j], room_pts[(j + 1) % len(room_pts)]
                if a == b:
                    continue
                # Both directions handle either winding of the room boundary.
                for (ax, ay), (bx, by) in ((a, b), (b, a)):
                    room_angle = math.atan2(by - ay, bx - ax)
                    for i in corners:
                        x, y = pts[i]
                        nx, ny = pts[i + 1]
                        angle = room_angle - math.atan2(ny - y, nx - x)
                        c, s = math.cos(angle), math.sin(angle)
                        # Rotate the outgoing edge parallel to the room edge, then
                        # translate this enclosure vertex onto its endpoint.
                        start = (ax - (c * x - s * y), ay - (s * x + c * y))
                        cons = Construction(start, math.degrees(angle), pieces, conns)
                        if validate_construction(cons, self.room, self.inventory).valid:
                            return cons

        # fallback: centroid then grid of anchors, several headings
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
