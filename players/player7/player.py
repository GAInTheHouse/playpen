import math
import time
from collections import Counter

from shapely.geometry import Point, Polygon, box
from shapely.prepared import prep

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

# orthogonal (edge-list) search tuning
ORTHO_CPU_BUDGET = 45.0  # seconds of CPU the orthogonal search may spend
POS_STEP = 1.0  # smallest grid step (units) when scanning rectangle positions
POS_MAX_STEPS = 32  # cap positions per axis so big rooms stay fast
MAX_ARMS = 8  # beam levels: each level adds one tab (one concave corner)
BEAM_WIDTH = 4  # shapes kept per level (level 0 = the best rectangles)
BEAM_EVAL = 24  # children fully validated per level before moving on
ALLOC_NODES = 2000  # backtracking budget when assigning pieces to faces


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


def _build_table(coins, max_len=c.MAX_FACE_LENGTH):
    """Which face lengths in [MIN_WALL, max_len] can be built as a sum of the
    available wall lengths, and one decomposition for each.

    Returns {length: [wall_length, ...]}; the empty sum (0) maps to []. A single
    wall counts, so every available length appears; longer faces are runs of
    walls joined by straight connectors (the simulator's notion of a face)."""
    coins = sorted({int(w) for w in coins if w >= c.MIN_WALL_LENGTH})
    table = {0: []}
    for length in range(c.MIN_WALL_LENGTH, max_len + 1):
        for w in coins:
            if w <= length and (length - w) in table:
                table[length] = table[length - w] + [w]
                break
    return table


def _clean(verts):
    """Drop repeated points and collinear corners so every remaining vertex is
    a real right-angle turn (one polygon edge == one face)."""
    pts = []
    for p in verts:
        if not pts or abs(p[0] - pts[-1][0]) > 1e-9 or abs(p[1] - pts[-1][1]) > 1e-9:
            pts.append(p)
    if (
        len(pts) > 1
        and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-9
    ):
        pts.pop()
    out = []
    m = len(pts)
    for i in range(m):
        a, b, d = pts[i - 1], pts[i], pts[(i + 1) % m]
        cross = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
        if abs(cross) < 1e-9:
            continue  # b lies on the a->d line, so it isn't a corner
        out.append(b)
    return out


def _canon(verts):
    """A rotation-independent key for an outline, to spot duplicate shapes."""
    pts = [(round(x, 6), round(y, 6)) for x, y in verts]
    k = pts.index(min(pts))
    return tuple(pts[k:] + pts[:k])


def _annex(verts, i, s, w, h):
    """Push a rectangular tab (width w, depth h) outward from edge i, starting
    s units along it. Returns a new vertex list; the loop stays closed because
    a tab is a closed detour. All five new edges have lengths s, h, w, h,
    (L-s-w), each chosen from buildable lengths by the caller."""
    n = len(verts)
    p, q = verts[i], verts[(i + 1) % n]
    length = math.hypot(q[0] - p[0], q[1] - p[1])
    dx, dy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
    nx, ny = dy, -dx  # outward normal for a counter-clockwise loop
    a = (p[0] + dx * s, p[1] + dy * s)
    b = (p[0] + dx * (s + w), p[1] + dy * (s + w))
    a_out = (a[0] + nx * h, a[1] + ny * h)
    b_out = (b[0] + nx * h, b[1] + ny * h)
    return verts[: i + 1] + [a, a_out, b_out, b] + verts[i + 1 :]


class Player7(Player):
    """Build the best enclosure we can and return it. Two strategies compete and
    the higher-scoring valid one wins:

    - "classic": the largest valid rectangle and octagon (good on convex rooms).
    - "orthogonal": an edge-list beam search -- start from the best buildable
      rectangles, add one rectangular tab (one concave corner) per level, keep
      the best few shapes per level, return the best seen (good on stars,
      L-shapes, crosses...).
    """

    def build_enclosure(self) -> Construction | None:
        candidates = []
        classic = self._classic()
        if classic is not None:
            candidates.append(("classic", classic))
        try:
            ortho = self._orthogonal()
        except Exception:  # noqa: BLE001 -- a search bug must never beat the valid fallback
            ortho = None
        if ortho is not None:
            candidates.append(("orthogonal", ortho))

        best, best_score, best_name = None, None, None
        for name, construction in candidates:
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                continue
            score = score_construction(result, self.weights)
            if best_score is None or score > best_score:
                best, best_score, best_name = construction, score, name

        if best is None:
            self.note = "nothing fits"
            return None
        if best_score <= 0:
            # an empty room scores 0, which beats any net-negative enclosure
            self.note = f"best valid scores {best_score:.0f} <= 0; returning nothing"
            return None
        self.note = f"{best_name} ({best_score:.0f})"
        return best

    # ---- classic strategy (rectangle / octagon) -----------------------------

    def _classic(self):
        best, best_score = None, None
        for construction in (
            self._best(self._rectangles(), range(0, 180, 15)),
            self._best(self._octagons(), range(0, 45, 5)),
        ):
            if construction is None:
                continue
            construction = self._add_second_gate(construction)
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                continue
            score = score_construction(result, self.weights)
            if best_score is None or score > best_score:
                best, best_score = construction, score
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

    # ---- orthogonal strategy (edge-list rectilinear search) -----------------

    def _orthogonal(self):
        """Beam search over rectilinear shapes. Level 0 is the best few buildable
        rectangles; each level adds one tab (one more concave corner) to every
        shape in the beam and keeps the best BEAM_WIDTH results, even when they
        score below their parent, so the search can pass through a worse
        intermediate shape. The best shape seen at any level wins."""
        if not self.inventory.has_gate():
            return None
        if self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
            return None
        coins = [
            length
            for length, n in self.inventory.walls.items()
            if n > 0 and length >= c.MIN_WALL_LENGTH
        ]
        if not coins:
            return None

        self._coins = sorted(set(coins))
        self._table = _build_table(coins)
        self._buildables = sorted(L for L in self._table if L >= c.MIN_WALL_LENGTH)
        if not self._buildables:
            return None
        self._gate_lengths = sorted(g for g, n in self.inventory.gates.items() if n > 0)
        self._hostable = {
            L
            for L in self._buildables
            if any(L == g or (L - g) in self._table for g in self._gate_lengths)
        }
        if not self._hostable:  # no face length can carry the one required gate
            return None
        self._room_prep = prep(self.room.polygon)
        self._decomp_cache = {}

        deadline = time.process_time() + ORTHO_CPU_BUDGET
        beam = [
            (score, _clean(verts), construction)
            for verts, construction, score in self._best_rectangles(
                deadline, BEAM_WIDTH
            )
        ]
        best = max(beam, key=lambda s: s[0], default=None)
        for _level in range(MAX_ARMS):
            if not beam or time.process_time() > deadline:
                break
            children = {}
            for _score, verts, _construction in beam:
                for est, new in self._tab_moves(verts, deadline):
                    key = _canon(new)
                    if key not in children or est > children[key][0]:
                        children[key] = (est, new)
            next_beam, tried = [], 0
            for _est, new in sorted(
                children.values(), key=lambda s: s[0], reverse=True
            ):
                if len(next_beam) >= BEAM_WIDTH or tried >= BEAM_EVAL:
                    break
                if time.process_time() > deadline:
                    break
                tried += 1
                construction = self._to_construction(new)
                if construction is None:
                    continue
                result = validate_construction(construction, self.room, self.inventory)
                if result.valid:
                    next_beam.append(
                        (score_construction(result, self.weights), new, construction)
                    )
            beam = next_beam
            for state in beam:
                if best is None or state[0] > best[0]:
                    best = state
        return best[2] if best else None

    def _best_rectangles(self, deadline, k):
        """Up to k highest-scoring buildable rectangles that fit, each placed most
        centrally. Returns [(vertices, construction, score)]."""
        minx, miny, maxx, maxy = self.room.polygon.bounds
        bbx, bby = maxx - minx, maxy - miny
        A, C = self.weights.A, self.weights.C
        pairs = []
        for w in self._buildables:
            if w > bbx + 1e-9:
                continue
            for h in self._buildables:
                if h > bby + 1e-9:
                    continue
                if w not in self._hostable and h not in self._hostable:
                    continue  # neither side could carry the gate
                est = A * w * h + C * 2 * (w + h)
                pairs.append((est, w, h))
        pairs.sort(reverse=True)

        out = []
        for _est, w, h in pairs:
            if time.process_time() > deadline:
                break
            pos = self._fit_rect(minx, miny, maxx, maxy, w, h)
            if pos is None:
                continue
            x, y = pos
            verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            construction = self._to_construction(verts)
            if construction is None:
                continue
            result = validate_construction(construction, self.room, self.inventory)
            if result.valid:
                out.append(
                    (verts, construction, score_construction(result, self.weights))
                )
                if len(out) >= k:
                    break
        return out

    def _fit_rect(self, minx, miny, maxx, maxy, w, h):
        """Most central feasible placement (closest to the room centroid), so an
        arm has room to grow toward each side of a non-convex room."""
        cx, cy = self.room.polygon.centroid.x, self.room.polygon.centroid.y
        sx = max(POS_STEP, (maxx - w - minx) / POS_MAX_STEPS)
        sy = max(POS_STEP, (maxy - h - miny) / POS_MAX_STEPS)
        best, best_d = None, float("inf")
        x = minx
        while x <= maxx - w + 1e-9:
            y = miny
            while y <= maxy - h + 1e-9:
                if self._room_prep.contains(box(x, y, x + w, y + h)):
                    d = (x + w / 2 - cx) ** 2 + (y + h / 2 - cy) ** 2
                    if d < best_d:
                        best, best_d = (x, y), d
                y += sy
            x += sx
        return best

    def _tab_moves(self, verts, deadline):
        """Every buildable tab on every edge of this (clean, CCW) outline, each at
        the deepest depth that fits. Yields (estimated pre-gate score, new
        outline). The estimate is exact for area and perimeter: a w x h tab adds
        w*h area and 2h perimeter wherever it sits."""
        A, C = self.weights.A, self.weights.C
        poly = Polygon(verts)
        base = A * poly.area + C * poly.length
        poly_prep = prep(poly)
        n = len(verts)
        for i in range(n):
            if time.process_time() > deadline:
                return
            p, q = verts[i], verts[(i + 1) % n]
            length = round(math.hypot(q[0] - p[0], q[1] - p[1]))
            if length < c.MIN_WALL_LENGTH:
                continue
            for s in [0] + self._buildables:  # 0 = tab flush to this corner
                if s > length - c.MIN_WALL_LENGTH:
                    break
                for w in self._buildables:
                    rem = length - s - w
                    if rem < 0:
                        break  # w only grows, rem only shrinks
                    if rem not in self._table:  # table includes 0
                        continue
                    h = self._deepest_tab(verts, i, s, w, poly_prep)
                    if h is None:
                        continue
                    yield (
                        base + A * w * h + C * 2 * h,
                        _clean(_annex(verts, i, s, w, h)),
                    )

    def _deepest_tab(self, verts, i, s, w, poly_prep):
        """Deepest buildable depth for a tab (width w, offset s on edge i) whose
        rectangle lies inside the room and touches the shape only along its base,
        or None. Testing the tab's own rectangle is much cheaper than rebuilding
        the whole polygon; feasibility shrinks with depth, so binary-search it."""
        n = len(verts)
        p, q = verts[i], verts[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        dx, dy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
        nx, ny = dy, -dx  # outward normal for a counter-clockwise loop
        ax, ay = p[0] + dx * s, p[1] + dy * s
        bx, by = p[0] + dx * (s + w), p[1] + dy * (s + w)

        def fits(h):
            xs = (ax, bx, ax + nx * h, bx + nx * h)
            ys = (ay, by, ay + ny * h, by + ny * h)
            tab = box(min(xs), min(ys), max(xs), max(ys))
            if not self._room_prep.contains(tab):
                return False
            # the same rectangle minus a sliver at its base must not touch the
            # shape, or the tab would overlap it or pinch against another arm
            e = 1e-6
            xs = (ax + nx * e, bx + nx * e, ax + nx * h, bx + nx * h)
            ys = (ay + ny * e, by + ny * e, ay + ny * h, by + ny * h)
            return not poly_prep.intersects(box(min(xs), min(ys), max(xs), max(ys)))

        asc = self._buildables
        if not fits(asc[0]):
            return None
        lo, hi, best = 0, len(asc) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(asc[mid]):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return asc[best]

    def _to_construction(self, verts):
        """Turn an axis-aligned rectilinear vertex loop into a Construction: each
        edge is one face, built from pieces the inventory can actually supply
        (see `_allocate`), joined by straight connectors within a face and right
        connectors (reflex where the corner turns in) between faces. Returns
        None if it can't be built."""
        verts = _clean(verts)
        n = len(verts)
        if n < c.MIN_ENCLOSURE_PIECES:
            return None
        if n > self.inventory.connectors.get(ConnectorType.RIGHT, 0):
            return None

        lengths, headings = [], []
        for i in range(n):
            p, q = verts[i], verts[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            if abs(dx) > 1e-6 and abs(dy) > 1e-6:
                return None  # not axis-aligned
            length = round(math.hypot(dx, dy))
            if length < c.MIN_WALL_LENGTH or length not in self._table:
                return None
            lengths.append(length)
            headings.append(round(math.degrees(math.atan2(dy, dx))) % 360)

        faces = None
        if self.weights.G > 0 and sum(self.inventory.gates.values()) >= 2:
            faces = self._allocate(lengths, 2)
        if faces is None:
            faces = self._allocate(lengths, 1)
        if faces is None:
            return None

        pieces, connectors = [], []
        for i in range(n):
            turn = ((headings[i] - headings[i - 1] + 180) % 360) - 180  # +90 or -90
            corner = Connector(ConnectorType.RIGHT, reflex=turn < 0)
            for k, piece in enumerate(faces[i]):
                pieces.append(piece)
                connectors.append(
                    corner if k == 0 else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(verts[0], float(headings[0]), pieces, connectors)

    def _decomps(self, length):
        """All ways to write `length` as a sum of available wall lengths (as
        non-increasing tuples), fewest pieces first. Ignores counts."""
        if length in self._decomp_cache:
            return self._decomp_cache[length]
        out = []

        def rec(rest, cap, parts):
            if rest == 0:
                out.append(tuple(parts))
                return
            for w in reversed(self._coins):
                if w <= cap and w <= rest:
                    parts.append(w)
                    rec(rest - w, w, parts)
                    parts.pop()

        rec(length, length, [])
        out.sort(key=len)
        self._decomp_cache[length] = out
        return out

    def _allocate(self, lengths, gates_wanted):
        """Choose the pieces for every face so the whole shape fits the inventory:
        wall and gate counts and straight connectors (a face of k pieces needs
        k-1). Exactly `gates_wanted` faces get one gate each. Backtracks over
        each face's decompositions, fewest pieces first. Returns a list of piece
        lists (one per face) or None."""
        walls = Counter({k: v for k, v in self.inventory.walls.items() if v > 0})
        gates = Counter({k: v for k, v in self.inventory.gates.items() if v > 0})
        straights = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        n = len(lengths)
        options = []
        for length in lengths:
            opts = [(None, d) for d in self._decomps(length)]
            for g in self._gate_lengths:
                if length == g:
                    opts.append((g, ()))
                elif length - g >= c.MIN_WALL_LENGTH:
                    opts.extend((g, d) for d in self._decomps(length - g))
            opts.sort(key=lambda o: len(o[1]) + (o[0] is not None))
            options.append(opts)
        order = sorted(range(n), key=lambda i: len(options[i]))
        chosen = [None] * n
        budget = [ALLOC_NODES]

        def rec(k, gates_left, straights_left):
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if k == n:
                return gates_left == 0
            i = order[k]
            for gate, parts in options[i]:
                if gate is not None and gates_left == 0:
                    continue
                need = len(parts) + (gate is not None) - 1
                if need > straights_left:
                    continue
                use = Counter(parts)
                if any(walls[w] < m for w, m in use.items()):
                    continue
                if gate is not None and gates[gate] < 1:
                    continue
                walls.subtract(use)
                if gate is not None:
                    gates[gate] -= 1
                chosen[i] = (gate, parts)
                if rec(k + 1, gates_left - (gate is not None), straights_left - need):
                    return True
                walls.update(use)
                if gate is not None:
                    gates[gate] += 1
            return False

        if not rec(0, gates_wanted, straights):
            return None
        faces = []
        for gate, parts in chosen:
            face = [Piece(PieceType.GATE, gate)] if gate is not None else []
            face += [Piece(PieceType.WALL, w) for w in parts]
            faces.append(face)
        return faces
