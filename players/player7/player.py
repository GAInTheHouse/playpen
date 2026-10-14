import math
import time
from collections import Counter

from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polylabel
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

# shape search (edge-list beam search) tuning
SEARCH_CPU_BUDGET = 60.0  # seconds of CPU the shape search may spend
OCT_PLACEMENTS = 1500  # octagon candidates tried for placement before giving up
OCT_HEADINGS = [i * 7.5 for i in range(6)]  # rotations in [0, 45) when placing
# tilts for starting rectangles; turning w x h by 90 gives h x w, so [0, 90)
RECT_HEADINGS = [i * 7.5 for i in range(12)]
OCT_GRID = 12  # grid of candidate centers (per axis) for placing octagons
OCT_ANCHORS = 12  # deepest centers tried per octagon
POS_STEP = 1.0  # smallest grid step (units) when scanning rectangle positions
POS_MAX_STEPS = 32  # cap positions per axis so big rooms stay fast
MAX_ARMS = 30  # beam levels: each level adds one tab (one concave corner)
BEAM_WIDTH = 6  # shapes kept per level (and per kind of starting shape)
BEAM_EVAL = 36  # children fully validated per level before moving on
OPEN_BONUS = 0.5  # weight on the room left ahead of a bump (see _tab_moves)
OPEN_RAY_STEP = 2.5  # degrees between rays when looking for open directions
OPEN_DIRECTIONS = 3  # most open directions to tilt starting shapes toward
OPEN_SEPARATION = 15.0  # degrees two open directions must differ by
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


def _octagon_faces(a, b, cc, d):
    """The non-empty faces of the octagon E=a, NE=cc, N=b, NW=d, W=a, SW=cc,
    S=b, SE=d, as (direction index 0..7, length), counter-clockwise."""
    return [(k, f) for k, f in enumerate((a, cc, b, d, a, cc, b, d)) if f > 0]


def _octagon_turns(a, b, cc, d):
    """(45-degree corners, 90-degree corners) of that octagon, or None if some
    corner would need a 135-degree turn, which no connector makes."""
    faces = _octagon_faces(a, b, cc, d)
    n45 = n90 = 0
    for i in range(len(faces)):
        step = (faces[i][0] - faces[i - 1][0]) % 8
        if step == 1:
            n45 += 1
        elif step == 2:
            n90 += 1
        else:
            return None
    return n45, n90


def _octagon_verts(a, b, cc, d):
    """That octagon's corners, centered on its centroid."""
    pts, x, y = [], 0.0, 0.0
    for k, f in _octagon_faces(a, b, cc, d):
        pts.append((x, y))
        t = math.radians(45 * k)
        x, y = x + f * math.cos(t), y + f * math.sin(t)
    cx, cy = _centroid(pts)
    return [(px - cx, py - cy) for px, py in pts]


def _merge_headings(headings, directions, period):
    """`headings` plus each direction taken modulo `period` (the turn after
    which the shape looks the same), skipping any within a degree of one
    already listed. A shape turned by d mod period has a side facing d."""
    out = list(headings)
    for d in directions:
        h = d % period
        if h > period - 1.0:
            h = 0.0
        if all(abs(h - o) > 1.0 for o in [*out, 0.0]):
            out.append(h)
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
    - "search": an edge-list beam search -- start from the best buildable
      rectangles and octilinear octagons (several rotations), add one
      rectangular tab (one concave corner) per level, keep the best few shapes
      per level, return the best seen (good on stars, L-shapes, crosses,
      slanted rooms...).
    """

    def build_enclosure(self) -> Construction | None:
        candidates = []
        classic = self._classic()
        if classic is not None:
            candidates.append(("classic", classic))
        try:
            found = self._search()
        except Exception:  # noqa: BLE001 -- a search bug must never beat the valid fallback
            found = None
        if found is not None:
            candidates.append(("search", found))

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

    # ---- shape search (edge-list beam search, right and 45-degree faces) ----

    def _search(self):
        """Beam search over buildable shapes, run twice and the better result
        kept. The first run starts from straight rectangles and octilinear
        octagons (any face may be missing, so hexagons and tilted rectangles
        too, at several tilts). The second starts from tilted rectangles, plus
        rectangles and octagons turned so a side faces one of the room's most
        open directions, and uses whatever time is left; keeping it separate
        means those starts add to the first run instead of crowding it."""
        if not self.inventory.has_gate():
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
            for L in set(self._buildables) | set(self._gate_lengths)
            if any(L == g or (L - g) in self._table for g in self._gate_lengths)
        }
        self._faceable = set(self._buildables) | set(self._gate_lengths)
        self._room_prep = prep(self.room.polygon)
        self._decomp_cache = {}
        minx, miny, maxx, maxy = self.room.polygon.bounds
        self._diameter = math.hypot(maxx - minx, maxy - miny) + 1.0

        deadline = time.process_time() + SEARCH_CPU_BUDGET
        rectangles = self.inventory.connectors.get(ConnectorType.RIGHT, 0) >= 4
        starts = []
        if rectangles:
            starts += self._best_rectangles(deadline, BEAM_WIDTH, [0.0])
        starts += self._octilinear_bases(deadline, BEAM_WIDTH)
        best = self._beam(starts, deadline)
        # second run: tilted starts, including tilts that face a side toward the
        # room's most open directions
        open_dirs = self._open_directions()
        tilted = []
        if rectangles:
            headings = _merge_headings(RECT_HEADINGS[1:], open_dirs, 90)
            tilted += self._best_rectangles(deadline, BEAM_WIDTH, headings)
        extra = [h for h in _merge_headings([], open_dirs, 45) if h > 0]
        extra = [h for h in extra if all(abs(h - o) > 1.0 for o in OCT_HEADINGS)]
        if extra:
            tilted += self._octilinear_bases(deadline, BEAM_WIDTH, extra)
        other = self._beam(tilted, deadline)
        if best is None or (other is not None and other[0] > best[0]):
            best = other
        return best[2] if best else None

    def _open_directions(self):
        """The room's OPEN_DIRECTIONS most open directions (degrees): from its
        deepest point, every OPEN_RAY_STEP degrees, how far a strip as wide as
        the shortest buildable face can run (a bare ray would aim at far
        corners instead of down hallways). Keeps the longest local peaks at
        least OPEN_SEPARATION degrees apart."""
        room = self.room.polygon
        pole = polylabel(room, tolerance=0.25)
        width = self._buildables[0]
        n = round(360 / OPEN_RAY_STEP)
        runs = []
        for i in range(n):
            t = math.radians(i * OPEN_RAY_STEP)
            runs.append(
                self._strip_ahead(pole.x, pole.y, math.cos(t), math.sin(t), width)
            )
        peaks = sorted(
            (
                (runs[i], i * OPEN_RAY_STEP)
                for i in range(n)
                if runs[i] > 0
                and runs[i] >= runs[i - 1]
                and runs[i] >= runs[(i + 1) % n]
            ),
            reverse=True,
        )
        kept = []
        for _run, angle in peaks:
            gaps = (min(abs(angle - k), 360 - abs(angle - k)) for k in kept)
            if all(g >= OPEN_SEPARATION for g in gaps):
                kept.append(angle)
            if len(kept) == OPEN_DIRECTIONS:
                break
        return kept

    def _strip_ahead(self, x, y, dx, dy, width):
        """How far a strip of this width, centered on (x, y), can run along the
        unit direction (dx, dy) before touching a wall: the shortest of three
        parallel rays (middle and just inside both edges)."""
        half = 0.45 * width
        px, py = -dy, dx
        return min(
            self._room_ahead(x + px * o, y + py * o, dx, dy) for o in (-half, 0, half)
        )

    def _room_ahead(self, x, y, dx, dy):
        """How far from (x, y) along the unit direction (dx, dy) we can go
        before leaving the room (0 if (x, y) isn't inside it)."""
        reach = self._diameter
        ray = LineString([(x, y), (x + dx * reach, y + dy * reach)])
        start = Point(x, y)
        inside = self.room.polygon.intersection(ray)
        for part in getattr(inside, "geoms", [inside]):
            if part.geom_type == "LineString" and part.distance(start) < 1e-6:
                return part.length
        return 0.0

    def _beam(self, starts, deadline):
        """Level 0 is `starts`. Each level adds one tab (one more concave
        corner) to every shape in the beam. Children are checked best-first by
        estimated score plus an open-room bonus (see `_tab_moves`); of the valid
        ones, the BEAM_WIDTH best by true score plus bonus survive, even when
        they score below their parent, so the search can pass through a worse
        intermediate shape and favours bumps with room to keep growing. Returns
        the best (score, vertices, construction) by true score seen anywhere,
        or None."""
        beam = [(score, _clean(verts), cons) for verts, cons, score in starts]
        best = max(beam, key=lambda s: s[0], default=None)
        for _level in range(MAX_ARMS):
            if not beam or time.process_time() > deadline:
                break
            children = {}
            for _score, verts, _construction in beam:
                for est, bonus, new in self._tab_moves(verts, deadline):
                    key = _canon(new)
                    if key not in children or est + bonus > sum(children[key][:2]):
                        children[key] = (est, bonus, new)
            valid, tried = [], 0
            for _est, bonus, new in sorted(
                children.values(), key=lambda s: s[0] + s[1], reverse=True
            ):
                if len(valid) >= 2 * BEAM_WIDTH or tried >= BEAM_EVAL:
                    break
                if time.process_time() > deadline:
                    break
                tried += 1
                construction = self._to_construction(new)
                if construction is None:
                    continue
                result = validate_construction(construction, self.room, self.inventory)
                if result.valid:
                    score = score_construction(result, self.weights)
                    valid.append((score, bonus, new, construction))
            for score, _bonus, new, construction in valid:
                if best is None or score > best[0]:
                    best = (score, new, construction)
            valid.sort(key=lambda s: s[0] + s[1], reverse=True)
            beam = [(score, new, cons) for score, _b, new, cons in valid[:BEAM_WIDTH]]
        return best

    def _best_rectangles(self, deadline, k, headings):
        """Up to k highest-scoring buildable rectangles that fit, each at the
        first tilt in `headings` where it fits, placed most centrally. To
        tilt, we turn the room the other way and search it axis-aligned, then
        turn the rectangle back. Returns [(vertices, construction, score)]."""
        room = self.room.polygon
        # radius of the largest disk the room holds (polylabel is within 0.25)
        reach = room.exterior.distance(polylabel(room, tolerance=0.25)) + 0.25
        frames = []
        for heading in headings:
            turned = affinity.rotate(room, -heading, origin=(0, 0))
            frames.append((heading, prep(turned), turned.bounds, turned.centroid))
        A, C = self.weights.A, self.weights.C
        pairs = []
        for w in self._buildables:
            for h in self._buildables:
                if w not in self._hostable and h not in self._hostable:
                    continue  # neither side could carry the gate
                if min(w, h) / 2 > reach or w * h > room.area:
                    continue  # its inscribed disk, or its area, can't fit
                est = A * w * h + C * 2 * (w + h)
                pairs.append((est, w, h))
        pairs.sort(reverse=True)

        out = []
        for _est, w, h in pairs:
            if len(out) >= k or time.process_time() > deadline:
                break
            for heading, turned_prep, bounds, centroid in frames:
                minx, miny, maxx, maxy = bounds
                if w > maxx - minx + 1e-9 or h > maxy - miny + 1e-9:
                    continue  # wider or taller than the turned room itself
                pos = self._fit_rect(turned_prep, bounds, centroid, w, h)
                if pos is None:
                    continue
                x, y = pos
                t = math.radians(heading)
                ct, st = math.cos(t), math.sin(t)
                verts = [
                    (px * ct - py * st, px * st + py * ct)
                    for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
                ]
                construction = self._to_construction(verts)
                if construction is None:
                    break  # this size can't be built at any tilt
                result = validate_construction(construction, self.room, self.inventory)
                if result.valid:
                    score = score_construction(result, self.weights)
                    out.append((verts, construction, score))
                    break  # one placement per size
        return out

    def _fit_rect(self, room_prep, bounds, centroid, w, h):
        """Most central axis-aligned placement of a w x h rectangle in this
        (possibly turned) room, closest to its centroid, so an arm has room to
        grow toward each side of a non-convex room. Returns (x, y) or None."""
        minx, miny, maxx, maxy = bounds
        cx, cy = centroid.x, centroid.y
        sx = max(POS_STEP, (maxx - w - minx) / POS_MAX_STEPS)
        sy = max(POS_STEP, (maxy - h - miny) / POS_MAX_STEPS)
        best, best_d = None, float("inf")
        x = minx
        while x <= maxx - w + 1e-9:
            y = miny
            while y <= maxy - h + 1e-9:
                if room_prep.contains(box(x, y, x + w, y + h)):
                    d = (x + w / 2 - cx) ** 2 + (y + h / 2 - cy) ** 2
                    if d < best_d:
                        best, best_d = (x, y), d
                y += sy
            x += sx
        return best

    def _octilinear_bases(self, deadline, k, headings=OCT_HEADINGS):
        """Up to k highest-scoring octagons with faces E=W=a, NE=SW=c, N=S=b,
        NW=SE=d that fit somewhere in the room. With integer faces the loop only
        closes if opposite faces have equal totals, but a face may be several
        pieces (5+7 opposite 12), so this is far wider than matched pairs. Any
        of a, b, c, d may be 0, which also gives hexagons, parallelograms and
        (rotated) rectangles. Returns [(vertices, construction, score)]."""
        room = self.room.polygon
        pole = polylabel(room, tolerance=0.25)
        # candidate centers, deepest first, with their distance to the walls
        depths = sorted(
            (
                (room.exterior.distance(Point(x, y)), x, y)
                for x, y in [(pole.x, pole.y), *self._grid_points(OCT_GRID)]
            ),
            reverse=True,
        )
        reach = depths[0][0] + 0.25  # polylabel is within 0.25 of the true max
        A, C = self.weights.A, self.weights.C
        lengths = [0] + [L for L in sorted(self._faceable) if L >= c.MIN_WALL_LENGTH]
        r2 = math.sqrt(2)
        have45 = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        have90 = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        candidates = []
        for a in lengths:
            for b in lengths:
                for cc in lengths:
                    for d in lengths:
                        turns = _octagon_turns(a, b, cc, d)
                        if turns is None or turns[0] > have45 or turns[1] > have90:
                            continue
                        wx = a + (cc + d) / r2
                        hy = b + (cc + d) / r2
                        # these octagons are centrally symmetric, so their largest
                        # inscribed disk is centered on the centroid with radius
                        # half their smallest width, which is one of these four
                        inner = min(wx, hy, cc + (a + b) / r2, d + (a + b) / r2) / 2
                        if inner > reach:
                            continue  # no spot in the room is that deep
                        area = wx * hy - (cc * cc + d * d) / 2
                        if area <= 1e-9 or area > room.area:
                            continue
                        if not any(f in self._hostable for f in (a, b, cc, d) if f):
                            continue
                        perimeter = 2 * (a + b + cc + d)
                        est = A * area + C * perimeter
                        candidates.append((est, inner, a, b, cc, d))
        candidates.sort(reverse=True)

        out, attempts = [], 0
        for _est, inner, a, b, cc, d in candidates:
            if len(out) >= k or attempts >= OCT_PLACEMENTS:
                break
            if time.process_time() > deadline:
                break
            # the centroid must sit at least `inner` from every wall
            anchors = [(x, y) for depth, x, y in depths if depth >= inner - 1e-9]
            anchors = anchors[:OCT_ANCHORS]
            if not anchors:
                continue
            attempts += 1
            verts = self._place(_octagon_verts(a, b, cc, d), anchors, headings)
            if verts is None:
                continue
            construction = self._to_construction(verts)
            if construction is None:
                continue
            result = validate_construction(construction, self.room, self.inventory)
            if result.valid:
                out.append(
                    (verts, construction, score_construction(result, self.weights))
                )
        return out

    def _grid_points(self, steps):
        """Points of a steps x steps grid over the room's bounds that lie inside."""
        polygon = self.room.polygon
        minx, miny, maxx, maxy = polygon.bounds
        for i in range(steps):
            for j in range(steps):
                x = minx + (maxx - minx) * (i + 0.5) / steps
                y = miny + (maxy - miny) * (j + 0.5) / steps
                if polygon.contains(Point(x, y)):
                    yield (x, y)

    def _place(self, local, anchors, headings):
        """First anchor/rotation at which this shape (centered on its centroid)
        fits in the room, or None. Turning one of these octagons by 45 degrees
        gives another candidate, so rotations below 45 degrees are enough."""
        for ax, ay in anchors:
            for heading in headings:
                t = math.radians(heading)
                ct, st = math.cos(t), math.sin(t)
                pts = [(ax + x * ct - y * st, ay + x * st + y * ct) for x, y in local]
                if self._room_prep.contains(Polygon(pts)):
                    return pts
        return None

    def _tab_moves(self, verts, deadline):
        """Every buildable tab on every edge of this (clean, CCW) outline, each at
        the deepest depth that fits. Yields (estimated pre-gate score, open-room
        bonus, new outline). The estimate is exact for area and perimeter: a
        w x h tab adds w*h area and 2h perimeter wherever it sits. The bonus
        rewards tabs with room left ahead of them: from the tab's outer edge,
        straight out (the way the next tab would grow), how far a strip of the
        same width runs before a wall, capped at the longest face; it is
        OPEN_BONUS times the area a tab of the same width could still add. Edges may run in any of
        the shape's eight directions; a tab's sides are perpendicular to its
        edge, so they stay in those directions too."""
        A, C = self.weights.A, self.weights.C
        grow = OPEN_BONUS * max(A, 0.0)
        poly = Polygon(verts)
        base = A * poly.area + C * poly.length
        poly_prep = prep(poly)
        n = len(verts)
        for i in range(n):
            if time.process_time() > deadline:
                return
            p, q = verts[i], verts[(i + 1) % n]
            raw = math.hypot(q[0] - p[0], q[1] - p[1])
            length = round(raw)
            if length < c.MIN_WALL_LENGTH:
                continue
            dx, dy = (q[0] - p[0]) / raw, (q[1] - p[1]) / raw
            nx, ny = dy, -dx  # outward normal for a counter-clockwise loop
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
                    mx = p[0] + dx * (s + w / 2) + nx * h
                    my = p[1] + dy * (s + w / 2) + ny * h
                    ahead = self._strip_ahead(mx, my, nx, ny, w)
                    yield (
                        base + A * w * h + C * 2 * h,
                        grow * w * min(ahead, c.MAX_FACE_LENGTH),
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

        def rect(lo, hi):
            return Polygon(
                [
                    (ax + nx * lo, ay + ny * lo),
                    (bx + nx * lo, by + ny * lo),
                    (bx + nx * hi, by + ny * hi),
                    (ax + nx * hi, ay + ny * hi),
                ]
            )

        def fits(h):
            if not self._room_prep.contains(rect(0, h)):
                return False
            # the same rectangle minus a sliver at its base must not touch the
            # shape, or the tab would overlap it or pinch against another arm
            return not poly_prep.intersects(rect(1e-6, h))

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
        """Turn a vertex loop whose edges all run in multiples of 45 degrees from
        the first one into a Construction: each edge is one face, built from
        pieces the inventory can actually supply (see `_allocate`), joined by
        straight connectors within a face; corners get right connectors (90
        degree turns) or diagonal ones (45 degree turns), reflex where the corner
        turns in. Returns None if it can't be built."""
        verts = _clean(verts)
        n = len(verts)
        if n < c.MIN_ENCLOSURE_PIECES:
            return None

        lengths, steps = [], []
        h0 = None
        for i in range(n):
            p, q = verts[i], verts[(i + 1) % n]
            dx, dy = q[0] - p[0], q[1] - p[1]
            raw = math.hypot(dx, dy)
            length = round(raw)
            if abs(raw - length) > 1e-6 or length not in self._faceable:
                return None
            angle = math.degrees(math.atan2(dy, dx))
            if h0 is None:
                h0 = angle
            k = round((angle - h0) / 45)
            if abs(angle - h0 - 45 * k) > 1e-4:
                return None  # not one of this shape's eight directions
            lengths.append(length)
            steps.append(k)

        corners = []
        for i in range(n):
            turn = ((steps[i] - steps[i - 1]) * 45 + 180) % 360 - 180
            if abs(turn) == 90:
                corners.append(Connector(ConnectorType.RIGHT, reflex=turn < 0))
            elif abs(turn) == 45:
                corners.append(Connector(ConnectorType.DIAGONAL, reflex=turn < 0))
            else:
                return None  # a 135-degree turn has no connector
        need = Counter(conn.connector_type for conn in corners)
        if any(need[t] > self.inventory.connectors.get(t, 0) for t in need):
            return None

        faces = None
        if self.weights.G > 0 and sum(self.inventory.gates.values()) >= 2:
            faces = self._allocate(lengths, 2)
        if faces is None:
            faces = self._allocate(lengths, 1)
        if faces is None:
            return None

        pieces, connectors = [], []
        for i in range(n):
            for k, piece in enumerate(faces[i]):
                pieces.append(piece)
                connectors.append(
                    corners[i] if k == 0 else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(verts[0], h0, pieces, connectors)

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
