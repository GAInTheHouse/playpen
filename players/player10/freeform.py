"""Free-form enclosures: a beam search that walks the boundary inside the room.

The walk is counter-clockwise. Each step adds one face (a length the inventory can
make) after a corner of +-90 or +-45 degrees. Faces are only kept if they lie inside
the room and stay clear of the earlier path, so containment is pruned as we go.

Closure is exact: headings are multiples of 45 degrees and lengths are integers, so
the loop closes iff the lengths walked along each of the axes 0/45/90/135 cancel
(sqrt(2) is irrational). `Walk.lattice` tracks those four integers.

Pieces are not tracked here; closed shapes are handed to Assigner.assign, which
works for any list of face lengths.

Beam search logic after finding 4 shapes
"""

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient

from players.player10.shapes import Shape
from src.pieces import ConnectorType

TURNS = (90, 45, -45, -90)  # degrees, left (+) is counter-clockwise
MAX_FACES = 14
WIDTHS = (30, 120, 480)  # beam widths, narrowest first (anytime behaviour)


@dataclass(frozen=True)
class Walk:
    points: tuple  # corners so far; points[-1] is where the walk is now
    heading: int  # in 45-degree steps from the base angle, mod 8
    turned: int  # degrees turned at the corners so far
    lattice: tuple  # exact lengths walked along the 0/45/90/135 axes
    faces: tuple
    turns: tuple  # turns[i] = corner at the start of face i (turns[0] set on closing)
    rights: int
    diagonals: int


def _step(h: int, length: int) -> list[int]:
    v = [0, 0, 0, 0]
    v[h % 4] = length if h < 4 else -length
    return v


def _wrap(angle: float) -> float:
    return (angle + 180) % 360 - 180


class Walker:
    def __init__(
        self, room_search, lengths, total_length, rights, diagonals, weights
    ) -> None:
        self.region = room_search.region
        self.lengths = sorted(lengths)
        self.length_set = set(lengths)
        self.total_length = total_length
        self.max_rights, self.max_diagonals = rights, diagonals
        self.A, self.C = weights.A, weights.C
        self.max_faces = MAX_FACES

    # ------------------------------------------------------------ scoring

    def _value(self, w: Walk) -> float:
        """Score if the walk were closed by a straight chord: what the beam ranks by."""
        p = np.array(w.points)
        area = 0.5 * abs(
            np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))
        )
        perimeter = sum(w.faces) + math.dist(w.points[-1], w.points[0])
        return self.A * area + self.C * perimeter

    # ---------------------------------------------------------- expansion

    def _extend(self, w: Walk, base: float) -> list[Walk]:
        n = len(w.faces)
        used = sum(w.faces)
        earlier = LineString(w.points[:-1]) if len(w.points) > 2 else None
        lengths = [L for L in self.lengths if used + L <= self.total_length]
        if not lengths:
            return []
        out = []
        for turn in (0,) if n == 0 else TURNS:
            right = abs(turn) == 90
            rights = w.rights + (1 if turn and right else 0)
            diagonals = w.diagonals + (1 if turn and not right else 0)
            if rights > self.max_rights or diagonals > self.max_diagonals:
                continue
            turned = w.turned + turn
            # can the remaining faces still turn enough to come round?
            if turned + (self.max_faces - n) * 90 < 360:
                continue
            h = (w.heading + turn // 45) % 8
            t = math.radians(base + 45 * h)
            x, y = w.points[-1]
            ends = np.array(
                [(x + L * math.cos(t), y + L * math.sin(t)) for L in lengths]
            )
            starts = np.tile([x, y], (len(lengths), 1))
            segs = shapely.linestrings(np.stack([starts, ends], axis=1))
            ok = shapely.contains(self.region, segs)
            if earlier is not None:
                ok &= ~shapely.intersects(segs, earlier)
            for L, end, good in zip(lengths, ends, ok):
                if good:
                    lattice = tuple(a + b for a, b in zip(w.lattice, _step(h, L)))
                    out.append(
                        Walk(
                            w.points + ((float(end[0]), float(end[1])),),
                            h, turned, lattice, w.faces + (L,), w.turns + (turn,),
                            rights, diagonals,
                        )
                    )
        return out

    def _closing(self, w: Walk):
        """(faces, turns) if one more face brings the walk exactly back to its start."""
        if len(w.points) < 4:
            return None
        need = [-v for v in w.lattice]
        nonzero = [k for k in range(4) if need[k]]
        if len(nonzero) != 1:
            return None
        k = nonzero[0]
        length = abs(need[k])
        if length not in self.length_set or sum(w.faces) + length > self.total_length:
            return None
        h = k if need[k] > 0 else k + 4
        t1, t2 = _wrap((h - w.heading) * 45), _wrap(-h * 45)
        if t1 not in TURNS or t2 not in TURNS or w.turned + t1 + t2 != 360:
            return None
        rights = w.rights + sum(abs(t) == 90 for t in (t1, t2))
        diagonals = w.diagonals + sum(abs(t) == 45 for t in (t1, t2))
        if rights > self.max_rights or diagonals > self.max_diagonals:
            return None
        seg = LineString([w.points[-1], w.points[0]])
        if not shapely.contains(self.region, seg):
            return None
        touch = shapely.intersection(seg, LineString(w.points[:-1]))
        if not touch.equals(Point(w.points[0])):
            return None
        return w.faces + (length,), (int(t2),) + w.turns[1:] + (int(t1),)

    # ------------------------------------------------------------- search

    def run(self, start, base: float, width: int, deadline: float):
        """Closed polygons found by a beam of `width` walks from `start`, whose first
        face heads along `base` degrees."""
        beam = [Walk((start,), 0, 0, (0, 0, 0, 0), (), (), 0, 0)]
        closed = []
        for _ in range(self.max_faces):
            nxt: dict = {}
            for w in beam:
                if time.process_time() > deadline:
                    return closed
                for child in self._extend(w, base):
                    done = self._closing(child)
                    if done:
                        closed.append(done)
                    value = self._value(child)
                    key = (child.lattice, child.heading, child.turned, round(value, 1))
                    nxt[key] = (value, child)
            beam = [w for _, w in heapq.nlargest(width, nxt.values(), key=lambda t: t[0])]
            if not beam:
                break
        return closed


def _make_shape(faces: tuple, turns: tuple) -> Shape:
    draft = Shape("free", faces, faces, tuple(float(t) for t in turns), 0.0, 0.0)
    pts = np.array(draft.vertices())
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return Shape("free", faces, faces, draft.turns, float(area), float(sum(faces)))


def free_shapes(room_search, assigner, inventory, weights, deadline):
    """Yield (shape, start, heading) for closed polygons that fit the room, widening
    the beam each round and best-ranked first within a round. The caller checks the
    pieces (Assigner) and the simulator's validator."""
    walker = Walker(
        room_search,
        assigner.lengths,
        assigner.total_length,
        inventory.connectors.get(ConnectorType.RIGHT, 0),
        inventory.connectors.get(ConnectorType.DIAGONAL, 0),
        weights,
    )
    ring = list(orient(room_search.polygon, 1.0).exterior.coords)[:-1]
    starts = []
    for i, p in enumerate(ring):
        q = ring[(i + 1) % len(ring)]
        # first face runs along the wall leaving this corner (interior on the left)
        starts.append((p, math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))))

    seen = set()
    for width in WIDTHS:
        batch = []
        for start, base in starts:
            if time.process_time() > deadline:
                break
            for faces, turns in walker.run(start, base, width, deadline):
                key = (start, round(base, 6), faces, turns)
                if key not in seen:
                    seen.add(key)
                    batch.append((_make_shape(faces, turns), start, base))
        batch.sort(key=lambda t: -(weights.A * t[0].area + weights.C * t[0].perimeter))
        yield from batch
        if time.process_time() > deadline:
            return