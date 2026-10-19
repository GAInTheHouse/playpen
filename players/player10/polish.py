"""Polishing a rectilinear enclosure by adding or cutting axis-aligned rectangles.

All work is in the shape's own frame: origin at its start point, x axis along its
heading. There every corner is an integer point, and adding or cutting a rectangle
with integer corners keeps it that way, so each variant has integer faces and closes
exactly. A variant has to be one simple polygon inside the room, with every face a
length the inventory can make, and needing no more corners than there are
right-angle connectors.

For a shape with no inward bays the perimeter is its bounding box's, so jogs and
notches buy area at no extra perimeter: that is what this pass finds.
"""

import math
import time
from collections.abc import Iterator

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from players.player10.shapes import Shape
from src.pieces import ConnectorType

MAX_DEPTH = 30  # the longest face the rules allow
CUT_DEPTHS = (5, 10)
EPS = 1e-9


def _rect(a, b, out, depth) -> Polygon:
    """The rectangle on segment a-b, `depth` units along the normal `out`."""
    return Polygon(
        [
            a,
            b,
            (b[0] + out[0] * depth, b[1] + out[1] * depth),
            (a[0] + out[0] * depth, a[1] + out[1] * depth),
        ]
    )


def _corners(poly: Polygon) -> list[tuple[int, int]]:
    """Integer corners of a counter-clockwise ring, collinear points removed."""
    pts = [(round(x), round(y)) for x, y in poly.exterior.coords[:-1]]
    n = len(pts)
    keep = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            keep.append(b)
    return keep


class Polisher:
    def __init__(self, region, assigner, inventory, rank) -> None:
        self.region = region
        self.lengths = sorted(assigner.lengths)
        self.length_set = set(self.lengths)
        self.total_length = assigner.total_length
        self.rights = inventory.connectors.get(ConnectorType.RIGHT, 0)
        self.rank = rank  # rank(area, perimeter): the most a shape could score

    def neighbours(self, shape, start, heading, floor, deadline) -> Iterator:
        """Variants of `shape` as (shape, start, heading), best-ranked first. Only
        variants that could score above `floor` are offered."""
        if any(abs(t) != 90 for t in shape.turns):
            return  # diagonal faces: not handled here
        poly = orient(Polygon([(round(x), round(y)) for x, y in shape.vertices()]), 1.0)
        # the room as seen from the shape's frame
        region = affinity.rotate(
            affinity.translate(self.region, -start[0], -start[1]),
            -heading,
            origin=(0, 0),
        )
        shapely.prepare(region)

        ranked: dict = {}
        pts = list(poly.exterior.coords)[:-1]
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            dx, dy = q[0] - p[0], q[1] - p[1]
            length = int(abs(dx) + abs(dy))
            d = (int(np.sign(dx)), int(np.sign(dy)))
            out = (d[1], -d[0])  # right of travel: outward for a counter-clockwise ring
            for w in self.lengths:
                if w > length or time.process_time() > deadline:
                    break
                for s in {0, length - w, (length - w) // 2}:
                    a = (p[0] + d[0] * s, p[1] + d[1] * s)
                    b = (a[0] + d[0] * w, a[1] + d[1] * w)
                    free = self._free_depth(a, b, out, region)
                    moves = [
                        (depth, 1)
                        for depth in {free, free // 2, 5}
                        if 1 <= depth <= free
                    ]
                    moves += [(depth, -1) for depth in CUT_DEPTHS]
                    for depth, sign in moves:
                        rect = _rect(a, b, out, sign * depth)
                        new = poly.union(rect) if sign > 0 else poly.difference(rect)
                        self._offer(new, floor, ranked)

        for _, new, corners in sorted(ranked.values(), key=lambda t: -t[0]):
            if new.is_valid and shapely.contains(region, new):
                yield self._shape(new, corners, start, heading)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _free_depth(a, b, out, region) -> int:
        """How far a rectangle on a-b can be pushed outward and stay in the room."""
        low, high = 0, MAX_DEPTH
        while low < high:
            mid = (low + high + 1) // 2
            if shapely.contains(region, _rect(a, b, out, mid)):
                low = mid
            else:
                high = mid - 1
        return low

    def _offer(self, new, floor, ranked) -> None:
        """Keep `new` if it passes the cheap tests and could beat `floor`."""
        if new.is_empty or new.geom_type != "Polygon" or len(new.interiors):
            return
        new = orient(new, 1.0)
        corners = _corners(new)
        n = len(corners)
        if n < 4 or n > self.rights or new.length > self.total_length + EPS:
            return
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            if abs(b[0] - a[0]) + abs(b[1] - a[1]) not in self.length_set:
                return
        est = self.rank(new.area, new.length)
        if est > floor + EPS:
            ranked[frozenset(corners)] = (est, new, corners)

    @staticmethod
    def _shape(new, corners, start, heading):
        """A Shape for the polygon, whose first face runs east from its first corner,
        and where that corner lies in the room."""
        n = len(corners)
        east = [
            i
            for i in range(n)
            if corners[(i + 1) % n][1] == corners[i][1]
            and corners[(i + 1) % n][0] > corners[i][0]
        ]
        k = min(east, key=lambda i: (corners[i][1], corners[i][0]))
        pts = corners[k:] + corners[:k]
        faces, turns = [], []
        for i in range(n):
            before, p, after = pts[i - 1], pts[i], pts[(i + 1) % n]
            faces.append(abs(after[0] - p[0]) + abs(after[1] - p[1]))
            cross = (p[0] - before[0]) * (after[1] - p[1]) - (p[1] - before[1]) * (
                after[0] - p[0]
            )
            turns.append(90.0 if cross > 0 else -90.0)
        t = math.radians(heading)
        x, y = pts[0]
        world = (
            start[0] + x * math.cos(t) - y * math.sin(t),
            start[1] + x * math.sin(t) + y * math.cos(t),
        )
        faces_t = tuple(faces)
        shape = Shape(
            "polish", faces_t, faces_t, tuple(turns), float(new.area), float(sum(faces))
        )
        return shape, world, heading
