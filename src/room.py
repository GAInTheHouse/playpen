"""The room footprint the enclosure must be built inside of.

The room polygon may be non-convex but has no holes (see spec). This module
also exposes `Room.contains_enclosure`, the containment check players are
told they can use to sanity-check their own construction before submitting.
"""

import math
import random
from tkinter import Canvas

from shapely.geometry import Polygon
from shapely.validation import explain_validity

import src.constants as c


class InvalidRoomException(Exception):
    pass


class Room:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        ok, reason = room_is_ok(polygon)
        if not ok:
            raise InvalidRoomException(f"room is invalid: {reason}")

    def copy(self) -> "Room":
        return Room(Polygon(self.polygon.exterior.coords))

    def contains_enclosure(
        self, enclosure_polygon: Polygon, tol: float = c.TOL
    ) -> bool:
        """Whether `enclosure_polygon` fits entirely within this room, within
        a small floating point tolerance. This is the method exposed to
        players so they can check containment themselves before submitting."""
        return self.polygon.buffer(tol).contains(enclosure_polygon)

    def get_boundary_points(self) -> list[tuple[float, float]]:
        return list(self.polygon.exterior.coords[:-1])

    def draw(self, canvas: Canvas, scale: float, x_off: float, y_off: float):
        """Draw the room outline; a point (x, y) lands at
        (x * scale + x_off, y * scale + y_off)."""
        pts = []
        for x, y in self.get_boundary_points():
            pts.extend([x * scale + x_off, y * scale + y_off])
        canvas.create_polygon(pts, outline=c.ROOM_OUTLINE, fill=c.ROOM_FILL, width=3)


def room_is_ok(p: Polygon) -> tuple[bool, str]:
    if not p.is_valid:
        return False, explain_validity(p)
    if p.is_empty:
        return False, "room is empty"
    if p.area <= 0:
        return False, "room has zero area"
    if len(p.interiors) > 0:
        return False, "room polygon must not have holes"
    return True, ""


def generate_room(
    rng: random.Random, min_extent: float = 25, max_extent: float = 45
) -> Room:
    """Generate a random simple polygon (often non-convex) as a room footprint.

    Uses the classic "star-shaped" construction: pick a center, sample angles
    around the full circle and sort them, then sample a radius per angle.
    Because angles are sorted and radii are measured from a common center,
    the resulting polygon is guaranteed to be simple (non-self-intersecting).
    """
    num_vertices = rng.randint(7, 14)
    cx, cy = max_extent / 2, max_extent / 2

    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(num_vertices))
    base_r = (min_extent + max_extent) / 4
    points = []
    for angle in angles:
        # vary the radius to create concave "bites" out of the shape
        r = base_r * rng.uniform(0.45, 1.0)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append((round(x, 2), round(y, 2)))

    poly = Polygon(points)
    if not poly.is_valid or poly.area <= 0:
        # exceedingly unlikely with the star-shaped construction, but fall
        # back to a plain rectangle rather than fail outright
        return Room(
            Polygon(
                [(0, 0), (max_extent, 0), (max_extent, max_extent), (0, max_extent)]
            )
        )
    return Room(poly)
