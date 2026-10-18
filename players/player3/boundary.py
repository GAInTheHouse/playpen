"""Seed a score-guided beam along multiple open room faces."""

import math
import time
from itertools import pairwise

from shapely.geometry import LineString
from shapely.geometry.polygon import orient
from shapely.prepared import prep

from src.constants import MAX_FACE_LENGTH, TOL


def boundary_starts(room, lengths, deadline, limit=36):
    """Rank boundary faces by nearby interior space; keep CCW headings.

    Whole faces start at a corner or end at a corner where possible. Extra
    midpoint placements allow physical integer lengths to fit fractional room
    edges. No rectangle seeds or arbitrary rotation sweep are used.
    """
    polygon = orient(room.polygon, sign=1.0)
    prepared = prep(polygon.buffer(TOL))
    coords = list(polygon.exterior.coords)
    families = []
    radius = min(MAX_FACE_LENGTH, max(lengths))
    for a, b in pairwise(coords):
        if time.process_time() >= deadline:
            break
        edge = LineString((a, b))
        size = edge.length
        ux, uy = (b[0] - a[0]) / size, (b[1] - a[1]) / size
        # Flat caps measure space beside the face, without giving tiny edges
        # an artificial advantage from a large circular endpoint buffer.
        openness = polygon.intersection(edge.buffer(radius, cap_style=2)).area / size
        seeds, seen = [], set()
        for length in sorted(lengths, reverse=True):
            if length > size + TOL:
                continue
            for offset in (0.0, size - length, (size - length) / 2):
                start = (a[0] + offset * ux, a[1] + offset * uy)
                key = tuple(round(v, 6) for v in start)
                if key in seen:
                    continue
                seen.add(key)
                end = (start[0] + length * ux, start[1] + length * uy)
                if prepared.covers(LineString((start, end))):
                    seeds.append((start, math.degrees(math.atan2(uy, ux)), length))
        if seeds:
            families.append((openness, size, seeds))
    families.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # First try the most open face, then other room faces before revisiting
    # different placements on a single face.
    result = []
    for index in range(limit):
        for _, _, seeds in families:
            if index < len(seeds):
                result.append(seeds[index])
                if len(result) == limit:
                    return result
    return result


def boundary_search(
    room,
    inventory,
    catalog,
    weights,
    deadline,
    consider,
    *,
    branches=24,
    compactness=1.0,
    max_faces=24,
):
    """Run a diverse layered beam from multiple open boundary faces.

    Every seed explores all attainable first-face lengths. Each start gets a
    bounded time share, then alternatives compete on score within each layer.
    """
    from .beam import face_beam

    if not catalog.lengths or time.process_time() >= deadline:
        return {}
    seeds = boundary_starts(room, catalog.lengths, deadline)
    starts, seen = [], set()
    for point, angle, _ in seeds:
        key = (round(point[0], 6), round(point[1], 6), round(angle, 6))
        if key not in seen:
            seen.add(key)
            starts.append((point, angle))
    return face_beam(
        room,
        inventory,
        catalog,
        weights,
        deadline,
        consider,
        beam_width=branches,
        max_faces=max_faces,
        starts=starts,
        boundary_guided=True,
        compactness=compactness,
    )
