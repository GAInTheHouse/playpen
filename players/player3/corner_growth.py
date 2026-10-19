"""Four interleaved, closed growth fronts sharing one physical inventory."""

import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

from src.constants import TOL
from src.enclosure import score_construction, validate_construction


def face_vertices(polygon):
    """Remove union-generated straight joints before packing complete faces."""
    if polygon.geom_type != "Polygon" or polygon.interiors or not polygon.is_valid:
        return None
    points = list(orient(polygon, sign=1.0).exterior.coords[:-1])
    changed = True
    while changed and len(points) >= 3:
        changed = False
        result = []
        for i, b in enumerate(points):
            a, c = points[i - 1], points[(i + 1) % len(points)]
            u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
            if abs(u[0] * v[1] - u[1] * v[0]) <= TOL and u[0] * v[0] + u[1] * v[1] >= 0:
                changed = True
            else:
                result.append(b)
        points = result
    return points if len(points) >= 3 else None


def four_corners(polygon):
    """Choose spatially separated actual face corners, starting at one corner."""
    points = face_vertices(polygon) or []
    chosen = points[:1]
    while len(chosen) < min(4, len(points)):
        chosen.append(
            max(
                (p for p in points if p not in chosen),
                key=lambda p: min(math.dist(p, q) for q in chosen),
            )
        )
    return chosen


def growth_patches(polygon, anchors, lengths):
    """Outward closed edge/corner extensions; blocked corners try other edges.

    Moving a short end face outward also extends a long narrow enclosure.
    Partial-face patches create inward corners for following room concavities.
    """
    points = face_vertices(polygon)
    if not points:
        return
    edges = list(pairwise(points + points[:1]))
    edges.sort(
        key=lambda edge: min(
            min(math.dist(p, edge[0]), math.dist(p, edge[1])) for p in anchors
        )
    )
    choices = sorted(lengths, reverse=True)
    for a, b in edges:
        size = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / size, (b[1] - a[1]) / size
        nx, ny = uy, -ux  # exterior is on the right of a CCW face
        widths = [size] + [n for n in choices if n < size - TOL]
        # Spread partial-face sizes instead of spending all time on similar ones.
        if len(widths) > 7:
            widths = [widths[round(i * (len(widths) - 1) / 6)] for i in range(7)]
        for width in widths:
            offsets = (
                (0.0,)
                if abs(width - size) < TOL
                else (0.0, size - width, (size - width) / 2)
            )
            for offset in offsets:
                p = (a[0] + offset * ux, a[1] + offset * uy)
                q = (p[0] + width * ux, p[1] + width * uy)
                for depth in choices:
                    yield Polygon(
                        (
                            p,
                            q,
                            (q[0] + depth * nx, q[1] + depth * ny),
                            (p[0] + depth * nx, p[1] + depth * ny),
                        )
                    )


@dataclass(eq=False)
class Front:
    polygon: Polygon
    construction: object
    anchors: list
    members: set
    pending: object = None
    seen: set = field(default_factory=set)
    exhausted: bool = False


def grow_enclosure(room, inventory, weights, catalog, initial, deadline, *, batch=48):
    """Round-robin growth, merge on contact outside the common seed.

    All current fronts' union must fit a single inventory before committing a
    move. Removed internal walls can be reused. Unmerged fronts are alternative
    finished enclosures; at the deadline the largest is returned.
    """
    seed_result = validate_construction(initial, room, inventory)
    seed = seed_result.polygon
    corners = four_corners(seed)
    fronts = [Front(seed, initial, [point], {i}) for i, point in enumerate(corners)]
    stats = {
        "initial_fronts": len(fronts),
        "growth_candidates": 0,
        "growth_steps": 0,
        "merges": 0,
        "resource_rejections": 0,
        "geometry_rejections": 0,
        "initial_area": seed.area,
        "rounds": 0,
    }
    cache = {}
    prepared_room = prep(room.polygon.buffer(TOL))

    def pack(polygon):
        if time.process_time() >= deadline:
            return None
        vertices = face_vertices(polygon)
        if not vertices or not prepared_room.covers(polygon):
            return None
        key = polygon.normalize().wkb
        if key in cache:
            return cache[key]
        construction = catalog.build(
            vertices, prefer_bonus=weights.G > 0, deadline=deadline
        )
        if construction is not None:
            result = validate_construction(construction, room, inventory)
            if not result.valid:
                construction = None
        if time.process_time() < deadline:
            cache[key] = construction
        return construction

    while fronts and time.process_time() < deadline:
        stats["rounds"] += 1
        progressed = False
        for front in list(fronts):
            if (
                front not in fronts
                or front.exhausted
                or time.process_time() >= deadline
            ):
                continue
            if front.pending is None:
                front.pending = iter(
                    growth_patches(front.polygon, front.anchors, catalog.lengths)
                )
            best = None
            local_deadline = min(deadline, time.process_time() + 0.2)
            for _ in range(batch):
                if time.process_time() >= local_deadline:
                    break
                patch = next(front.pending, None)
                if patch is None:
                    front.exhausted = True
                    break
                stats["growth_candidates"] += 1
                candidate = front.polygon.union(patch)
                key = candidate.normalize().wkb
                if candidate.area <= front.polygon.area + TOL or key in front.seen:
                    continue
                front.seen.add(key)
                if not prepared_room.covers(candidate):
                    stats["geometry_rejections"] += 1
                    continue
                construction = pack(candidate)
                if construction is None:
                    stats["resource_rejections"] += 1
                    continue
                shared = unary_union(
                    [candidate] + [f.polygon for f in fronts if f is not front]
                )
                if pack(shared) is None:
                    stats["resource_rejections"] += 1
                    continue
                rank = candidate.area
                if best is None or rank > best[0]:
                    best = (rank, candidate, construction)
            if best is None:
                continue
            _, front.polygon, front.construction = best
            front.pending, front.exhausted = None, False
            stats["growth_steps"] += 1
            progressed = True
            # The seed belongs to every front. Only new territory counts as
            # meeting; otherwise all four would falsely merge at initialization.
            merging = True
            while merging and time.process_time() < deadline:
                merging = False
                territory = front.polygon.difference(seed)
                for other in list(fronts):
                    if other is front:
                        continue
                    contact = territory.intersection(other.polygon.difference(seed))
                    if contact.area <= TOL and contact.length <= TOL:
                        continue
                    merged = front.polygon.union(other.polygon)
                    construction = pack(merged)
                    if construction is None:
                        continue
                    front.polygon, front.construction = merged, construction
                    front.anchors.extend(other.anchors)
                    front.members.update(other.members)
                    fronts.remove(other)
                    front.pending, front.exhausted = None, False
                    territory = front.polygon.difference(seed)
                    stats["merges"] += 1
                    merging = True
                    break
        if not progressed and all(f.exhausted for f in fronts):
            break
    winner = max(
        fronts,
        key=lambda f: (
            f.polygon.area,
            score_construction(
                validate_construction(f.construction, room, inventory), weights
            ),
        ),
    )
    stats.update(
        active_fronts=len(fronts),
        final_area=winner.polygon.area,
        winning_fronts=sorted(winner.members),
        front_areas=[front.polygon.area for front in fronts],
    )
    return winner.construction, stats
