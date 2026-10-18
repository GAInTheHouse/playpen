"""Four interleaved, closed growth fronts sharing one physical inventory."""

import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

from shapely.geometry import LineString, Polygon
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


def face_key(face):
    return tuple(tuple(round(value, 6) for value in point) for point in face)


def outward_normal(face):
    a, b = face
    length = math.dist(a, b)
    return (b[1] - a[1]) / length, (a[0] - b[0]) / length


def ray_clearance(room, origin, direction):
    """Distance to the first room exit, never to a later re-entry wall."""
    minx, miny, maxx, maxy = room.bounds
    reach = 2 * math.hypot(maxx - minx, maxy - miny) + 1
    dx, dy = direction
    ray = LineString((origin, (origin[0] + reach * dx, origin[1] + reach * dy)))
    intersection = room.intersection(ray)

    def contiguous(geometry):
        if geometry.is_empty:
            return 0.0
        if geometry.geom_type == "LineString":
            projections = [
                (x - origin[0]) * dx + (y - origin[1]) * dy for x, y in geometry.coords
            ]
            return max(projections) if min(projections) <= TOL else 0.0
        if hasattr(geometry, "geoms"):
            return max((contiguous(part) for part in geometry.geoms), default=0.0)
        return 0.0

    return max(0.0, contiguous(intersection))


def face_clearance(room, face):
    a, b = face
    direction = outward_normal(face)
    distances = [
        ray_clearance(
            room, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), direction
        )
        for f in (0.05, 0.25, 0.5, 0.75, 0.95)
    ]
    return max(distances), sum(distances) / len(distances)


def ranked_faces(polygon, room, blocked=(), deadline=math.inf):
    """Greatest perpendicular clearance first, with potential area as a tie-break."""
    points = face_vertices(polygon) or []
    ranked = []
    for face in pairwise(points + points[:1]):
        if time.process_time() >= deadline:
            break
        if face_key(face) in blocked:
            continue
        clearance, mean = face_clearance(room, face)
        if clearance > TOL:
            ranked.append((clearance, mean * math.dist(*face), face))
    ranked.sort(key=lambda item: item[:2], reverse=True)
    return [face for _, _, face in ranked]


def starting_faces(polygon, room, deadline=math.inf):
    """Up to four different faces, preferring different outward headings."""
    ranked = ranked_faces(polygon, room, deadline=deadline)
    points = face_vertices(polygon) or []
    # Keep blocked faces as initial fronts too: they will explicitly redirect.
    ranked += [face for face in pairwise(points + points[:1]) if face not in ranked]
    chosen, headings = [], set()
    for face in ranked:
        direction = tuple(round(n, 6) for n in outward_normal(face))
        if direction not in headings:
            chosen.append(face)
            headings.add(direction)
        if len(chosen) == 4:
            return chosen
    return (chosen + [face for face in ranked if face not in chosen])[:4]


def outer_face(polygon, patch, previous):
    """Track the new outer boundary after extending the active face."""
    nx, ny = outward_normal(previous)
    target = max(x * nx + y * ny for x, y in patch.exterior.coords)
    points = face_vertices(polygon) or []
    for face in pairwise(points + points[:1]):
        normal = outward_normal(face)
        if normal[0] * nx + normal[1] * ny < 1 - 1e-6:
            continue
        if (
            all(abs(x * nx + y * ny - target) <= TOL for x, y in face)
            and LineString(face).distance(patch) <= TOL
        ):
            return face
    return None


def growth_patches(polygon, faces, lengths):
    """Try full and partial extensions perpendicular to the selected face only."""
    choices = sorted(lengths, reverse=True)
    for a, b in faces:
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
    active_face: object
    members: set
    pending: object = None
    seen: set = field(default_factory=set)
    exhausted: bool = False
    blocked: set = field(default_factory=set)


def grow_enclosure(room, inventory, weights, catalog, initial, deadline, *, batch=48):
    """Round-robin growth, merge on contact outside the common seed.

    All current fronts' union must fit a single inventory before committing a
    move. Removed internal walls can be reused. Unmerged fronts are alternative
    finished enclosures; at the deadline the largest is returned.
    """
    seed_result = validate_construction(initial, room, inventory)
    seed = seed_result.polygon
    faces = starting_faces(seed, room.polygon, deadline)
    fronts = [Front(seed, initial, face, {i}) for i, face in enumerate(faces)]
    stats = {
        "initial_fronts": len(fronts),
        "growth_candidates": 0,
        "growth_steps": 0,
        "merges": 0,
        "resource_rejections": 0,
        "geometry_rejections": 0,
        "initial_area": seed.area,
        "rounds": 0,
        "face_switches": 0,
        "face_switch_history": [],
        "initial_faces": [face_key(face) for face in faces],
        "growth_history": [],
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
                if front.active_face is None:
                    ranked = ranked_faces(
                        front.polygon, room.polygon, front.blocked, deadline
                    )
                    claimed = {
                        face_key(f.active_face)
                        for f in fronts
                        if f is not front and f.active_face is not None
                    }
                    available = [
                        face for face in ranked if face_key(face) not in claimed
                    ]
                    if not available:
                        available = ranked
                    if not available:
                        front.exhausted = True
                        continue
                    front.active_face = available[0]
                    stats["face_switches"] += 1
                    stats["face_switch_history"].append(
                        {
                            "fronts": sorted(front.members),
                            "face": face_key(front.active_face),
                            "clearance": face_clearance(
                                room.polygon, front.active_face
                            )[0],
                        }
                    )
                front.pending = iter(
                    growth_patches(front.polygon, [front.active_face], catalog.lengths)
                )
            best = None
            selected_face = front.active_face
            local_deadline = min(deadline, time.process_time() + 0.2)
            for _ in range(batch):
                if time.process_time() >= local_deadline:
                    break
                patch = next(front.pending, None)
                if patch is None:
                    front.blocked.add(face_key(front.active_face))
                    front.active_face, front.pending = None, None
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
                    best = (rank, candidate, construction, patch)
            if best is None:
                continue
            previous_face = front.active_face
            # The iterator may have exhausted after finding a valid proposal.
            if previous_face is None:
                previous_face = selected_face
            _, front.polygon, front.construction, patch = best
            front.active_face = outer_face(front.polygon, patch, previous_face)
            front.pending, front.exhausted = None, False
            stats["growth_steps"] += 1
            progressed = True
            stats["growth_history"].append(
                {
                    "fronts": sorted(front.members),
                    "face": face_key(previous_face),
                    "direction": outward_normal(previous_face),
                    "area": front.polygon.area,
                }
            )
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
                    front.members.update(other.members)
                    fronts.remove(other)
                    front.pending, front.exhausted = None, False
                    front.blocked.clear()
                    if front.active_face is not None:
                        current = face_vertices(front.polygon)
                        if face_key(front.active_face) not in {
                            face_key(face) for face in pairwise(current + current[:1])
                        }:
                            front.active_face = None
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
