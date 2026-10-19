"""Coarse free-space routes used only as construction-search heuristics."""

import heapq
import math
import time
from dataclasses import dataclass

from shapely.geometry import LineString, Point, box
from shapely.prepared import prep


@dataclass
class Route:
    face: tuple
    path: tuple
    target: tuple
    value: float
    cost: float
    territory: frozenset

    def progress(self, polygon):
        """Credit contiguous path progress; distance helps before a cell is reached."""
        prepared = prep(polygon.buffer(1e-7))
        covered = 0
        for point in self.path:
            if not prepared.covers(Point(point)):
                break
            covered += 1
        if covered == len(self.path):
            return 1.0
        gap = polygon.distance(Point(self.path[covered]))
        return (covered + 1 / (1 + gap)) / max(1, len(self.path))


class RoomMap:
    """Clipped grid cells connected only by segments contained in the room.

    Clipping cells instead of testing their centers retains narrow passages.
    Separate pieces of a clipped cell get separate nodes, so a coarse cell
    spanning a thin wall cannot silently connect its two sides.
    """

    def __init__(self, polygon, minimum_piece, deadline=math.inf, resolution=48):
        self.polygon = polygon
        self.prepared = prep(polygon.buffer(1e-7))
        self.minimum_piece = minimum_piece
        minx, miny, maxx, maxy = polygon.bounds
        self.origin = minx, miny
        self.step = max(minimum_piece / 2, max(maxx - minx, maxy - miny) / resolution)
        self.points, self.areas, self.widths, self.cells = [], [], [], []
        self.buckets, self.neighbors = {}, []
        self.complete = False
        nx, ny = (
            math.ceil((maxx - minx) / self.step),
            math.ceil((maxy - miny) / self.step),
        )
        for x in range(nx):
            for y in range(ny):
                if time.process_time() >= deadline:
                    return
                cell = box(
                    minx + x * self.step,
                    miny + y * self.step,
                    minx + (x + 1) * self.step,
                    miny + (y + 1) * self.step,
                )
                clipped = polygon.intersection(cell)
                parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                for part in parts:
                    if part.geom_type != "Polygon" or part.area < 1e-8:
                        continue
                    point = part.representative_point()
                    index = len(self.points)
                    self.points.append((point.x, point.y))
                    self.areas.append(part.area)
                    self.widths.append(2 * polygon.boundary.distance(point))
                    self.cells.append((x, y))
                    self.buckets.setdefault((x, y), []).append(index)
                    self.neighbors.append([])
        for i, (x, y) in enumerate(self.cells):
            if time.process_time() >= deadline:
                return
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in self.buckets.get((x + dx, y + dy), ()):
                        if j <= i:
                            continue
                        a, b = self.points[i], self.points[j]
                        if not self.prepared.covers(LineString((a, b))):
                            continue
                        distance = math.dist(a, b)
                        width = min(self.widths[i], self.widths[j])
                        cost = distance * (1 + minimum_piece / (minimum_piece + width))
                        self.neighbors[i].append((j, cost))
                        self.neighbors[j].append((i, cost))
        self.complete = True

    def portals(self, face, enclosure):
        a, b = face
        length = math.dist(a, b)
        nx, ny = (b[1] - a[1]) / length, (a[0] - b[0]) / length
        interior = prep(enclosure.buffer(-1e-6))
        candidates = {}
        for fraction in (0.25, 0.5, 0.75):
            origin = (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
            x = math.floor((origin[0] - self.origin[0]) / self.step)
            y = math.floor((origin[1] - self.origin[1]) / self.step)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for i in self.buckets.get((x + dx, y + dy), ()):
                        point = self.points[i]
                        forward = (point[0] - origin[0]) * nx + (
                            point[1] - origin[1]
                        ) * ny
                        if forward <= 1e-7:
                            continue
                        segment = LineString((origin, point))
                        if not self.prepared.covers(segment) or interior.intersects(
                            segment
                        ):
                            continue
                        candidates[i] = min(candidates.get(i, math.inf), segment.length)
        return sorted(candidates.items(), key=lambda item: item[1])[:4]

    def routes(self, enclosure, faces, claimed=frozenset(), deadline=math.inf):
        """Rank face/target routes by discounted reachable area and travel cost.

        Claimed target regions are discounted to give other fronts different
        destinations. The map never rejects a geometric construction.
        """
        if not self.complete:
            return []
        interior = prep(enclosure.buffer(-1e-6))
        blocked = {i for i, p in enumerate(self.points) if interior.covers(Point(p))}
        scale = max(4 * self.minimum_piece, math.sqrt(self.polygon.area) / 3)
        routes = []
        edge_visibility = {}
        for face in faces:
            if time.process_time() >= deadline:
                break
            roots = self.portals(face, enclosure)
            costs, parents, heap = {}, {}, []
            for i, cost in roots:
                if i not in blocked:
                    costs[i], parents[i] = cost, None
                    heapq.heappush(heap, (cost, i))
            visits = 0
            while heap:
                cost, i = heapq.heappop(heap)
                if cost != costs[i]:
                    continue
                visits += 1
                if visits % 64 == 0 and time.process_time() >= deadline:
                    break
                for j, edge_cost in self.neighbors[i]:
                    if j in blocked:
                        continue
                    # Room edges are already checked; this second check prevents
                    # planning through the enclosure we are trying to grow.
                    edge_key = (min(i, j), max(i, j))
                    if edge_key not in edge_visibility:
                        edge_visibility[edge_key] = not interior.intersects(
                            LineString((self.points[i], self.points[j]))
                        )
                    if not edge_visibility[edge_key]:
                        continue
                    turn_cost = 0.0
                    parent = parents[i]
                    if parent is not None:
                        a, b, c = self.points[parent], self.points[i], self.points[j]
                        u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
                        if abs(u[0] * v[1] - u[1] * v[0]) > 1e-6:
                            turn_cost = self.minimum_piece * 0.25
                    candidate = cost + edge_cost + turn_cost
                    if candidate < costs.get(j, math.inf):
                        costs[j], parents[j] = candidate, i
                        heapq.heappush(heap, (candidate, j))
            regions = {}
            reachable = 0.0
            for i, cost in costs.items():
                discount = 0.15 if i in claimed else 1.0
                reward = self.areas[i] * discount / (1 + cost / scale)
                reachable += reward
                x, y = self.cells[i]
                regions.setdefault((x // 4, y // 4), []).append((i, reward))
            options = []
            for region in regions.values():
                target = min((i for i, _ in region), key=lambda i: costs[i])
                distance = costs[target]
                value = sum(reward for _, reward in region)
                value *= (0.2 + min(1.0, distance / scale)) / (
                    1 + distance / (4 * scale)
                )
                options.append((value, target, region))
            for value, target, region in sorted(
                options, key=lambda item: item[0], reverse=True
            )[:3]:
                path, cursor = [], target
                while cursor is not None:
                    path.append(self.points[cursor])
                    cursor = parents[cursor]
                path.reverse()
                routes.append(
                    Route(
                        face,
                        tuple(path),
                        self.points[target],
                        value + 0.05 * reachable,
                        costs[target],
                        frozenset(i for i, _ in region),
                    )
                )
        routes.sort(key=lambda route: route.value, reverse=True)
        return routes
