"""Find template fallbacks, then follow and repair room boundaries."""

import heapq
import math
from itertools import count, pairwise, product, zip_longest

from shapely import union_all
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from shapely.prepared import prep
from src.constants import BASELINE_SCORE, TOL

from .geometry import Budget, SearchStopped, Shape, boundary_shape, record, vertices


def estimate(polygon, weights) -> float:
    return (
        BASELINE_SCORE
        + weights.A * polygon.area
        + weights.C * polygon.length
        + max(0, weights.G)
    )


def templates(stock, weights, room_area: float, budget: Budget):
    """Keep small, narrow, and high-scoring sizes from three convex families."""
    lengths = sorted(stock.options)
    sampled = sorted({lengths[round(i * (len(lengths) - 1) / 5)] for i in range(6)})
    families = ((0, 2, 4, 6), (0, 1, 2, 4, 5, 6), tuple(range(8)))
    queues = []
    for directions in families:
        size = len(directions) // 2
        rights = 4 if size == 2 else 2 if size == 3 else 0
        if rights > stock.rights or len(directions) - rights > stock.diagonals:
            continue
        candidates = []
        generation = budget.slice(min(0.15, budget.remaining / 5))
        try:
            for values in product(sampled if size == 4 else lengths, repeat=size):
                generation.check()
                if size == 2 and values[0] > values[1]:
                    continue
                a, b = values[:2]
                area = a * b
                if size == 3:
                    c = values[2]
                    area = a * c + (a + c) * b / math.sqrt(2)
                elif size == 4:
                    c, d = values[2:]
                    area = a * c + b * d + (a + c) * (b + d) / math.sqrt(2)
                pieces = 2 * sum(stock.minimum_pieces[n] for n in values)
                perimeter = 2 * sum(values)
                if (
                    area > room_area + TOL
                    or perimeter > stock.material
                    or pieces > sum(stock.counts)
                    or pieces - len(directions) > stock.straights
                ):
                    continue
                score = (
                    BASELINE_SCORE
                    + weights.A * area
                    + weights.C * perimeter
                    + max(0, weights.G)
                )
                if score > 0:
                    candidates.append((score, perimeter, values))
        except SearchStopped:
            record(
                "template_generation", "limit", reason="deadline", faces=len(directions)
            )
        small = sorted(candidates, key=lambda item: (item[1], -item[0]))[:6]
        top = sorted(candidates, reverse=True)[:192]
        diverse = sorted(
            (item for item in candidates if all(n in sampled for n in item[2])),
            reverse=True,
        )[:96]
        narrow = sorted(
            (item for item in candidates if size == 2 and min(item[2]) <= 7),
            reverse=True,
        )
        ordered = (
            small
            + narrow
            + [item for batch in zip_longest(diverse, top) for item in batch if item]
        )
        sizes = dict.fromkeys(values for _, _, values in ordered)
        record(
            "template_generation",
            "shortlist",
            faces=len(directions),
            generated=len(candidates),
            retained=len(sizes),
        )
        queues.append((directions, iter(sizes)))
    for batch in zip_longest(*(stream for _, stream in queues)):
        for (directions, _), values in zip(queues, batch):
            budget.check()
            if values is not None:
                yield Shape(directions, values * 2)


def intervals(low, high, pattern, shift):
    start, index = low + shift * sum(pattern) / len(pattern), 0
    while start + pattern[index % len(pattern)] <= high + TOL:
        end = start + pattern[index % len(pattern)]
        yield start, end
        start, index = end, index + 1


def grid_roots(room, stock, budget: Budget):
    """Join contained cells. Fine cells preserve passages; mixed sizes share stock."""
    lengths = sorted(stock.options)
    steps = sorted((n for n in lengths if n <= 15), key=lambda n: (abs(n - 10), -n))
    short = sorted(
        (
            (count, p.length)
            for p, count in stock.items
            if not p.is_gate and p.length <= 10
        ),
        key=lambda item: (-item[0], item[1]),
    )
    patterns = [(1,)] + [(n,) for n in steps[:1]]
    patterns += [
        tuple(sorted(n for _, n in short[:size]))
        for size in (3, 2)
        if len(short) >= size
    ]
    patterns += [(n,) for n in steps[1:]]
    for shift in (0, 0.5):
        for angle in dict.fromkeys((0.0, *room.axes[:2])):
            local_room = rotate(room.polygon, -angle, origin=(0, 0))
            allowed = prep(local_room.buffer(TOL))
            xmin, ymin, xmax, ymax = local_room.bounds
            for pattern in dict.fromkeys(patterns):
                budget.check()
                spacing = sum(pattern) / len(pattern)
                if (xmax - xmin) * (ymax - ymin) / spacing**2 > 10000:
                    continue

                cells = []
                for (x, xx), (y, yy) in product(
                    intervals(xmin, xmax, pattern, shift),
                    intervals(ymin, ymax, pattern, shift),
                ):
                    budget.check()
                    cell = box(x, y, xx, yy)
                    if allowed.covers(cell):
                        cells.append(cell)
                merged = union_all(cells)
                parts = (
                    [merged]
                    if isinstance(merged, Polygon)
                    else getattr(merged, "geoms", ())
                )
                for polygon in sorted(parts, key=lambda p: -p.area):
                    if simple_polygon(polygon):
                        yield Polygon(vertices(polygon)), angle, allowed, max(pattern)


def simple_polygon(polygon) -> bool:
    return (
        isinstance(polygon, Polygon)
        and not polygon.is_empty
        and polygon.is_valid
        and not polygon.interiors
    )


def edges(polygon):
    points = vertices(polygon)
    return [(math.dist(a, b), a, b) for a, b in pairwise((*points, points[0]))]


def repair_cost(polygon, stock) -> float:
    limit = max(stock.options)
    return sum(
        max(0, length - limit) + (round(length) not in stock.options)
        for length, _, _ in edges(polygon)
    )


def rectangle(a, along, normal, first, width, depth):
    return Polygon(
        [
            (a[0] + x * along[0] + y * normal[0], a[1] + x * along[1] + y * normal[1])
            for x, y in (
                (first, 0),
                (first + width, 0),
                (first + width, depth),
                (first, depth),
            )
        ]
    )


def notches(polygon, stock, width, budget):
    """Split a long face without cutting the enclosure into separate parts."""
    limit = max(stock.options)
    depths = [n for n in sorted(stock.options) if n <= width]
    for length, a, b in sorted(edges(polygon), reverse=True):
        if length <= limit:
            continue
        along = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        normal = (-along[1], along[0])
        for first in sorted(stock.options, reverse=True):
            if length - first - width < min(stock.options) - TOL:
                continue
            for depth in depths:
                budget.check()
                cut = rectangle(a, along, normal, first, width, depth)
                if polygon.buffer(TOL).covers(cut):
                    child = polygon.difference(cut)
                    if simple_polygon(child):
                        yield child


def corner_cuts(polygon, stock, budget):
    points = vertices(polygon)
    for i, b in enumerate(points):
        a, c = points[i - 1], points[(i + 1) % len(points)]
        left, right = math.dist(a, b), math.dist(b, c)
        u = ((a[0] - b[0]) / left, (a[1] - b[1]) / left)
        v = ((c[0] - b[0]) / right, (c[1] - b[1]) / right)
        if u[0] * v[1] - u[1] * v[0] >= -TOL:
            continue
        if round(left) in stock.options and round(right) in stock.options:
            continue
        widths = sorted(
            (n for n in stock.options if n < left),
            key=lambda n: (round(left - n) not in stock.options, n),
        )[:3]
        depths = sorted(
            (n for n in stock.options if n < right),
            key=lambda n: (round(right - n) not in stock.options, n),
        )[:3]
        for width, depth in product(widths, depths):
            budget.check()
            cut = rectangle(b, u, v, 0, width, depth)
            if polygon.buffer(TOL).covers(cut):
                child = polygon.difference(cut)
                if simple_polygon(child):
                    yield child


def repair(polygon, stock, width, budget):
    """Repair long boundaries greedily before the small boundary search."""
    original = polygon
    for cuts in (False, True):
        polygon = original
        for _ in range(250):
            budget.check()
            if len(vertices(polygon)) > min(500, stock.rights, sum(stock.counts)):
                return None
            cost = repair_cost(polygon, stock)
            if not cost:
                return polygon
            candidates = notches(polygon, stock, width, budget)
            if cuts:
                candidates = corner_cuts(polygon, stock, budget)
            replacement = None
            best_rank = None
            for child in candidates:
                if len(vertices(child)) > min(500, stock.rights, sum(stock.counts)):
                    continue
                lengths = [length for length, _, _ in edges(child)]
                if any(
                    round(n) not in stock.options
                    for n in lengths
                    if n <= max(stock.options) + TOL
                ):
                    continue
                remaining = repair_cost(child, stock)
                rank = remaining, -child.area
                if remaining < cost - TOL and (best_rank is None or rank < best_rank):
                    replacement = child
                    best_rank = rank
                    if not cuts:
                        break
            if replacement is None:
                break
            polygon = replacement
    return None


def boundary_edits(polygon, stock, allowed, budget, growth):
    """Try large clips first, then local cuts and optional outward rectangles."""
    points = vertices(polygon)
    xmin, ymin, xmax, ymax = polygon.bounds
    step = min(stock.options)
    for axis, low, high in ((0, xmin, xmax), (1, ymin, ymax)):
        positions = sorted(
            {p[axis] for p in points if low + step <= p[axis] <= high - step}
        )
        for cut in dict.fromkeys(positions[:3] + positions[-3:]):
            bounds = (
                ((cut, ymin, xmax, ymax), (xmin, ymin, cut, ymax))
                if axis == 0
                else ((xmin, cut, xmax, ymax), (xmin, ymin, xmax, cut))
            )
            for bounds_part in bounds:
                budget.check()
                child = polygon.intersection(box(*bounds_part))
                if simple_polygon(child):
                    yield child
    yield from notches(polygon, stock, step, budget)
    if not growth:
        return
    yield from corner_cuts(polygon, stock, budget)
    produced = 0
    for length, a, b in edges(polygon):
        along = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        outward = (along[1], -along[0])
        for width, depth in product(sorted(stock.options)[:3], repeat=2):
            for first in dict.fromkeys((0, length - width, (length - width) / 2)):
                budget.check()
                if first < 0 or abs(first - round(first)) > TOL:
                    continue
                child = polygon.union(rectangle(a, along, outward, first, width, depth))
                if (
                    simple_polygon(child)
                    and child.area > polygon.area + TOL
                    and allowed.covers(child)
                ):
                    yield child
                    produced += 1
                    if produced == 32:
                        return


def boundaries(room, stock, weights, budget: Budget):
    """Repair complete footprints, then search a bounded set of smaller edits."""
    if stock.rights < 4:
        record("grid", "stop", reason="insufficient_right_connectors")
        return
    record("grid", "start")
    roots, seen = [], set()
    root_budget = budget.slice(min(3, budget.remaining / 2))
    try:
        for polygon, angle, allowed, width in grid_roots(room, stock, root_budget):
            key = angle, polygon.normalize().wkb
            if key in seen:
                continue
            seen.add(key)
            roots.append((polygon, angle, allowed))
            try:
                fixed = repair(polygon, stock, width, root_budget.slice(0.3))
                if fixed is not None:
                    yield boundary_shape(fixed, angle)
            except (ValueError, SearchStopped):
                pass
    except SearchStopped:
        record("grid", "stop", reason="deadline", roots=len(roots))
    else:
        record("grid", "stop", reason="grid_stream_exhausted", roots=len(roots))
    for growth in (False, True):
        local = budget.slice(budget.remaining / 2) if not growth else budget
        yield from boundary_pass(roots, stock, weights, local, growth)


def boundary_pass(roots, stock, weights, budget, growth):
    queue, visited, serial = [], set(), count()
    stage = "boundary_mixed" if growth else "boundary_inward"
    expanded = 0
    pruned = 0
    record(stage, "start")

    def push(polygon, angle, allowed):
        nonlocal pruned
        budget.check()
        key = angle, polygon.normalize().wkb
        if key in visited or len(vertices(polygon)) > 32:
            return
        visited.add(key)
        rank = estimate(polygon, weights) - 25 * repair_cost(polygon, stock)
        heapq.heappush(queue, (-rank, next(serial), polygon, angle, allowed))
        if len(queue) > 32:
            pruned += 1
            queue[:] = heapq.nsmallest(32, queue)
            heapq.heapify(queue)

    try:
        for polygon, angle, allowed in roots:
            push(polygon, angle, allowed)
        for _ in range(1000):
            budget.check()
            if not queue:
                break
            expanded += 1
            _, _, polygon, angle, allowed = heapq.heappop(queue)
            try:
                yield boundary_shape(polygon, angle)
            except ValueError:
                pass
            for child in boundary_edits(polygon, stock, allowed, budget, growth):
                push(child, angle, allowed)
    except SearchStopped:
        record(stage, "stop", reason="deadline", expanded=expanded, queue_pruned=pruned)
    else:
        record(
            stage,
            "stop",
            reason="state_limit" if queue else "queue_exhausted",
            expanded=expanded,
            queue_pruned=pruned,
        )
