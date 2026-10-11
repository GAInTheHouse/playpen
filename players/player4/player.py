"""Group 4: face-level geometric beam search, with a safe rectangle fallback.

Only this file needs to be copied into players/player4/player.py.  Search constants
marked EMPIRICAL are intentionally initial guesses, not validated optimal values.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache

from shapely import affinity
from shapely.geometry import LineString, Polygon

from players.player0 import Player
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

# EMPIRICAL: tune these on held-out scenarios for score versus CPU cost.
BEAM_WIDTH = 110
MAX_FACES = 8
MAX_LENGTH_CHOICES = 22
SEARCH_CPU_SECONDS = 35.0  # Intentionally conservative vs simulator's 300s default.
MAX_COMPLETE_POLYGONS = 240
MAX_PLACEMENT_TESTS = 450

# Heading vectors have coordinates a + b*sqrt(2)/2, all EXACT integers.
DIRECTIONS = (
    (1, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 0),
    (0, -1, 0, 1), (-1, 0, 0, 0), (0, -1, 0, -1),
    (0, 0, -1, 0), (0, 1, 0, -1),
)
TURNS = (2, 1, -1, -2)
SQRT2_HALF = math.sqrt(2) / 2


def real_point(q):
    return (q[0] + q[1] * SQRT2_HALF, q[2] + q[3] * SQRT2_HALF)


def advance(pos, heading, length):
    vec = DIRECTIONS[heading]
    return tuple(pos[i] + length * vec[i] for i in range(4))


def edge_headings(polygon):
    points = list(polygon.exterior.coords)
    return sorted({round(math.degrees(math.atan2(y2-y1, x2-x1)) % 45, 8)
                   for (x1, y1), (x2, y2) in zip(points, points[1:])})


@dataclass(frozen=True)
class State:
    lengths: tuple[int, ...] = ()
    turns: tuple[int, ...] = ()  # Turn AFTER each face, including final closing turn.
    path: tuple[tuple[int, int, int, int], ...] = ((0, 0, 0, 0),)
    heading: int = 0  # heading for NEXT face
    turn_sum: int = 0
    perimeter: int = 0


def face_options(inventory):
    """All individually constructible lengths (bounded knapsack, 30-unit cap).

    Joint allocation is checked later; this alone does not ensure enough
    pieces exist to realize all faces simultaneously.
    """
    stocks = [(length, count) for pool in (inventory.walls, inventory.gates)
              for length, count in pool.items() if count > 0]
    max_pieces = inventory.connectors.get(ConnectorType.STRAIGHT, 0) + 1
    reachable = {(0, 0)}  # (length, piece count)
    for length, count in stocks:
        for _ in range(min(count, 30 // length)):
            reachable |= {(s + length, n + 1) for s, n in tuple(reachable)
                          if s + length <= 30 and n < max_pieces}
    options = sorted({s for s, n in reachable if s >= 5 and n >= 1})
    if len(options) > MAX_LENGTH_CHOICES:
        # EMPIRICAL: sampled lengths keep the branching factor manageable.
        # Include small and large lengths; the right sampling policy needs a benchmark.
        ix = {round(i * (len(options) - 1) / (MAX_LENGTH_CHOICES - 1))
              for i in range(MAX_LENGTH_CHOICES)}
        options = [options[i] for i in sorted(ix)]
    return options


def partial_simple(state, new_pos):
    """Reject crossings and retracing; a future closing edge is allowed."""
    a = real_point(state.path[-1])
    b = real_point(new_pos)
    if a == b:
        return False
    segment = LineString((a, b))
    points = [real_point(p) for p in state.path]
    for i in range(len(points) - 2):  # Last existing segment is adjacent.
        if segment.intersects(LineString((points[i], points[i+1]))):
            # Meeting the start is permitted ONLY as the terminal edge.
            if new_pos == state.path[0] and i == 0:
                intersection = segment.intersection(LineString((points[i], points[i+1])))
                if intersection.geom_type == 'Point' and intersection.distance(
                        LineString((points[0], points[0])).centroid) < 1e-7:
                    continue
            return False
    return True


def terminal(state):
    return (len(state.lengths) >= 3 and state.path[-1] == state.path[0]
            and state.heading == 0 and state.turn_sum == 8)


def generate_child(state, length, turn):
    endpoint = advance(state.path[-1], state.heading, length)
    if not partial_simple(state, endpoint):
        return None
    return State(state.lengths + (length,), state.turns + (turn,),
                 state.path + (endpoint,), (state.heading + turn) % 8,
                 state.turn_sum + turn, state.perimeter + length)


def make_connector(turn):
    kind = ConnectorType.RIGHT if abs(turn) == 2 else ConnectorType.DIAGONAL
    return Connector(kind, reflex=(turn < 0))


def allocate(lengths, turns, inventory, prefer_two_gates):
    """Joint exact piece allocation across all faces, with shared connector counts.

    Returns per-face sequences of Piece, favoring distinct-face gates if requested.
    Bounded recursive solver; the inventory is never mutated.
    """
    right = sum(abs(t) == 2 for t in turns)
    diag = len(turns) - right
    if (right > inventory.connectors.get(ConnectorType.RIGHT, 0)
            or diag > inventory.connectors.get(ConnectorType.DIAGONAL, 0)):
        return None
    items = [(PieceType.WALL, int(k), int(n)) for k, n in inventory.walls.items() if n]
    items += [(PieceType.GATE, int(k), int(n)) for k, n in inventory.gates.items() if n]
    items.sort(key=lambda x: (x[0] != PieceType.GATE, -x[1]))
    counts = tuple(n for _, _, n in items)
    straight_budget = inventory.connectors.get(ConnectorType.STRAIGHT, 0)
    total_length = sum(k * n for _, k, n in items)
    if total_length < sum(lengths) or not any(t == PieceType.GATE for t, _, _ in items):
        return None

    @lru_cache(maxsize=150000)
    def solve_face(i, remaining, straights, gate_faces):
        if i == len(lengths):
            if gate_faces >= (2 if prefer_two_gates else 1):
                return ()
            return None
        target = lengths[i]
        # Dynamic enumeration of every in-face combination whose sum is target.
        combinations = []
        def combinations_dfs(j, left, used, selection, gate_used):
            if left == 0:
                if used >= 1 and used - 1 <= straights:
                    combinations.append((tuple(selection) + (0,) * (len(items)-len(selection)), gate_used, used - 1))
                return
            if j == len(items) or used - 1 >= straights:
                return
            typ, size, _ = items[j]
            max_take = min(remaining[j], left // size, straights + 1 - used)
            for n in range(max_take, -1, -1):
                selection.append(n)
                combinations_dfs(j+1, left-n*size, used+n, selection,
                                 gate_used or (n > 0 and typ == PieceType.GATE))
                selection.pop()
        combinations_dfs(0, target, 0, [], False)
        # EMPIRICAL: prioritize allocations with fewer joins to preserve straights.
        combinations.sort(key=lambda c: c[2])
        for picked, has_gate, joins in combinations:
            following = tuple(n - use for n, use in zip(remaining, picked))
            result = solve_face(i+1, following, straights-joins,
                                min(2, gate_faces + int(has_gate)))
            if result is not None:
                face = tuple(Piece(items[j][0], items[j][1])
                             for j, num in enumerate(picked) for _ in range(num))
                return (face,) + result
        return None
    try:
        return solve_face(0, counts, straight_budget, 0)
    except RecursionError:
        return None


def construction_from_faces(faces, turns, start, heading):
    pieces, connectors = [], []
    # Connector at start of face i turns from previous face; face 0 uses final turn.
    for i, face in enumerate(faces):
        corner = turns[i-1] if i else turns[-1]
        for j, piece in enumerate(face):
            pieces.append(piece)
            connectors.append(make_connector(corner) if j == 0
                              else Connector(ConnectorType.STRAIGHT))
    return Construction(start, heading, pieces, connectors)


def placements(path, room, max_tests=MAX_PLACEMENT_TESTS):
    """Yield (start, heading) for the path's local origin and heading zero.

    EMPIRICAL: only finitely many contacts and centre samples are attempted.
    Boundary contact existence does NOT imply this candidate list is complete.
    """
    poly = Polygon([real_point(p) for p in path[:-1]])
    room_vertices = list(room.polygon.exterior.coords)[:-1]
    angles = edge_headings(room.polygon)
    angles += [0., 15., 30.]
    angles = sorted(set(round(a, 7) for a in angles))
    center = room.polygon.centroid
    interior = room.polygon.representative_point()
    seen = set()
    count = 0
    for base in angles:
        for offset in (0, 45, 90, 135, 180, 225, 270, 315):
            angle = base + offset
            rotated = affinity.rotate(poly, angle, origin=(0, 0))
            vertices = list(rotated.exterior.coords)[:-1]
            for cx, cy in ((center.x, center.y), (interior.x, interior.y)):
                center_shift = (cx - rotated.centroid.x, cy - rotated.centroid.y)
                key = (round(center_shift[0], 5), round(center_shift[1], 5), round(angle % 360, 5))
                if key not in seen:
                    seen.add(key)
                    yield center_shift, angle
                    count += 1
                    if count >= max_tests:
                        return
            for x, y in room_vertices:
                for u, v in vertices:
                    translation = (x-u, y-v)
                    key = (round(translation[0], 5), round(translation[1], 5), round(angle % 360, 5))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield translation, angle
                    count += 1
                    if count >= max_tests:
                        return


def rank_state(state, room_area, max_remaining_faces):
    """Heuristic ONLY. Never use this as a safe global upper bound.

    EMPIRICAL: bounding-box area, closure distance, perimeter and turn progress
    need benchmark-based tuning. Preserve shape diversity when selecting beam.
    """
    points = [real_point(p) for p in state.path]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    box_area = (max(xs)-min(xs))*(max(ys)-min(ys))
    dx, dy = points[-1][0], points[-1][1]
    gap = math.hypot(dx, dy)
    turning_gap = abs(8 - state.turn_sum)
    # A room-scale cap prevents unbounded inflated partial boxes.
    return min(box_area, room_area) - 0.7*gap - 2.0*turning_gap - 0.08*state.perimeter


class Player4(Player):
    """Geometry-first beam search: faces -> exact closure -> pieces -> placement."""

    def build_enclosure(self) -> Construction | None:
        start_cpu = time.process_time()
        best, best_score = None, 0.0
        try:
            # Reliable fallback. Its inventory is not consumed by our search.
            from players.example_player import ExamplePlayer
            fallback = ExamplePlayer(self.room, self.inventory, self.weights).build_enclosure()
            if fallback is not None:
                checked = validate_construction(fallback, self.room, self.inventory)
                if checked.valid:
                    best, best_score = fallback, max(0.0, score_construction(checked, self.weights))
        except Exception:
            pass

        if not self.inventory.has_gate():
            return best
        lengths = face_options(self.inventory)
        if not lengths:
            return best
        right_supply = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diag_supply = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        # Deterministic seeds guarantee that beam ranking cannot starve rectangles.
        # EMPIRICAL: increase/decrease rectangle count based on validation benchmarks.
        rectangle_candidates = sorted(
            ((w, h) for w in lengths for h in lengths),
            key=lambda wh: self.weights.A * wh[0] * wh[1]
            + self.weights.C * 2 * (wh[0] + wh[1]), reverse=True,
        )[:100]
        for w, h in rectangle_candidates:
            if time.process_time() - start_cpu >= SEARCH_CPU_SECONDS * 0.30:
                break
            face_lengths = (w, h, w, h)
            turns = (2, 2, 2, 2)
            potential = 1000 + self.weights.A * w*h + self.weights.C*2*(w+h) + max(0, self.weights.G)
            if potential <= best_score:
                continue
            faces = allocate(face_lengths, turns, self.inventory,
                             prefer_two_gates=self.weights.G > 0)
            if faces is None and self.weights.G > 0:
                faces = allocate(face_lengths, turns, self.inventory, False)
            if faces is None:
                continue
            rectangle_path = ((0,0,0,0), (w,0,0,0), (w,0,h,0),
                              (0,0,h,0), (0,0,0,0))
            rectangle_poly = Polygon([real_point(p) for p in rectangle_path[:-1]])
            if rectangle_poly.area > self.room.polygon.area + 1e-6:
                continue
            for (tx, ty), angle in placements(rectangle_path, self.room, max_tests=200):
                moved = affinity.translate(affinity.rotate(rectangle_poly, angle,
                                           origin=(0,0)), tx, ty)
                if not self.room.contains_enclosure(moved):
                    continue
                candidate = construction_from_faces(faces, turns, (tx,ty), angle)
                validation = validate_construction(candidate, self.room, self.inventory)
                if validation.valid:
                    score = score_construction(validation, self.weights)
                    if score > best_score:
                        best, best_score = candidate, score
                    break

        beam = [State()]
        evaluated = 0
        seen_completed = set()
        for depth in range(1, MAX_FACES + 1):
            if time.process_time() - start_cpu >= SEARCH_CPU_SECONDS:
                break
            children = []
            for state in beam:
                if time.process_time() - start_cpu >= SEARCH_CPU_SECONDS:
                    break
                # First edge goes east. Its AFTER-face turn is enumerated as usual.
                for length in lengths:
                    for turn in TURNS:
                        new_turns = state.turns + (turn,)
                        nr = sum(abs(t) == 2 for t in new_turns)
                        nd = len(new_turns) - nr
                        if nr > right_supply or nd > diag_supply:
                            continue
                        turn_sum = state.turn_sum + turn
                        remaining = MAX_FACES - depth
                        if turn_sum + 2*remaining < 8 or turn_sum - 2*remaining > 8:
                            continue
                        # Any future closure must have at least enough length left.
                        child = generate_child(state, length, turn)
                        if child is None:
                            continue
                        if child.path[-1] == child.path[0]:
                            if not terminal(child):
                                continue
                            signature = (child.lengths, child.turns)
                            if signature in seen_completed:
                                continue
                            seen_completed.add(signature)
                            polygon = Polygon([real_point(p) for p in child.path[:-1]])
                            if not polygon.is_valid or polygon.area <= 1e-6:
                                continue
                            if polygon.area > self.room.polygon.area + 1e-5:
                                continue
                            # Check potential score before expensive assignment and placement.
                            upper_score = (1000 + self.weights.A*polygon.area
                                           + self.weights.C*polygon.length
                                           + max(0, self.weights.G))
                            if upper_score <= best_score:
                                continue
                            faces = allocate(child.lengths, child.turns, self.inventory,
                                             prefer_two_gates=self.weights.G > 0)
                            if faces is None and self.weights.G > 0:
                                faces = allocate(child.lengths, child.turns, self.inventory,
                                                 prefer_two_gates=False)
                            if faces is None:
                                continue
                            evaluated += 1
                            for (tx, ty), angle in placements(child.path, self.room):
                                moved = affinity.translate(affinity.rotate(polygon, angle,
                                                         origin=(0, 0)), tx, ty)
                                if not self.room.contains_enclosure(moved):
                                    continue
                                candidate = construction_from_faces(faces, child.turns,
                                                                    (tx, ty), angle)
                                valid = validate_construction(candidate, self.room, self.inventory)
                                if valid.valid:
                                    score = score_construction(valid, self.weights)
                                    if score > best_score:
                                        best, best_score = candidate, score
                                    break
                            if evaluated >= MAX_COMPLETE_POLYGONS:
                                break
                        else:
                            if depth >= MAX_FACES:
                                continue
                            remaining_length = (MAX_FACES-depth)*max(lengths)
                            x,y = real_point(child.path[-1])
                            if math.hypot(x,y) > remaining_length + 1e-7:
                                continue
                            # EMPIRICAL: add a safe exact remaining-direction
                            # reachability DP here; current Euclidean bound is weak.
                            children.append(child)
                    if evaluated >= MAX_COMPLETE_POLYGONS:
                        break
                if evaluated >= MAX_COMPLETE_POLYGONS:
                    break
            if evaluated >= MAX_COMPLETE_POLYGONS or not children:
                break
            children.sort(key=lambda s: rank_state(s, self.room.polygon.area,
                                                  MAX_FACES-depth), reverse=True)
            # EMPIRICAL: encourage varied turn patterns, rather than keeping
            # near-duplicates of the same high-ranked geometric family.
            next_beam, seen = [], set()
            for candidate in children:
                key = (candidate.path[-1], candidate.heading, candidate.turn_sum,
                       candidate.lengths, candidate.turns)
                if key in seen:
                    continue
                seen.add(key)
                next_beam.append(candidate)
                if len(next_beam) >= BEAM_WIDTH:
                    break
            beam = next_beam
        self.note = (f"face beam: {evaluated} buildable closed geometries; "
                     f"best={best_score:.1f}; CPU={time.process_time()-start_cpu:.1f}s")
        return best
