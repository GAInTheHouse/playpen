"""Integer-face polygon template search for the Playpen simulator."""

import math
import signal
import time
from functools import lru_cache
from itertools import chain, product, zip_longest

from players.player0 import Player
from shapely.geometry import Polygon
from shapely.prepared import prep
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player2(Player):
    """Search rectangles, octagons, L-shapes, and 90/135-degree hexagons."""

    # Leave ample headroom under the simulator's default 300 CPU seconds.
    # Lower this when testing with a smaller --cpu-limit.
    SEARCH_CPU_SECONDS = 45.0
    ALLOCATION_NODE_LIMIT = 2000
    ALLOCATION_CPU_SECONDS = 0.04
    PLACEMENT_CPU_SECONDS = 0.03
    MAX_CANDIDATES_PER_FAMILY = 500

    def build_enclosure(self) -> Construction | None:
        started = time.process_time()
        budget = self.SEARCH_CPU_SECONDS
        # The submission checker defaults to 60s, including initialization.
        # On Unix, respect an already-running simulator timer with headroom.
        if hasattr(signal, "getitimer") and hasattr(signal, "ITIMER_PROF"):
            remaining, _ = signal.getitimer(signal.ITIMER_PROF)
            if remaining > 0:
                budget = min(budget, remaining * 0.7)
        deadline = started + budget
        wall_deadline = time.monotonic() + budget * 1.5

        def expired():
            return time.process_time() >= deadline or time.monotonic() >= wall_deadline

        print(
            f"Playpen: searching integer-sided templates ({budget:.1f}s CPU budget)",
            flush=True,
        )
        best = None
        best_score = 0.0  # Returning None is preferable to a negative score.
        straight = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)

        types = []
        stock = []
        for kind, pool in (
            (PieceType.WALL, self.inventory.walls),
            (PieceType.GATE, self.inventory.gates),
        ):
            for length, count in sorted(pool.items()):
                if count > 0 and length <= MAX_FACE_LENGTH:
                    types.append(Piece(kind, length))
                    stock.append(count)
        if not any(piece.is_gate for piece in types):
            self.note = "No usable gate of length at most 30."
            return None

        # Enumerate unordered face assemblies. Each face contains at most six
        # pieces; preserve different resource allocations for the same length.
        faces = {}
        counts = [0] * len(types)
        face_deadline = time.process_time() + min(0.4, budget * 0.08)

        def enumerate_faces(first, length, number, has_gate):
            if expired() or time.process_time() >= face_deadline:
                return
            if number == face_piece_limit:
                faces.setdefault(length, []).append(
                    (tuple(counts), number - 1, has_gate)
                )
            if number >= face_piece_limit:
                return
            for i in range(first, len(types)):
                new_length = length + types[i].length
                if new_length > MAX_FACE_LENGTH or counts[i] >= stock[i]:
                    continue
                counts[i] += 1
                enumerate_faces(i, new_length, number + 1, has_gate or types[i].is_gate)
                counts[i] -= 1

        for face_piece_limit in range(1, min(6, straight + 1) + 1):
            enumerate_faces(0, 0, 0, False)
            if expired() or time.process_time() >= face_deadline:
                break
        for options in faces.values():
            options.sort(key=lambda option: (option[1], not option[2]))

        def allocate(lengths, required_gate_faces):
            """Bounded backtracking with shared inventory and connector counts."""
            nodes = 0
            allocation_deadline = min(
                deadline, time.process_time() + self.ALLOCATION_CPU_SECONDS
            )

            @lru_cache(maxsize=20000)
            def search(side, remaining, available_straight, gate_faces):
                nonlocal nodes
                nodes += 1
                if (
                    nodes > self.ALLOCATION_NODE_LIMIT
                    or expired()
                    or time.process_time() >= allocation_deadline
                ):
                    return None
                if side == len(lengths):
                    return () if gate_faces >= required_gate_faces else None
                if gate_faces + len(lengths) - side < required_gate_faces:
                    return None
                for usage, joins, has_gate in faces[lengths[side]]:
                    if expired() or time.process_time() >= allocation_deadline:
                        return None
                    if joins > available_straight:
                        continue
                    if any(used > left for used, left in zip(usage, remaining)):
                        continue
                    rest = search(
                        side + 1,
                        tuple(left - used for used, left in zip(usage, remaining)),
                        available_straight - joins,
                        min(required_gate_faces, gate_faces + int(has_gate)),
                    )
                    if rest is not None:
                        return (usage,) + rest
                return None

            return search(0, tuple(stock), straight, 0)

        room = self.room.polygon
        buffered_room = prep(room.buffer(TOL))
        corners = list(room.exterior.coords)[:-1]
        headings = set(range(0, 180, 15))
        room_headings = set()
        for p, q in zip(corners, corners[1:] + corners[:1]):
            angle = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            headings.add(round(angle % 180, 9))
            headings.add(round((angle + 90) % 180, 9))
            room_headings.add(round(angle % 180, 9))
        centers = [(room.centroid.x, room.centroid.y)]
        inside = room.representative_point()
        centers.append((inside.x, inside.y))
        # Test small offsets near the center before boundary anchors. Large
        # shapes can fit between notches while failing at every room corner.
        for dx, dy in ((0, -4), (0, 4)) + (
            (0, -5),
            (0, 5),
            (-5, 0),
            (5, 0),
            (-5, -5),
            (5, -5),
            (-5, 5),
            (5, 5),
        ):
            centers.append((room.centroid.x + dx, room.centroid.y + dy))
        minx, miny, maxx, maxy = room.bounds
        for i in range(1, 10):
            for j in range(1, 10):
                centers.append(
                    (minx + (maxx - minx) * i / 10, miny + (maxy - miny) * j / 10)
                )

        def placement(vertices, hint=None):
            placement_deadline = min(
                deadline, time.process_time() + self.PLACEMENT_CPU_SECONDS
            )
            local = Polygon(vertices)
            center = local.centroid
            if hint is not None:
                hint_center, hint_heading = hint
                for delta in (0, -0.5, 0.5, -1, 1, -2, 2):
                    heading = hint_heading + delta
                    theta = math.radians(heading)
                    co, si = math.cos(theta), math.sin(theta)
                    offsets = [(x * co - y * si, x * si + y * co) for x, y in vertices]
                    cx = center.x * co - center.y * si
                    cy = center.x * si + center.y * co
                    for dx, dy in product((0, -0.5, 0.5, -1, 1, -2, 2), repeat=2):
                        if expired() or time.process_time() >= placement_deadline:
                            return None
                        x, y = hint_center[0] + dx - cx, hint_center[1] + dy - cy
                        polygon = Polygon([(x + ox, y + oy) for ox, oy in offsets])
                        if buffered_room.contains(polygon):
                            return (x, y), heading
            # Non-symmetric templates need the full circle. Include alignment
            # of EVERY template edge with EVERY room edge, not just edge zero.
            angles = set(range(0, 360, 15))
            for heading in headings:
                for offset in range(0, 360, 45):
                    angles.add(round((heading + offset) % 360, 9))
            preferred = {
                (heading + offset) % 360
                for heading in room_headings
                for offset in range(0, 360, 45)
            }
            ordered_angles = sorted(preferred) + sorted(angles - preferred)
            for heading in ordered_angles:
                if expired() or time.process_time() >= placement_deadline:
                    return None
                theta = math.radians(heading)
                co, si = math.cos(theta), math.sin(theta)
                offsets = [(x * co - y * si, x * si + y * co) for x, y in vertices]
                cx = center.x * co - center.y * si
                cy = center.x * si + center.y * co
                starts = [(x - cx, y - cy) for x, y in centers[:10]]
                starts.extend(
                    (px - ox, py - oy) for px, py in corners for ox, oy in offsets
                )
                starts.extend((x - cx, y - cy) for x, y in centers[10:])
                for x, y in starts:
                    if expired() or time.process_time() >= placement_deadline:
                        return None
                    polygon = Polygon([(x + dx, y + dy) for dx, dy in offsets])
                    if buffered_room.contains(polygon):
                        return (x, y), heading
            return None

        # Face lengths are always sums of INTEGER physical piece lengths.
        # Enumerate integer parameters; never round a diagonal's length from
        # coordinates. Opposite equal faces make diagonal templates close.
        lengths = sorted(faces)
        right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonal = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        room_diagonal = math.hypot(maxx - minx, maxy - miny)
        candidates = []
        generation_deadline = time.process_time() + max(
            0.01, min(0.6, (deadline - time.process_time()) * 0.12)
        )

        sampled_lengths = (
            sorted(
                {lengths[round(i * (len(lengths) - 1) / 4)] for i in range(5)},
                reverse=True,
            )
            if lengths
            else []
        )

        def parameter_tuples(dimensions):
            # Cover the whole length range before the full enumeration.
            # A timed lexicographic product otherwise spends nearly all its
            # time with its first parameter fixed at the smallest length.
            yield from chain(
                product(sampled_lengths, repeat=dimensions),
                product(reversed(lengths), repeat=dimensions),
            )

        def templates():
            generators = []
            if right >= 4:
                generators.append(
                    (
                        ("rectangle", (a, b, a, b))
                        for a, b in parameter_tuples(2)
                        if a <= b
                    )
                )
            if right >= 2 and diagonal >= 4:
                generators.append(
                    (("hexagon", (a, b, c, a, b, c)) for a, b, c in parameter_tuples(3))
                )
            if diagonal >= 8:
                generators.append(
                    (
                        ("octagon", (a, b, c, d, a, b, c, d))
                        for a, b, c, d in parameter_tuples(4)
                        if a <= c
                    )
                )
            if right >= 6:
                generators.append(
                    (
                        ("L-shape", (a + b, c, b, d, a, c + d))
                        for a, b, c, d in parameter_tuples(4)
                        if a + b in faces and c + d in faces
                    )
                )
            # Round-robin enumeration gives each family search time.
            while generators and time.process_time() < generation_deadline:
                for generator in generators[:]:
                    try:
                        yield next(generator)
                    except StopIteration:
                        generators.remove(generator)

        available_perimeter = sum(p.length * n for p, n in zip(types, stock))
        for family, sides in templates():
            if sum(sides) > available_perimeter:
                continue
            if family == "rectangle":
                area = sides[0] * sides[1]
            elif family == "hexagon":
                a, b, c = sides[:3]
                area = a * c + (a + c) * b / math.sqrt(2)
            elif family == "octagon":
                a, b, c, d = sides[:4]
                area = a * c + (a + c) * (b + d) / math.sqrt(2) + b * d
            else:
                width, c, b, d, a, _ = sides
                area = width * c + a * d
            if area > room.area + TOL * room.length:
                continue
            base = BASELINE_SCORE + self.weights.A * area + self.weights.C * sum(sides)
            if base + self.weights.G > 0:
                candidates.append((base + self.weights.G, base, family, sides))
        # Retain both high-scoring candidates and small fallback candidates.
        # This avoids spending the whole budget on large shapes that do not fit.
        grouped = {}
        generated_seen = set()
        for candidate in candidates:
            if (candidate[2], candidate[3]) in generated_seen:
                continue
            generated_seen.add((candidate[2], candidate[3]))
            grouped.setdefault(candidate[2], []).append(candidate)
        fallback, narrow, improving_groups = [], [], []
        for family, group in grouped.items():
            small = sorted(group, key=lambda candidate: sum(candidate[3]))[:12]
            fallback.extend(small)
            if family == "rectangle":
                # Narrow corridors can accommodate a long rectangle even
                # when all large-area, balanced shapes fail containment.
                narrow.extend(
                    candidate for candidate in group if min(candidate[3]) <= 7
                )
            top = sorted(group, reverse=True)[: self.MAX_CANDIDATES_PER_FAMILY]
            diverse = sorted(
                (
                    candidate
                    for candidate in group
                    if family != "L-shape"
                    and all(n in sampled_lengths for n in candidate[3])
                ),
                reverse=True,
            )
            # Preserve broad sampled sizes instead of letting the largest
            # (often unplaceable) 500 candidates eliminate smaller shapes.
            family_queue = []
            for index in range(max(len(top), len(diverse))):
                if index < len(diverse):
                    family_queue.append(diverse[index])
                if index < len(top):
                    family_queue.append(top[index])
            improving_groups.append(family_queue)
        candidates = sorted(fallback, key=lambda candidate: sum(candidate[3]))[:4]
        candidates.extend(
            sorted(narrow, key=lambda candidate: (min(candidate[3]), -candidate[0]))
        )
        candidates.extend(sorted(fallback, key=lambda candidate: sum(candidate[3])))
        for batch in zip_longest(*improving_groups):
            candidates.extend(candidate for candidate in batch if candidate is not None)
        seen = set()

        best_family = ""
        tried = 0
        best_sides = None
        refinement_tried = 0

        def evaluate(base, family, sides, hint=None):
            nonlocal best, best_score, best_family, best_sides
            turns = self._template_turns(family)
            vertices = self._integer_vertices(sides, turns)
            if vertices is None:
                return False
            shape = Polygon(vertices)
            if not shape.is_valid or shape.area <= 0:
                return False
            # A shape's diameter cannot exceed the room bounding-box diagonal.
            if any(
                math.hypot(x - xx, y - yy) > room_diagonal + TOL
                for x, y in vertices
                for xx, yy in vertices
            ):
                return False
            # Geometry is unchanged by the face assembly or choice of gates.
            assemblies = None
            if self.weights.G > 0:
                assemblies = allocate(sides, 2)
            if assemblies is None and base > best_score:
                assemblies = allocate(sides, 1)
            if assemblies is None:
                return False
            bonus = (
                sum(
                    any(types[i].is_gate and n for i, n in enumerate(face))
                    for face in assemblies
                )
                >= 2
            )
            if base + (self.weights.G if bonus else 0) <= best_score:
                return False
            found = placement(vertices, hint)
            if found is None:
                return False
            pieces, connectors = [], []
            for usage, turn in zip(assemblies, turns):
                face = [piece for piece, n in zip(types, usage) for _ in range(n)]
                pieces.extend(face)
                # The connector belongs to the START of its outgoing piece.
                connectors.append(
                    Connector(
                        ConnectorType.RIGHT
                        if abs(turn) == 90
                        else ConnectorType.DIAGONAL,
                        reflex=turn < 0,
                    )
                )
                connectors.extend(Connector(ConnectorType.STRAIGHT) for _ in face[1:])
            start, heading = found
            construction = Construction(start, heading, pieces, connectors)
            result = validate_construction(construction, self.room, self.inventory)
            if result.valid:
                score = score_construction(result, self.weights)
                if score > best_score:
                    best, best_score = construction, score
                    best_family = family
                    best_sides = sides
                    print(f"Playpen: improved {family}, score {score:.2f}", flush=True)
                    return True
            return False

        # Reserve time to improve the strongest feasible seed, rather than
        # spending the whole run on unrelated template placements.
        broad_deadline = started + budget * 0.4
        for upper_score, base, family, sides in candidates:
            if expired() or (
                best is not None and time.process_time() >= broad_deadline
            ):
                break
            if upper_score <= best_score or (family, sides) in seen:
                continue
            seen.add((family, sides))
            tried += 1
            evaluate(base, family, sides)

        local_seen = set()
        while best is not None and not expired():
            family, seed = best_family, best_sides
            parameters = self._parameters(family, seed)
            neighbors = []
            for changes in self._neighbor_changes(len(parameters)):
                proposed = tuple(n + delta for n, delta in zip(parameters, changes))
                sides = self._sides_from_parameters(family, proposed)
                if (
                    any(n not in faces for n in sides)
                    or (family, sides) in local_seen
                    or sides == seed
                ):
                    continue
                vertices = self._integer_vertices(sides, self._template_turns(family))
                if vertices is None:
                    continue
                area = Polygon(vertices).area
                base = (
                    BASELINE_SCORE + self.weights.A * area + self.weights.C * sum(sides)
                )
                if base + self.weights.G > best_score:
                    neighbors.append((base + self.weights.G, base, sides))
            neighbors.sort(reverse=True)
            incumbent = validate_construction(best, self.room, self.inventory)
            center = incumbent.polygon.centroid
            hint = ((center.x, center.y), best.start_heading)
            improved = False
            for _, base, sides in neighbors:
                if expired():
                    break
                local_seen.add((family, sides))
                refinement_tried += 1
                if evaluate(base, family, sides, hint):
                    improved = True
                    break  # Expand around the newly improved shape.
            if not improved:
                break

        timed_out = expired()
        progress = f"tried {tried} of {len(candidates)} candidates, {refinement_tried} local refinements"
        if best:
            self.note = f"Composite {best_family}; score {best_score:.2f} ({progress})."
        elif not candidates:
            self.note = "No candidate shape: connectors/pieces can't form any template."
        elif timed_out:
            self.note = f"Search budget exhausted after {time.process_time() - started:.2f}s CPU: {progress}, none fit."
        else:
            self.note = (
                f"Searched everything: {progress}; none could be built and placed."
            )
        return best

    @staticmethod
    def _template_turns(family):
        # Turns are exterior angles, indexed at the START of each face.
        return {
            "rectangle": (90, 90, 90, 90),
            "hexagon": (90, 45, 45, 90, 45, 45),
            "octagon": (45,) * 8,
            "L-shape": (90, 90, 90, -90, 90, 90),
        }[family]

    @staticmethod
    def _integer_vertices(lengths, turns):
        """Check closure exactly in Z + Z/sqrt(2), then emit float vertices.

        Integer lengths at multiples of 45 degrees give coordinates
        (axis_integer + diagonal_integer/sqrt(2)). Both coefficients must
        separately return to zero; arbitrary rounding cannot create closure.
        """
        if (
            len(lengths) != len(turns)
            or sum(turns) != 360
            or any(
                not isinstance(n, int) or not 5 <= n <= MAX_FACE_LENGTH for n in lengths
            )
        ):
            return None
        axis_x = axis_y = diagonal_x = diagonal_y = 0
        direction = 0
        vertices = []
        directions = (
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        )
        for i, length in enumerate(lengths):
            if i:
                direction = (direction + turns[i] // 45) % 8
            vertices.append(
                (axis_x + diagonal_x / math.sqrt(2), axis_y + diagonal_y / math.sqrt(2))
            )
            dx, dy = directions[direction]
            if direction % 2:
                diagonal_x += length * dx
                diagonal_y += length * dy
            else:
                axis_x += length * dx
                axis_y += length * dy
        if (axis_x, axis_y, diagonal_x, diagonal_y) != (0, 0, 0, 0):
            return None
        return vertices

    @staticmethod
    def _parameters(family, sides):
        if family == "L-shape":
            return (sides[4], sides[2], sides[1], sides[3])
        return sides[: len(sides) // 2]

    @staticmethod
    def _sides_from_parameters(family, parameters):
        if family == "L-shape":
            a, b, c, d = parameters
            return (a + b, c, b, d, a, c + d)
        return parameters + parameters

    @staticmethod
    def _neighbor_changes(count):
        # Integer moves preserve the opposite-side relationships exactly.
        for index in range(count):
            for amount in (-2, -1, 1, 2):
                change = [0] * count
                change[index] = amount
                yield tuple(change)
        for first in range(count):
            for second in range(first + 1, count):
                for a, b in product((-1, 1), repeat=2):
                    change = [0] * count
                    change[first], change[second] = a, b
                    yield tuple(change)
