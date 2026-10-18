"""Assign physical stock jointly to a polygon's complete straight faces.

Geometry search can work with one edge per face. This helper does the separate
combinatorial job of assembling those faces from walls, gates and straight
connectors without spending the same inventory twice.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.constants import ANGLE_TOL, MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


@dataclass(frozen=True, slots=True)
class _Option:
    usage: tuple[tuple[int, int], ...]
    pieces: tuple[Piece, ...]
    gates: int

    @property
    def straights(self):
        return len(self.pieces) - 1


class FaceInventory:
    """Bounded face packing with geometry-independent assignment memoization."""

    ASSIGNMENT_NODES = 5000

    def __init__(self, inventory, deadline: float):
        self.deadline = deadline
        self.connectors = dict(inventory.connectors)
        self.straights = max(0, self.connectors.get(ConnectorType.STRAIGHT, 0))
        self.types = tuple(
            (Piece(kind, length), int(count))
            for kind, pool in (
                (PieceType.WALL, inventory.walls),
                (PieceType.GATE, inventory.gates),
            )
            for length, count in sorted(pool.items(), reverse=True)
            if isinstance(length, int)
            and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
            and count > 0
        )
        self.stock = tuple(count for _, count in self.types)
        self.max_perimeter = sum(piece.length * count for piece, count in self.types)
        self.gate_stock = sum(count for piece, count in self.types if piece.is_gate)
        self.options = {}
        self._complete = True
        self._cache = {}
        self._prefix_cache = {}
        usage, pieces = [], []
        visits = 0

        def enumerate_options(first, length, gates):
            nonlocal visits
            visits += 1
            if visits % 64 == 1 and time.process_time() >= deadline:
                self._complete = False
                return
            if pieces:
                self.options.setdefault(length, []).append(
                    _Option(tuple(usage), tuple(pieces), gates)
                )
            if len(pieces) > self.straights:
                return
            for index in range(first, len(self.types)):
                piece, available = self.types[index]
                room = (MAX_FACE_LENGTH - length) // piece.length
                count_limit = min(available, room, self.straights + 1 - len(pieces))
                for count in range(1, count_limit + 1):
                    usage.append((index, count))
                    pieces.extend([piece] * count)
                    enumerate_options(
                        index + 1,
                        length + piece.length * count,
                        gates + (count if piece.is_gate else 0),
                    )
                    del pieces[-count:]
                    usage.pop()
                    if not self._complete:
                        return

        enumerate_options(0, 0, 0)
        self.lengths = tuple(sorted(self.options))
        for options in self.options.values():
            options.sort(key=lambda option: (len(option.pieces), option.gates))

    def _assign(self, lengths, minimum_gated, maximum_gated, deadline, node_limit):
        """Return (assignment, exhaustive); an exhausted budget is not a proof."""
        if time.process_time() >= deadline:
            return None, False
        if minimum_gated > self.gate_stock or sum(lengths) > self.max_perimeter:
            return None, True
        if any(length not in self.options for length in lengths):
            return None, self._complete
        # Few choices first keeps scarce sizes from being consumed greedily.
        order = sorted(
            range(len(lengths)),
            key=lambda index: (len(self.options[lengths[index]]), -lengths[index]),
        )
        ordered = [self.options[lengths[index]] for index in order]
        min_straights = [min(option.straights for option in opts) for opts in ordered]
        suffix_straights = [0] * (len(lengths) + 1)
        for depth in range(len(lengths) - 1, -1, -1):
            suffix_straights[depth] = suffix_straights[depth + 1] + min_straights[depth]
        remaining = list(self.stock)
        chosen = [None] * len(lengths)
        failures = set()
        nodes = 0
        exhaustive = True

        def search(depth, straights, gated):
            nonlocal nodes, exhaustive
            nodes += 1
            if nodes > node_limit or (
                nodes % 32 == 1 and time.process_time() >= deadline
            ):
                exhaustive = False
                return False
            if gated + len(lengths) - depth < minimum_gated:
                return False
            if gated > maximum_gated or suffix_straights[depth] > straights:
                return False
            if depth == len(lengths):
                return gated >= minimum_gated
            key = (depth, straights, min(gated, minimum_gated), tuple(remaining))
            if key in failures:
                return False
            # Choose a gate early when still needed, but do not commit greedily:
            # the recursion revisits wall/gate choices when later faces need them.
            want_gate = gated < minimum_gated
            options = ordered[depth]
            for gated_option in (want_gate, not want_gate):
                for option in options:
                    if bool(option.gates) != gated_option:
                        continue
                    if option.straights > straights:
                        continue
                    if any(remaining[index] < count for index, count in option.usage):
                        continue
                    for index, count in option.usage:
                        remaining[index] -= count
                    chosen[order[depth]] = option
                    success = search(
                        depth + 1,
                        straights - option.straights,
                        gated + bool(option.gates),
                    )
                    for index, count in option.usage:
                        remaining[index] += count
                    if success:
                        return True
                    if not exhaustive:
                        return False
            failures.add(key)
            return False

        if search(0, self.straights, 0):
            return tuple(chosen), exhaustive
        return None, exhaustive and self._complete

    def feasible_prefix(self, lengths, node_limit=100):
        """Conservative stock-only pruning: True also means not yet disproved."""
        lengths = tuple(sorted(lengths))
        if sum(lengths) > self.max_perimeter:
            return False
        if lengths in self._prefix_cache:
            return self._prefix_cache[lengths]
        assignment, exhaustive = self._assign(
            lengths, 0, len(lengths), self.deadline, node_limit
        )
        answer = assignment is not None or not exhaustive
        if assignment is not None or exhaustive:
            self._prefix_cache[lengths] = answer
        return answer

    def build(self, vertices, prefer_bonus=True, deadline=None):
        """Pack a simple integer-faced polygon; room containment is caller-owned."""
        deadline = self.deadline if deadline is None else min(self.deadline, deadline)
        if time.process_time() >= deadline or len(vertices) < 3 or not self.gate_stock:
            return None
        vertices = tuple((float(x), float(y)) for x, y in vertices)
        if any(not math.isfinite(value) for point in vertices for value in point):
            return None
        lengths, headings = [], []
        for index, point in enumerate(vertices):
            following = vertices[(index + 1) % len(vertices)]
            length = math.dist(point, following)
            rounded = round(length)
            if (
                abs(length - rounded) > TOL / max(1, len(vertices))
                or not MIN_WALL_LENGTH <= rounded <= MAX_FACE_LENGTH
            ):
                return None
            lengths.append(rounded)
            headings.append(
                math.degrees(
                    math.atan2(following[1] - point[1], following[0] - point[0])
                )
            )
        corners = []
        for index, heading in enumerate(headings):
            turn = (heading - headings[index - 1] + 180) % 360 - 180
            expected = min((-90, -45, 45, 90), key=lambda angle: abs(angle - turn))
            if abs(expected - turn) > ANGLE_TOL / max(1, len(vertices)):
                return None
            kind = (
                ConnectorType.RIGHT if abs(expected) == 90 else ConnectorType.DIAGONAL
            )
            corners.append(Connector(kind, reflex=expected < 0))
        needed = Counter(corner.connector_type for corner in corners)
        if any(count > self.connectors.get(kind, 0) for kind, count in needed.items()):
            return None
        polygon = Polygon(vertices)
        if not polygon.is_valid or polygon.area <= TOL:
            return None
        ordering = sorted(range(len(lengths)), key=lambda index: lengths[index])
        sorted_lengths = tuple(lengths[index] for index in ordering)
        cache_key = (sorted_lengths, bool(prefer_bonus))
        if cache_key in self._cache:
            assignment = self._cache[cache_key]
        else:
            if prefer_bonus:
                targets = ((2, len(lengths)), (1, len(lengths)))
            else:
                targets = ((1, 1), (1, len(lengths)))
            assignment = None
            exhaustive = True
            for minimum, maximum in targets:
                assignment, complete = self._assign(
                    sorted_lengths, minimum, maximum, deadline, self.ASSIGNMENT_NODES
                )
                exhaustive = exhaustive and complete
                if assignment is not None:
                    break
            if assignment is not None or exhaustive:
                self._cache[cache_key] = assignment
        if assignment is None:
            return None
        by_face = [None] * len(vertices)
        for index, option in zip(ordering, assignment):
            by_face[index] = option
        pieces, connectors = [], []
        straight = Connector(ConnectorType.STRAIGHT)
        for corner, option in zip(corners, by_face):
            pieces.extend(option.pieces)
            connectors.extend([corner] + [straight] * option.straights)
        return Construction(vertices[0], headings[0], pieces, connectors)
