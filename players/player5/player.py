"""Bounded-width beam search with a fixed first-gate pose and one gate."""

import math
import time
from dataclasses import dataclass

from shapely.geometry import LineString

from players.player0 import Player
from src.constants import (
    ANGLE_TOL,
    MAX_FACE_LENGTH,
    MIN_ENCLOSURE_PIECES,
    MIN_WALL_LENGTH,
    TOL,
)
from src.enclosure import (
    Construction,
    compute_faces,
    score_construction,
    validate_construction,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

AREA_HEURISTIC_WEIGHT = 0.075


@dataclass
class SearchStats:
    nodes: int = 0
    closed_candidates: int = 0
    valid_polygons: int = 0
    boundary_prunes: int = 0
    crossing_prunes: int = 0
    face_prunes: int = 0
    exhausted_leaves: int = 0
    distance_prunes: int = 0
    turn_prunes: int = 0
    final_wall_prunes: int = 0


class Player5(Player):
    """Search partial wall chains from a fixed starting gate using a beam."""

    def __init__(
        self,
        room,
        inventory,
        weights,
        *,
        start=None,
        start_heading=None,
        gate_length=None,
        workers=None,
        time_limit=20.0,
        distance_pruning=True,
        turn_pruning=True,
        final_wall_pruning=True,
        beam_width=2048,
    ):
        super().__init__(room, inventory, weights)
        corners = room.get_boundary_points()
        edge_start, edge_end = max(
            zip(corners, corners[1:] + corners[:1]),
            key=lambda edge: math.dist(*edge),
        )
        self.edge_start = tuple(edge_start)
        self.start = None if start is None else tuple(start)
        self.start_heading = (
            math.degrees(
                math.atan2(edge_end[1] - edge_start[1], edge_end[0] - edge_start[0])
            )
            if start_heading is None
            else start_heading
        )
        if workers is not None and (not isinstance(workers, int) or workers < 1):
            raise ValueError("workers must be a positive integer")
        if time_limit is not None and (
            not math.isfinite(time_limit) or time_limit <= 0
        ):
            raise ValueError("time_limit must be finite and positive, or None")
        self.workers = (
            workers or 1
        )  # Accepted for compatibility; beam search is serial.
        self.time_limit = time_limit
        self.distance_pruning = distance_pruning
        self.turn_pruning = turn_pruning
        self.final_wall_pruning = final_wall_pruning
        if not isinstance(beam_width, int) or beam_width < 1:
            raise ValueError("beam_width must be a positive integer")
        self.beam_width = beam_width
        self.search_complete = False
        self.search_cpu_seconds = 0.0
        self.search_wall_seconds = 0.0
        self.gate_length = gate_length
        self.stats = SearchStats()
        self.best_score = float("-inf")

    def build_enclosure(self) -> Construction | None:
        """Run a bounded-width beam search over partial wall chains."""
        started = time.monotonic()
        cpu_started = time.process_time()
        deadline = None if self.time_limit is None else started + self.time_limit
        result = self._search_beam(deadline)
        self.search_wall_seconds = time.monotonic() - started
        self.search_cpu_seconds = time.process_time() - cpu_started
        self.search_complete = False  # Beam truncation makes this heuristic search.
        self.note = (
            f"Beam search (width {self.beam_width}): {self.stats.nodes} nodes, "
            f"{self.stats.valid_polygons} valid polygons; "
            f"{'solution score ' + format(self.best_score, 'g') if result else 'no solution found'}; "
            f"elapsed {self.search_wall_seconds:.2f}s"
        )
        return result

    def _search_beam(self, deadline):
        """Expand one complete straight face at a time, keeping a bounded beam."""
        self.stats = SearchStats()
        self.best_score = float("-inf")
        best = None
        stock0 = self.inventory.copy()
        room_area = self.room.polygon.buffer(TOL)
        choices = [
            Connector(kind, reflex)
            for kind in ConnectorType
            for reflex in (
                (False,) if kind == ConnectorType.STRAIGHT else (False, True)
            )
        ]
        turns = [joint for joint in choices if not joint.is_straight()]
        lengths = sorted(
            k
            for k, n in stock0.walls.items()
            if n and MIN_WALL_LENGTH <= k <= MAX_FACE_LENGTH
        )

        def heuristic(state):
            # Estimate area if the chain closed directly to its gate start.
            # Balance that reward against closure distance so large dead ends
            # do not crowd every closeable partial chain out of the beam.
            pts = state["points"]
            ring = [*pts, pts[0]]
            doubled_area = sum(
                x * ny - nx * y for (x, y), (nx, ny) in zip(ring, ring[1:])
            )
            estimated_area = abs(doubled_area) / 2
            distance = math.dist(pts[-1], pts[0])
            return distance - AREA_HEURISTIC_WEIGHT * estimated_area

        def face_extensions(state):
            """Use bounded knapsack choices to build a complete straight face."""
            at_gate = len(state["pieces"]) == 1
            starts = choices if at_gate else turns
            for first_joint in starts:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if state["stock"].connectors.get(first_joint.connector_type, 0) <= 0:
                    continue
                heading = state["heading"] + first_joint.turn_angle()
                initial_face = state["face"] if first_joint.is_straight() else 0
                face_limit = MAX_FACE_LENGTH - initial_face
                after_joint = state["stock"].copy()
                after_joint.connectors[first_joint.connector_type] -= 1
                max_pieces = min(
                    6,
                    sum(after_joint.walls.values()),
                    max(0, after_joint.count_connectors() - 1),
                    after_joint.connectors.get(ConnectorType.STRAIGHT, 0) + 1,
                    int(face_limit // MIN_WALL_LENGTH),
                )
                if max_pieces <= 0:
                    continue

                # dp[(piece_count, total_length)] holds up to two different
                # wall multisets. Length order within a straight face does not
                # affect its endpoint, so multisets avoid permutation repeats.
                zero = (0,) * len(lengths)
                dp = {(0, 0): [zero]}
                for index, length in enumerate(lengths):
                    updated = {
                        key: list(allocations) for key, allocations in dp.items()
                    }
                    available = max(0, state["stock"].walls.get(length, 0))
                    for (piece_count, total), allocations in dp.items():
                        for quantity in range(
                            1,
                            min(
                                available,
                                max_pieces - piece_count,
                                (face_limit - total) // length,
                            )
                            + 1,
                        ):
                            key = (piece_count + quantity, total + quantity * length)
                            options = updated.setdefault(key, [])
                            for allocation in allocations:
                                candidate_allocation = list(allocation)
                                candidate_allocation[index] += quantity
                                candidate_allocation = tuple(candidate_allocation)
                                if (
                                    candidate_allocation not in options
                                    and len(options) < 2
                                ):
                                    options.append(candidate_allocation)
                    dp = updated

                px, py = state["points"][-1]
                rad = math.radians(heading)
                legal_totals = {}
                for piece_count, total in dp:
                    if piece_count == 0:
                        continue
                    end = (px + total * math.cos(rad), py + total * math.sin(rad))
                    if not room_area.covers(LineString([state["points"][-1], end])):
                        continue
                    check_end = (
                        state["start"] if math.dist(end, state["start"]) <= TOL else end
                    )
                    if not LineString([*state["points"], check_end]).is_simple:
                        continue
                    legal_totals[(piece_count, total)] = end
                if not legal_totals:
                    continue

                candidates = []
                for (piece_count, total), end in legal_totals.items():
                    for allocation in dp[(piece_count, total)]:
                        next_stock = after_joint.copy()
                        pieces_to_add = []
                        for length, quantity in zip(lengths, allocation):
                            next_stock.walls[length] -= quantity
                            pieces_to_add.extend([length] * quantity)
                        next_stock.connectors[ConnectorType.STRAIGHT] -= piece_count - 1
                        points = list(state["points"])
                        for length in pieces_to_add:
                            prior_x, prior_y = points[-1]
                            points.append(
                                (
                                    prior_x + length * math.cos(rad),
                                    prior_y + length * math.sin(rad),
                                )
                            )
                        candidate = {
                            "start": state["start"],
                            "points": points,
                            "heading": heading,
                            "face": initial_face + total,
                            "pieces": [
                                *state["pieces"],
                                *(
                                    Piece(PieceType.WALL, length)
                                    for length in pieces_to_add
                                ),
                            ],
                            "joints": [
                                *state["joints"],
                                first_joint,
                                *(
                                    Connector(ConnectorType.STRAIGHT)
                                    for _ in range(piece_count - 1)
                                ),
                            ],
                            "stock": next_stock,
                        }
                        candidates.append((candidate, total, piece_count, allocation))

                selected = []

                def keep_best(options):
                    for option in sorted(options, key=lambda item: heuristic(item[0])):
                        signature = (option[1], option[2], option[3])
                        if all(
                            (item[1], item[2], item[3]) != signature
                            for item in selected
                        ):
                            selected.append(option)
                            return

                longest = max(option[1] for option in candidates)
                keep_best([option for option in candidates if option[1] == longest])
                keep_best(candidates)
                shorter = [option for option in candidates if option[1] < longest]
                if shorter:
                    target = longest * 0.7
                    closest_total = min(abs(option[1] - target) for option in shorter)
                    keep_best(
                        [
                            option
                            for option in shorter
                            if abs(option[1] - target) == closest_total
                        ]
                    )
                yield from (option[0] for option in selected)

        frontier = []
        theta = math.radians(self.start_heading)
        for gate_length, count in sorted(stock0.gates.items()):
            if (
                count <= 0
                or gate_length < MIN_WALL_LENGTH
                or gate_length > MAX_FACE_LENGTH
            ):
                continue
            if self.gate_length is not None and gate_length != self.gate_length:
                continue
            start = self.start or self.edge_start
            end = (
                start[0] + gate_length * math.cos(theta),
                start[1] + gate_length * math.sin(theta),
            )
            if not room_area.covers(LineString([start, end])):
                self.stats.boundary_prunes += 1
                continue
            stock = stock0.copy()
            stock.gates[gate_length] -= 1
            frontier.append(
                {
                    "start": start,
                    "points": [start, end],
                    "heading": self.start_heading,
                    "face": gate_length,
                    "pieces": [Piece(PieceType.GATE, gate_length)],
                    "joints": [],
                    "stock": stock,
                }
            )

        while frontier:
            if deadline is not None and time.monotonic() >= deadline:
                break
            expanded = []
            for state in frontier:
                if deadline is not None and time.monotonic() >= deadline:
                    return best
                self.stats.nodes += 1
                stock = state["stock"]
                pts = state["points"]
                distance = math.dist(pts[-1], state["start"])
                if distance <= TOL and len(state["pieces"]) >= MIN_ENCLOSURE_PIECES:
                    for closing in choices:
                        if stock.connectors.get(closing.connector_type, 0) <= 0:
                            continue
                        gap = (
                            state["heading"] + closing.turn_angle() - self.start_heading
                        ) % 360
                        if min(gap, 360 - gap) > ANGLE_TOL:
                            continue
                        candidate = Construction(
                            state["start"],
                            self.start_heading,
                            list(state["pieces"]),
                            [closing, *state["joints"]],
                        )
                        candidate = self._add_second_gate(candidate)
                        self.stats.closed_candidates += 1
                        valid = validate_construction(
                            candidate, self.room, self.inventory
                        )
                        if valid.valid:
                            self.stats.valid_polygons += 1
                            score = score_construction(valid, self.weights)
                            if score > self.best_score:
                                best, self.best_score = candidate, score
                    continue
                if sum(stock.walls.values()) == 0 or stock.count_connectors() < 2:
                    self.stats.exhausted_leaves += 1
                    continue
                expanded.extend(face_extensions(state))
            expanded.sort(key=heuristic)
            frontier = expanded[: self.beam_width]
        return best

    def _add_second_gate(self, construction):
        """Replace a wall with a same-length gate on another face when possible."""
        faces = compute_faces(construction)
        used_gates = {}
        for piece in construction.pieces:
            if piece.is_gate:
                used_gates[piece.length] = used_gates.get(piece.length, 0) + 1

        for face in faces:
            if face.has_gate:
                continue
            for piece_index in face.piece_indices:
                piece = construction.pieces[piece_index]
                if piece.is_gate:
                    continue
                already_used = used_gates.get(piece.length, 0)
                if self.inventory.gates.get(piece.length, 0) <= already_used:
                    continue
                pieces = list(construction.pieces)
                pieces[piece_index] = Piece(PieceType.GATE, piece.length)
                candidate = Construction(
                    construction.start,
                    construction.start_heading,
                    pieces,
                    list(construction.connectors),
                )
                result = validate_construction(candidate, self.room, self.inventory)
                if result.valid and result.second_gate_bonus:
                    return candidate
        return construction
