"""Naive exhaustive search with a fixed first-gate pose and exactly one gate."""

import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from functools import lru_cache

try:
    import resource
except ImportError:  # Windows: search CPU is an estimate without child accounting.
    resource = None

from shapely.geometry import LineString

from players.player0 import Player
from src.constants import (
    ANGLE_TOL,
    MAX_FACE_LENGTH,
    MIN_ENCLOSURE_PIECES,
    MIN_WALL_LENGTH,
    TOL,
)
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


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


@lru_cache(maxsize=8192)
def _turn_completion_possible(turns, right, diagonal, straight, minimum, maximum):
    """Exact connector-subset feasibility in units of 45 degrees.

    A simple polygon has total turn +8 or -8. For r right connectors the
    possible sums are -2r, -2r+4, ..., +2r. Diagonals contribute +/-1;
    extra pairs can cancel, and straight connectors can fill unused joints.
    """
    if minimum > maximum:
        return False
    max_turn = 2 * min(right, maximum) + min(diagonal, max(0, maximum - right))
    for target in (8 - turns, -8 - turns):
        if abs(target) > max_turn or (diagonal == 0 and target % 2):
            continue
        for used_right in range(min(right, maximum) + 1):
            # Closest attainable right-connector sum to the required turn.
            positive = min(used_right, max(0, (target + 2 * used_right + 2) // 4))
            residual = abs(target - (4 * positive - 2 * used_right))
            used_diagonal = max(residual, minimum - used_right - straight, 0)
            used_diagonal += (target - used_diagonal) % 2
            if used_diagonal <= diagonal and used_right + used_diagonal <= maximum:
                return True
    return False


class _DeadlineReached(Exception):
    pass


def _children_cpu_seconds():
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _run_partition(player, partition, partitions, deadline):
    return player._search_partition(partition, partitions, deadline)


class Player5(Player):
    """Enumerate every wall/connector chain rooted at a fixed starting gate.

    Defaults to centering the gate along the longest room edge.
    Optional keyword arguments fix another pose or a particular gate length.
    Workers explore disjoint first-wall branches. The default five-second
    elapsed-time deadline is shared across workers, including process startup.
    Return the best polygon found so far. Set workers=1, time_limit=None for
    the unbounded exhaustive DFS.
    """

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
        time_limit=5.0,
        distance_pruning=True,
        turn_pruning=True,
        final_wall_pruning=True,
    ):
        super().__init__(room, inventory, weights)
        corners = room.get_boundary_points()
        edge_start, edge_end = max(
            zip(corners, corners[1:] + corners[:1]),
            key=lambda edge: math.dist(*edge),
        )
        self.edge_midpoint = (
            (edge_start[0] + edge_end[0]) / 2,
            (edge_start[1] + edge_end[1]) / 2,
        )
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
        self.workers = min(4, os.cpu_count() or 1) if workers is None else workers
        self.time_limit = time_limit
        self.distance_pruning = distance_pruning
        self.turn_pruning = turn_pruning
        self.final_wall_pruning = final_wall_pruning
        self.search_complete = False
        self.search_cpu_seconds = 0.0
        self.search_wall_seconds = 0.0
        self.gate_length = gate_length
        self.stats = SearchStats()
        self.best_score = float("-inf")

    def build_enclosure(self) -> Construction | None:
        wall_started = time.monotonic()
        cpu_started = time.process_time()
        children_started = _children_cpu_seconds()
        # The same absolute deadline goes to every task. It is never divided
        # by worker count or restarted when a worker begins execution.
        deadline = None if self.time_limit is None else wall_started + self.time_limit
        if self.workers == 1:
            results = [self._search_partition(0, 1, deadline)]
            worker_cpu = 0.0
        else:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                futures = [
                    pool.submit(_run_partition, self, i, self.workers, deadline)
                    for i in range(self.workers)
                ]
                results = [future.result() for future in futures]
            children_finished = _children_cpu_seconds()
            worker_cpu = (
                children_finished - children_started
                if children_started is not None
                else sum(result[4] for result in results)
            )
        self.stats = SearchStats()
        self.best_score = float("-inf")
        self.search_complete = all(result[3] for result in results)
        best = None
        for candidate, stats, score, _, _ in results:
            for field in fields(SearchStats):
                setattr(
                    self.stats,
                    field.name,
                    getattr(self.stats, field.name) + getattr(stats, field.name),
                )
            if candidate is not None and score > self.best_score:
                best, self.best_score = candidate, score
        self.search_cpu_seconds = time.process_time() - cpu_started + worker_cpu
        self.search_wall_seconds = time.monotonic() - wall_started
        outcome = (
            f"best score {self.best_score:g}"
            if best is not None
            else "no solution found"
        )
        state = "complete" if self.search_complete else "deadline reached"
        self.note = (
            f"DFS ({self.workers} workers, {state}): {self.stats.nodes} nodes, "
            f"{self.stats.valid_polygons} valid polygons; {outcome}; "
            f"elapsed {self.search_wall_seconds:.2f}s, total CPU {self.search_cpu_seconds:.2f}s"
        )
        return best

    def _search_partition(self, partition, partitions, deadline):
        started = time.process_time()

        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise _DeadlineReached

        self.stats = SearchStats()
        self.best_score = float("-inf")
        best = None
        stock = self.inventory.copy()
        room_area = self.room.polygon.buffer(TOL)
        return_lengths = sorted(
            (
                length
                for length in stock.walls
                if MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
            ),
            reverse=True,
        )

        def maximum_return_distance():
            # Each added wall needs a connector, and the root needs one more.
            # Ignore turns/obstacles to overestimate reach, never underestimate it.
            slots = max(0, stock.count_connectors() - 1)
            reach = 0
            for length in return_lengths:
                count = min(max(0, stock.walls[length]), slots)
                reach += length * count
                slots -= count
                if slots == 0:
                    break
            return reach

        choices = [
            Connector(kind, reflex)
            for kind in ConnectorType
            for reflex in (
                (False,) if kind == ConnectorType.STRAIGHT else (False, True)
            )
        ]
        pieces = []
        # Internal joints only. The root's closing joint is chosen at closure.
        joints = []
        points = []
        start = None

        def endpoint(heading, length):
            theta = math.radians(heading)
            x, y = points[-1]
            return x + length * math.cos(theta), y + length * math.sin(theta)

        def search(heading, face_length):
            nonlocal best
            check_deadline()
            self.stats.nodes += 1
            distance_to_start = math.dist(points[-1], start)
            closed = distance_to_start <= TOL
            if closed:
                if len(pieces) >= MIN_ENCLOSURE_PIECES:
                    for closing in choices:
                        if stock.connectors.get(closing.connector_type, 0) <= 0:
                            continue
                        gap = (
                            heading + closing.turn_angle() - self.start_heading
                        ) % 360
                        if min(gap, 360 - gap) > ANGLE_TOL:
                            continue
                        candidate = Construction(
                            start,
                            self.start_heading,
                            list(pieces),
                            [closing, *joints],
                        )
                        self.stats.closed_candidates += 1
                        result = validate_construction(
                            candidate, self.room, self.inventory
                        )
                        if result.valid:
                            self.stats.valid_polygons += 1
                            score = score_construction(result, self.weights)
                            if score > self.best_score:
                                best, self.best_score = candidate, score
                # Extending a closed loop would revisit a vertex: never simple.
                return

            # One extra connector is always required to close at the root.
            if sum(stock.walls.values()) == 0 or stock.count_connectors() < 2:
                self.stats.exhausted_leaves += 1
                return

            # By the triangle inequality, no chain of the remaining walls
            # can reach the start if even its maximum total length is too short.
            if (
                self.distance_pruning
                and distance_to_start > maximum_return_distance() + TOL
            ):
                self.stats.distance_prunes += 1
                return

            if self.turn_pruning or self.final_wall_pruning:
                remaining_walls = sum(stock.walls[length] for length in return_lengths)
                remaining_connectors = stock.count_connectors()
                maximum_walls = min(remaining_walls, remaining_connectors - 1)
            if self.turn_pruning and not _turn_completion_possible(
                round((heading - self.start_heading) / 45),
                stock.connectors.get(ConnectorType.RIGHT, 0),
                stock.connectors.get(ConnectorType.DIAGONAL, 0),
                stock.connectors.get(ConnectorType.STRAIGHT, 0),
                max(2, MIN_ENCLOSURE_PIECES - len(pieces) + 1),
                maximum_walls + 1,
            ):
                self.stats.turn_prunes += 1
                return

            last_wall = self.final_wall_pruning and maximum_walls == 1
            for joint_index, joint in enumerate(choices):
                if not stock.take_connector(joint.connector_type):
                    continue
                next_heading = heading + joint.turn_angle()
                for wall_index, (length, count) in enumerate(stock.walls.items()):
                    check_deadline()
                    if (
                        len(pieces) == 1
                        and (joint_index * len(stock.walls) + wall_index) % partitions
                        != partition
                    ):
                        continue
                    if count <= 0 or length < MIN_WALL_LENGTH:
                        continue
                    next_face = face_length + length if joint.is_straight() else length
                    if next_face > MAX_FACE_LENGTH + TOL:
                        self.stats.face_prunes += 1
                        continue
                    end = endpoint(next_heading, length)
                    if last_wall:
                        can_close = False
                        if (
                            len(pieces) + 1 >= MIN_ENCLOSURE_PIECES
                            and math.dist(end, start) <= TOL
                        ):
                            for closing in choices:
                                if stock.connectors.get(closing.connector_type, 0) <= 0:
                                    continue
                                gap = (
                                    next_heading
                                    + closing.turn_angle()
                                    - self.start_heading
                                ) % 360
                                if min(gap, 360 - gap) <= ANGLE_TOL:
                                    can_close = True
                                    break
                        if not can_close:
                            self.stats.final_wall_prunes += 1
                            continue
                    if not room_area.covers(LineString([points[-1], end])):
                        self.stats.boundary_prunes += 1
                        continue
                    # Snap only the simplicity check to handle floating point
                    # closure. Keep actual coordinates for the official judge.
                    check_end = start if math.dist(end, start) <= TOL else end
                    if not LineString([*points, check_end]).is_simple:
                        self.stats.crossing_prunes += 1
                        continue
                    stock.walls[length] -= 1
                    pieces.append(Piece(PieceType.WALL, length))
                    joints.append(joint)
                    points.append(end)
                    search(next_heading, next_face)
                    points.pop()
                    joints.pop()
                    pieces.pop()
                    stock.walls[length] += 1
                stock.connectors[joint.connector_type] += 1

        complete = True
        try:
            check_deadline()
            for length, count in sorted(stock.gates.items()):
                check_deadline()
                if (
                    count <= 0
                    or length < MIN_WALL_LENGTH
                    or (self.gate_length is not None and length != self.gate_length)
                ):
                    continue
                if length > MAX_FACE_LENGTH + TOL:
                    self.stats.face_prunes += 1
                    continue
                theta = math.radians(self.start_heading)
                start = (
                    self.start
                    if self.start is not None
                    else (
                        self.edge_midpoint[0] - length * math.cos(theta) / 2,
                        self.edge_midpoint[1] - length * math.sin(theta) / 2,
                    )
                )
                points[:] = [start]
                end = endpoint(self.start_heading, length)
                if not room_area.covers(LineString([start, end])):
                    self.stats.boundary_prunes += 1
                    continue
                stock.gates[length] -= 1
                pieces.append(Piece(PieceType.GATE, length))
                points.append(end)
                search(self.start_heading, length)
                points.pop()
                pieces.pop()
                stock.gates[length] += 1

        except _DeadlineReached:
            complete = False
        return (
            best,
            self.stats,
            self.best_score,
            complete,
            time.process_time() - started,
        )
