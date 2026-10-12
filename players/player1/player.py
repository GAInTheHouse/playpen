"""Group 1: local search over a closed loop of straight faces.

Place the smallest rectangle that fits, in several spots spread around the
room, and grow each into the largest rectangle that fits there. From each,
run a local search for an equal share of the time budget:
score the loop's neighbors with the simulator's own validator and draw one
with probability ~ exp(gain / temperature). Every move keeps the loop closed
by construction, so a candidate only ever fails on containment or inventory.
Return the best loop seen, or None if no rectangle fits.
"""

import math
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Iterator

from shapely.geometry import Point

import src.constants as c
from players.player0 import Player
from src.enclosure import (
    Construction,
    build_geometry,
    score_construction,
    validate_construction,
)
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room

# the only setting: search time, most of the simulator's CPU limit (set
# PLAYER1_CPU_SECONDS=20 in the environment for quick development runs)
CPU_SECONDS = float(
    os.environ.get("PLAYER1_CPU_SECONDS", 0.01 * c.CPU_TIME_LIMIT_SECONDS)
)
NEARBY_RADIUS = c.MAX_FACE_LENGTH / 2  # what counts as "around" an initial solution
CONVERGENCE_STEP_NUM = 50  # stop after this many steps without a gain above MIN_GAIN
MIN_GAIN = 1.0  # score improvement that counts as progress

CONNECTOR_FOR_TURN = {  # turn in degrees -> connector
    90: Connector(ConnectorType.RIGHT),
    -90: Connector(ConnectorType.RIGHT, reflex=True),
    45: Connector(ConnectorType.DIAGONAL),
    -45: Connector(ConnectorType.DIAGONAL, reflex=True),
}
STRAIGHT = Connector(ConnectorType.STRAIGHT)

# ----------------------------------------------------------------- the loop


@dataclass(frozen=True)
class Face:
    direction: int  # degrees, multiple of 45, measured from the loop's rotation
    pieces: tuple[Piece, ...]  # in order along the face

    @property
    def length(self) -> int:
        return sum(piece.length for piece in self.pieces)

    @property
    def gate_count(self) -> int:
        return sum(1 for piece in self.pieces if piece.is_gate)


@dataclass(frozen=True)
class Loop:
    start: tuple[float, float]  # where face 0 begins
    rotation: float  # degrees; every face direction is measured from this
    faces: tuple[Face, ...]

    def with_faces(self, faces) -> "Loop":
        return replace(self, faces=tuple(faces))

    def corner_turns(self) -> list[int]:
        """The turn at the start of each face (from the face before it)."""
        return [
            turn_between(self.faces[i - 1].direction, face.direction)
            for i, face in enumerate(self.faces)
        ]

    def displacement_of(self, face: Face) -> tuple[float, float]:
        """How far `face` moves along x and y, in room coordinates."""
        angle = math.radians(self.rotation + face.direction)
        return face.length * math.cos(angle), face.length * math.sin(angle)


def turn_between(direction_from: int, direction_to: int) -> int:
    """The turn, in (-180, 180], from one direction to the other."""
    difference = (direction_to - direction_from) % 360
    return difference - 360 if difference > 180 else difference


def opposite(direction: int) -> int:
    return (direction + 180) % 360


def faces_except(faces, *indices: int) -> tuple[Face, ...]:
    return tuple(face for i, face in enumerate(faces) if i not in indices)


def join_aligned_faces(loop: Loop) -> Loop | None:
    """Merge any two adjacent faces that point the same way into one face.
    None if the merged face would be too long, or if the pair is the last
    and first face (merging them would move the start)."""
    faces = [loop.faces[0]]
    for face in loop.faces[1:]:
        if face.direction == faces[-1].direction:
            faces[-1] = Face(face.direction, faces[-1].pieces + face.pieces)
        else:
            faces.append(face)
    if faces[-1].direction == faces[0].direction:
        return None
    if any(face.length > c.MAX_FACE_LENGTH for face in faces):
        return None
    return loop.with_faces(faces)


def to_construction(loop: Loop) -> Construction:
    pieces, connectors = [], []
    for face, turn in zip(loop.faces, loop.corner_turns()):
        pieces += face.pieces
        connectors += [CONNECTOR_FOR_TURN[turn]] + [STRAIGHT] * (len(face.pieces) - 1)
    first_direction = loop.rotation + loop.faces[0].direction
    return Construction(loop.start, first_direction, pieces, connectors)


# ------------------------------------------------- choosing pieces for a face


def inventory_left_after(inventory: Inventory, faces) -> Inventory:
    """A copy of `inventory` without the pieces used by `faces`."""
    left = inventory.copy()
    for face in faces:
        for piece in face.pieces:
            left.take_piece(piece)
    return left


def pick_pieces(
    length: int, inventory: Inventory, gate_count: int = 0
) -> tuple[Piece, ...] | None:
    """Pieces in stock adding up to exactly `length`, exactly `gate_count` of
    them gates, or None. Largest pieces first, so few STRAIGHTs are spent.
    An empty tuple for length 0."""
    if length == 0:
        return () if gate_count == 0 else None
    if not c.MIN_WALL_LENGTH <= length <= c.MAX_FACE_LENGTH:
        return None
    options = [Piece(PieceType.GATE, n) for n in inventory.gates] if gate_count else []
    options += [Piece(PieceType.WALL, n) for n in sorted(inventory.walls, reverse=True)]
    for piece in options:
        left = inventory.copy()
        if piece.length > length or not left.take_piece(piece):
            continue
        rest = pick_pieces(length - piece.length, left, gate_count - int(piece.is_gate))
        if rest is not None:
            return (piece,) + rest
    return None


def build_faces(specs, inventory: Inventory) -> list[Face] | None:
    """Faces for a list of (direction, length, gate_count), each taking its
    pieces from what the previous ones left; zero lengths are skipped. None
    if any face can't be made."""
    faces = []
    for direction, length, gate_count in specs:
        pieces = pick_pieces(length, inventory, gate_count)
        if pieces is None:
            return None
        if pieces:
            faces.append(Face(direction % 360, pieces))
            inventory = inventory_left_after(inventory, faces[-1:])
    return faces


# -------------------------------------------------------------------- moves

# Each move yields loops that are closed by construction: a face's length only
# changes together with a face pointing the same or the opposite way, and the
# other moves reorder or insert faces without changing where the loop ends.


def resize_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Go through all possible ways of changing one face's length while keeping the loop closed."""
    faces = loop.faces
    for i, face in enumerate(faces):
        for j in range(i + 1, len(faces)):
            other = faces[j]
            if other.direction == face.direction:
                other_sign = -1
            elif other.direction == opposite(face.direction):
                other_sign = 1
            else:
                continue
            spare = inventory_left_after(inventory, faces_except(faces, i, j))
            for change in range(-c.MAX_FACE_LENGTH, c.MAX_FACE_LENGTH + 1):
                specs = [
                    (face.direction, face.length + change, face.gate_count),
                    (
                        other.direction,
                        other.length + other_sign * change,
                        other.gate_count,
                    ),
                ]
                pair = build_faces(specs, spare)
                if change == 0 or pair is None or len(pair) < 2:
                    continue
                new_faces = list(faces)
                new_faces[i], new_faces[j] = pair
                yield loop.with_faces(new_faces)
                dx, dy = loop.displacement_of(new_faces[i])
                back = change / new_faces[i].length
                moved_start = (loop.start[0] - dx * back, loop.start[1] - dy * back)
                yield Loop(moved_start, loop.rotation, tuple(new_faces))


def extend_with_a_turn(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Replace `width` units of a face by a rectangular detour of `depth`,
    outwards (a bump) or inwards (a notch), at the face's start, middle or
    end. The face's gate, if any, moves onto the detour.
    Example:
        before:   ────────────────────      one face, length = before + width + after

        after:    ───┐           ┌─────     five faces:
                     │           │            before     (same direction)
                     └───────────┘            depth      (turned 90°)
                                              width      (same direction, carries the gate if the face had one)
                                              depth      (turned back 90°)
                                              after      (same direction)
    """
    faces = loop.faces
    for i, face in enumerate(faces):
        spare = inventory_left_after(inventory, faces_except(faces, i))
        direction = face.direction
        for width in range(c.MIN_WALL_LENGTH, face.length + 1):
            rest = face.length - width
            for before in {0, rest // 2, rest}:
                for depth in sorted(inventory.walls):
                    # turning right (-90) leads outwards on a counter-clockwise loop
                    for side in (-1, 1):
                        specs = [
                            (direction, before, 0),
                            (direction + 90 * side, depth, 0),
                            (direction, width, face.gate_count),
                            (direction - 90 * side, depth, 0),
                            (direction, rest - before, 0),
                        ]
                        detour = build_faces(specs, spare)
                        if detour:
                            new_faces = faces[:i] + tuple(detour) + faces[i + 1 :]
                            yield loop.with_faces(new_faces)


def add_diagonal_pair(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Replace 90 connectors with 45/135 connectors on the diagonal,
    so the two cancel and the loop still
    closes. (One diagonal alone moves the loop's end by length/sqrt(2),
    which no integer wall can undo.)"""
    faces, turns = loop.faces, loop.corner_turns()
    spare = inventory_left_after(inventory, faces)
    corners = [i for i, turn in enumerate(turns) if abs(turn) == 90]
    for first in corners:
        for second in corners:
            if second <= first or turns[first] != turns[second]:
                continue
            if faces[second - 1].direction != opposite(faces[first - 1].direction):
                continue
            half_turn = turns[first] // 2
            for length in sorted(inventory.walls):
                specs = [
                    (faces[first - 1].direction + half_turn, length, 0),
                    (faces[second - 1].direction + half_turn, length, 0),
                ]
                diagonals = build_faces(specs, spare)
                if diagonals:
                    new_faces = list(faces)
                    new_faces.insert(second, diagonals[1])  # later index first
                    new_faces.insert(first, diagonals[0])
                    yield loop.with_faces(new_faces)


def add_gate_to_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Rebuild a face with a gate added (a second gate on another face earns
    the bonus) or taken away."""
    faces = loop.faces
    for i, face in enumerate(faces):
        spare = inventory_left_after(inventory, faces_except(faces, i))
        pieces = pick_pieces(face.length, spare, 0 if face.gate_count else 1)
        if pieces:
            new_faces = list(faces)
            new_faces[i] = Face(face.direction, pieces)
            if any(face.gate_count for face in new_faces):
                yield loop.with_faces(new_faces)


def shift_loop(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Slide the whole loop one unit. Score-neutral; it lets a loop pressed
    against a room wall grow again on the other side."""
    x, y = loop.start
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield Loop((x + dx, y + dy), loop.rotation, loop.faces)


MOVES = (
    resize_face,  # replace a wall / extend a face with a 180° connector
    extend_with_a_turn,  # extend a face with a turn
    shift_loop,  # translate (rotation = separate initial solutions)
    add_diagonal_pair,  # replace 90 connectors with 45/135 connectors on the diagonal
    add_gate_to_face,  # rebuild a face with a second gate for earning the second-gate bonus G
)


def neighboring_loops(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """
    Try different moves and drop the ones that do not form valid loops (e.g. a face is too long, or a turn has no connector to build).
    """
    for move in MOVES:
        for candidate in move(loop, inventory):
            candidate = join_aligned_faces(
                candidate
            )  # Connect separated faces that are on the same linear face
            if candidate is None:
                continue
            if all(
                turn in CONNECTOR_FOR_TURN for turn in candidate.corner_turns()
            ):  # Check whether all turns have a connector to be built
                yield candidate


# -------------------------------------------------------- initial solutions


def smallest_rectangle(inventory: Inventory) -> Loop | None:
    """The rectangle with the shortest perimeter the inventory can build, with
    a gate on its first side, at the origin."""
    if inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
        return None
    lengths = range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1)
    for a, b in sorted(((a, b) for a in lengths for b in lengths), key=sum):
        faces = build_faces(
            [(0, a, 1), (90, b, 0), (180, a, 0), (270, b, 0)], inventory
        )
        if faces:
            return Loop((0.0, 0.0), 0.0, tuple(faces))
    return None


def open_floor_around(loop: Loop, room: Room) -> float:
    """Floor area within NEARBY_RADIUS of the loop's centre: more in an open
    part of the room than in a narrow passage."""
    corners = build_geometry(to_construction(loop))[0]
    centre_x = sum(x for x, _ in corners) / len(corners)
    centre_y = sum(y for _, y in corners) / len(corners)
    nearby = Point(centre_x, centre_y).buffer(NEARBY_RADIUS)
    return room.polygon.intersection(nearby).area


def initial_solutions(shape: Loop, room: Room, inventory: Inventory) -> list[Loop]:
    """
    Place the initialized rectangle at different valid positions as starting points for local search, specifically:
    Rank and choose the placements that have more open area surrounded / can potentially reach more area if search from it,
    and are at least NEARBY_RADIUS apart from each other."""
    ring = list(room.polygon.exterior.coords)
    wall_directions = {
        round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90, 6) + quarter
        for (x1, y1), (x2, y2) in zip(ring, ring[1:])
        for quarter in (0, 90, 180, 270)
    }
    placed = []
    for rotation in wall_directions:
        rotated = Loop((0.0, 0.0), rotation, shape.faces)
        shape_corners = build_geometry(to_construction(rotated))[0]
        for room_x, room_y in ring[:-1]:
            for shape_x, shape_y in shape_corners:
                loop = Loop((room_x - shape_x, room_y - shape_y), rotation, shape.faces)
                if validate_construction(to_construction(loop), room, inventory).valid:
                    placed.append(loop)
    placed.sort(key=lambda loop: open_floor_around(loop, room), reverse=True)
    chosen: list[Loop] = []
    for loop in placed:
        if all(math.dist(loop.start, o.start) >= NEARBY_RADIUS for o in chosen):
            chosen.append(loop)
    return chosen


# ------------------------------------------------------------------- search


class Player1(Player):
    """Group 1: softmax local search from rectangles, closure kept by construction."""

    def evaluate(self, loop: Loop) -> float | None:
        """The simulator's score for `loop`, or None if it would be invalid."""
        result = validate_construction(to_construction(loop), self.room, self.inventory)
        return score_construction(result, self.weights) if result.valid else None

    def grow_to_largest(self, loop: Loop) -> Loop:
        """`loop` resized greedily, best-scoring resize first, until no resize
        scores higher: the largest rectangle (by score) in that spot."""
        score = self.evaluate(loop)
        while True:
            better = [
                (resized_score, resized)
                for resized in resize_face(loop, self.inventory)
                if (resized_score := self.evaluate(resized)) is not None
                and resized_score > score
            ]
            if not better:
                return loop
            score, loop = max(better, key=lambda pair: pair[0])

    def local_search(
        self, start: Loop, cpu_seconds: float, random_state: random.Random
    ) -> tuple[Loop, float]:
        """Local search from each initial candidate `start`. Each step scores every neighbor moves and
        draws one with probability ~ exp(gain / temperature). The temperature
        is the neighborhood's own mean |gain| scaled by the fraction of time
        left, so it needs no tuning: exploratory at first, greedy at the end.
        Returns the best loop seen and its score."""
        started = time.process_time()
        current, score = best, best_score = start, self.evaluate(start)
        steps_without_gain = 0
        while (elapsed := time.process_time() - started) < cpu_seconds:
            scored = [
                (loop, loop_score)
                for loop in neighboring_loops(current, self.inventory)
                if (loop_score := self.evaluate(loop)) is not None
            ]
            if not scored:
                break  # nothing around this loop is valid: a dead end
            gains = [loop_score - score for _, loop_score in scored]
            temperature = (
                (1 - elapsed / cpu_seconds) * sum(map(abs, gains)) / len(gains)
            )
            best_gain = max(gains)
            weights = [math.exp((g - best_gain) / (temperature + 1e-9)) for g in gains]
            current, score = random_state.choices(scored, weights)[0]
            steps_without_gain += 1
            if score > best_score:
                if score - best_score > MIN_GAIN:
                    steps_without_gain = 0
                best, best_score = current, score
            if steps_without_gain >= CONVERGENCE_STEP_NUM:
                break  # converged: early stop
        return best, best_score

    def build_enclosure(self) -> Construction | None:
        # Start with a valid rectangle and place it at different places in the room as possible start points for local search
        shape = smallest_rectangle(self.inventory)
        placements = (
            initial_solutions(shape, self.room, self.inventory) if shape else []
        )
        if not placements:
            self.note = "no rectangle fits: nothing built"
            return None

        # For each rectangle at different places, increase its size by as much as possible and run local search from there, returning the best result
        starts = list(dict.fromkeys(map(self.grow_to_largest, placements)))
        random_state = random.Random(0)
        deadline = time.process_time() + CPU_SECONDS
        results = []
        for i, start in enumerate(starts):
            # an equal share of the time left, so an early stop benefits the rest
            share = (deadline - time.process_time()) / (len(starts) - i)
            results.append(self.local_search(start, share, random_state))
        best, score = max(results, key=lambda result: result[1])
        self.note = f"{len(starts)} initial solutions; best {score:.0f} ({len(best.faces)} faces)"
        return to_construction(best)
