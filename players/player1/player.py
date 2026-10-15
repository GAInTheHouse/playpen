"""Group 1: local search over a closed loop of straight faces.

Place the smallest rectangle around the room, most open spots first. Grow
each greedily, then run a softmax local search until it converges or time
runs out. Return the best loop, with a second gate added if one fits.
"""

import functools
import itertools
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import NamedTuple

import shapely
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

# the only setting: total CPU time per room, kept under 10 s (a search step can
# run a little past it); set PLAYER1_CPU_SECONDS in the environment to override
CPU_SECONDS = float(os.environ.get("PLAYER1_CPU_SECONDS", "9"))
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

    def moved_to(self, start: tuple[float, float]) -> "Loop":
        return replace(self, start=start)

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


def corners(loop: Loop) -> list[tuple[float, float]]:
    return build_geometry(to_construction(loop))[0]


def centre(points) -> tuple[float, float]:
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


class PreparedRoom(Room):
    """The room with the simulator's tolerance buffer built once. The
    simulator's contains_enclosure rebuilds it on every call, which is slow
    for rooms with many corners; the answer is the same."""

    def __init__(self, room: Room):
        super().__init__(room.polygon)
        self.floor = room.polygon.buffer(c.TOL)
        shapely.prepare(self.floor)

    def contains_enclosure(self, enclosure_polygon, tol: float = c.TOL) -> bool:
        return self.floor.contains(enclosure_polygon)


# ------------------------------------------------- choosing pieces for a face


class FaceSpec(NamedTuple):
    direction: int
    length: int
    gate_count: int = 0


def spare(inventory: Inventory, faces, *skip: int) -> Inventory:
    """A copy of `inventory` without the pieces used by `faces`, except the
    faces at the indices in `skip`."""
    left = inventory.copy()
    for i, face in enumerate(faces):
        if i not in skip:
            for piece in face.pieces:
                left.take_piece(piece)
    return left


def pick_pieces(
    length: int, inventory: Inventory, gate_count: int = 0
) -> tuple[Piece, ...] | None:
    """Pieces in stock adding up to exactly `length`, exactly `gate_count` of
    them gates, or None. Largest pieces first, so few STRAIGHTs are spent.
    An empty tuple for length 0. Cached: the same question comes up for many
    candidates in every step."""
    walls, gates = tuple(inventory.walls.items()), tuple(inventory.gates.items())
    return _pick_pieces(length, gate_count, walls, gates)


def _take_one(stock, length: int):
    return tuple((n, count - (n == length)) for n, count in stock)


@functools.cache
def _pick_pieces(length: int, gate_count: int, walls, gates):
    if length == 0:
        return () if gate_count == 0 else None
    if not c.MIN_WALL_LENGTH <= length <= c.MAX_FACE_LENGTH:
        return None
    options = [(n, True) for n, count in gates if count > 0] if gate_count else []
    options += [(n, False) for n, count in sorted(walls, reverse=True) if count > 0]
    for n, is_gate in options:
        if n > length:
            continue
        if is_gate:
            rest = _pick_pieces(length - n, gate_count - 1, walls, _take_one(gates, n))
        else:
            rest = _pick_pieces(length - n, gate_count, _take_one(walls, n), gates)
        if rest is not None:
            return (Piece(PieceType.GATE if is_gate else PieceType.WALL, n),) + rest
    return None


def build_faces(specs: list[FaceSpec], inventory: Inventory) -> list[Face] | None:
    """Faces for `specs`, each taking its pieces from what the previous ones
    left; zero lengths are skipped. None if any face can't be made."""
    faces = []
    for direction, length, gate_count in specs:
        pieces = pick_pieces(length, inventory, gate_count)
        if pieces is None:
            return None
        if pieces:
            faces.append(Face(direction % 360, pieces))
            inventory = spare(inventory, faces[-1:])
    return faces


# -------------------------------------------------------------------- moves

# Each move yields loops that are closed by construction: a face's length only
# changes together with a face pointing the same or the opposite way, and the
# other moves reorder or insert faces without changing where the loop ends.


def parallel_faces(faces) -> Iterator[tuple[int, int, int]]:
    """(i, j, sign) for each pair of faces pointing the same way (sign -1)
    or opposite ways (sign 1)."""
    for i, j in itertools.combinations(range(len(faces)), 2):
        if faces[j].direction == faces[i].direction:
            yield i, j, -1
        elif faces[j].direction == opposite(faces[i].direction):
            yield i, j, 1


def resize_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Change one face's length together with a parallel face, at the
    face's end or at its start."""
    faces = loop.faces
    for i, j, sign in parallel_faces(faces):
        face, other = faces[i], faces[j]
        left = spare(inventory, faces, i, j)
        for length in range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1):
            change = length - face.length
            other_length = other.length + sign * change
            if change == 0 or other_length < c.MIN_WALL_LENGTH:
                continue
            specs = [
                FaceSpec(face.direction, length, face.gate_count),
                FaceSpec(other.direction, other_length, other.gate_count),
            ]
            pair = build_faces(specs, left)
            if pair is None:
                continue
            new_faces = list(faces)
            new_faces[i], new_faces[j] = pair
            resized = loop.with_faces(new_faces)
            yield resized
            dx, dy = loop.displacement_of(new_faces[i])
            back = change / new_faces[i].length
            yield resized.moved_to(
                (loop.start[0] - dx * back, loop.start[1] - dy * back)
            )


def detour_specs(
    face: Face, width: int, before: int, depth: int, side: int
) -> list[FaceSpec]:
    after = face.length - width - before
    direction = face.direction
    return [
        FaceSpec(direction, before),
        FaceSpec(direction + 90 * side, depth),
        FaceSpec(direction, width, face.gate_count),
        FaceSpec(direction - 90 * side, depth),
        FaceSpec(direction, after),
    ]


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
    depths = sorted(inventory.walls)
    for i, face in enumerate(faces):
        left = spare(inventory, faces, i)
        for width in range(c.MIN_WALL_LENGTH, face.length + 1):
            rest = face.length - width
            befores = {0, rest // 2, rest}
            for before, depth, side in itertools.product(befores, depths, (-1, 1)):
                detour = build_faces(
                    detour_specs(face, width, before, depth, side), left
                )
                if detour:
                    yield loop.with_faces(faces[:i] + tuple(detour) + faces[i + 1 :])


def add_diagonal_pair(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Cut two matching 90° corners with opposite diagonals, which cancel."""
    faces, turns = loop.faces, loop.corner_turns()
    left = spare(inventory, faces)
    right_angles = [i for i, turn in enumerate(turns) if abs(turn) == 90]
    for first, second in itertools.combinations(right_angles, 2):
        into_first, into_second = faces[first - 1], faces[second - 1]
        if turns[first] != turns[second]:
            continue
        if into_second.direction != opposite(into_first.direction):
            continue
        half_turn = turns[first] // 2
        for length in sorted(inventory.walls):
            specs = [
                FaceSpec(into_first.direction + half_turn, length),
                FaceSpec(into_second.direction + half_turn, length),
            ]
            diagonals = build_faces(specs, left)
            if diagonals:
                new_faces = list(faces)
                new_faces.insert(second, diagonals[1])  # later index first
                new_faces.insert(first, diagonals[0])
                yield loop.with_faces(new_faces)


def add_gate_to_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Rebuild a face with a gate added or taken away."""
    faces = loop.faces
    for i, face in enumerate(faces):
        left = spare(inventory, faces, i)
        pieces = pick_pieces(face.length, left, 0 if face.gate_count else 1)
        if pieces:
            new_faces = list(faces)
            new_faces[i] = Face(face.direction, pieces)
            if any(new_face.gate_count for new_face in new_faces):
                yield loop.with_faces(new_faces)


def shift_loop(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Slide the whole loop one unit. Score-neutral; it lets a loop pressed
    against a room wall grow again on the other side."""
    x, y = loop.start
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield loop.moved_to((x + dx, y + dy))


MOVES = (
    resize_face,
    extend_with_a_turn,
    shift_loop,
    add_diagonal_pair,
    add_gate_to_face,
)


def neighboring_loops(loop: Loop, inventory: Inventory, moves=MOVES) -> Iterator[Loop]:
    """Valid loops that `moves` make from `loop`."""
    for move in moves:
        for candidate in move(loop, inventory):
            candidate = join_aligned_faces(candidate)
            if candidate is None:
                continue
            if all(turn in CONNECTOR_FOR_TURN for turn in candidate.corner_turns()):
                yield candidate


# -------------------------------------------------------- initial solutions


def smallest_rectangle(inventory: Inventory) -> Loop | None:
    """The rectangle with the shortest perimeter the inventory can build, with
    a gate on its first side, at the origin."""
    if inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
        return None
    lengths = range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1)
    for a, b in sorted(((a, b) for a in lengths for b in lengths), key=sum):
        specs = [FaceSpec(0, a, 1), FaceSpec(90, b), FaceSpec(180, a), FaceSpec(270, b)]
        faces = build_faces(specs, inventory)
        if faces:
            return Loop((0.0, 0.0), 0.0, tuple(faces))
    return None


def open_floor_around(loop: Loop, room: Room) -> float:
    """Floor area within NEARBY_RADIUS of the loop's centre: more in an open
    part of the room than in a narrow passage."""
    nearby = Point(*centre(corners(loop))).buffer(NEARBY_RADIUS)
    return room.polygon.intersection(nearby).area


def room_rotations(room: Room) -> set[float]:
    return {
        round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90, 6) + quarter
        for (x1, y1), (x2, y2) in itertools.pairwise(room.polygon.exterior.coords)
        for quarter in (0, 90, 180, 270)
    }


def all_corners_inside(floor, starts, shape_corners) -> list[bool]:
    """For each start, whether the shape placed there has every corner on
    `floor`, tested for all starts at once."""
    xs = [x + sx for x, _ in starts for sx, _ in shape_corners]
    ys = [y + sy for _, y in starts for _, sy in shape_corners]
    return shapely.contains_xy(floor, xs, ys).reshape(len(starts), -1).all(axis=1)


def placements_at(
    shape: Loop, rotation: float, room: PreparedRoom, inventory: Inventory
) -> list[Loop]:
    """Valid placements of `shape` centred on a grid point or cornered on a
    room corner."""
    min_x, min_y, max_x, max_y = room.polygon.bounds
    step = c.MIN_WALL_LENGTH
    grid = [
        (min_x + i * step, min_y + j * step)
        for i in range(int((max_x - min_x) / step) + 1)
        for j in range(int((max_y - min_y) / step) + 1)
    ]
    turned = replace(shape, rotation=rotation)
    shape_corners = corners(turned)
    middle_x, middle_y = centre(shape_corners)
    starts = [(x - middle_x, y - middle_y) for x, y in grid]
    room_corners = room.polygon.exterior.coords[:-1]
    starts += [(x - sx, y - sy) for x, y in room_corners for sx, sy in shape_corners]
    inside = all_corners_inside(room.floor, starts, shape_corners)
    loops = [turned.moved_to(start) for start, ok in zip(starts, inside) if ok]
    return [
        loop
        for loop in loops
        if validate_construction(to_construction(loop), room, inventory).valid
    ]


def initial_solutions(
    shape: Loop, room: PreparedRoom, inventory: Inventory, deadline: float
) -> list[Loop]:
    """Placements of `shape`, most open floor first, spaced apart."""
    placed = []
    for rotation in room_rotations(room):
        if time.process_time() >= deadline:
            break  # a room with very many wall directions: keep what we have
        placed += [
            (open_floor_around(loop, room), loop)
            for loop in placements_at(shape, rotation, room, inventory)
        ]
    placed.sort(key=lambda placement: placement[0], reverse=True)
    chosen: list[Loop] = []
    for _, loop in placed:
        if chosen and time.process_time() >= deadline:
            break
        if all(math.dist(loop.start, o.start) >= c.MIN_WALL_LENGTH for o in chosen):
            chosen.append(loop)
    return chosen


# ------------------------------------------------------------------- search


class Scored(NamedTuple):
    loop: Loop
    score: float


def softmax_draw(
    neighbors: list[Scored], score: float, time_left: float, random_state: random.Random
) -> Scored:
    """A neighbor drawn with probability ~ exp(gain / temperature)."""
    gains = [neighbor.score - score for neighbor in neighbors]
    temperature = time_left * sum(map(abs, gains)) / len(gains)
    best_gain = max(gains)
    weights = [math.exp((g - best_gain) / (temperature + 1e-9)) for g in gains]
    return random_state.choices(neighbors, weights)[0]


class Player1(Player):
    """Group 1: softmax local search from rectangles, closure kept by construction."""

    def __init__(self, room, inventory, weights) -> None:
        super().__init__(PreparedRoom(room), inventory, weights)

    def evaluate(self, loop: Loop) -> float | None:
        """The simulator's score for `loop`, or None if it would be invalid.
        A loop with a corner outside the room is pruned before the (much
        slower) full validation, which would reject it anyway."""
        xs, ys = zip(*corners(loop))
        if not shapely.contains_xy(self.room.floor, xs, ys).all():
            return None
        result = validate_construction(to_construction(loop), self.room, self.inventory)
        return score_construction(result, self.weights) if result.valid else None

    def scored_neighbors(self, loop: Loop, moves) -> list[Scored]:
        return [
            Scored(neighbor, score)
            for neighbor in neighboring_loops(loop, self.inventory, moves)
            if (score := self.evaluate(neighbor)) is not None
        ]

    def hill_climb(self, loop: Loop, moves, deadline: float) -> Scored:
        """Step to the best neighbor until none is better."""
        current = Scored(loop, self.evaluate(loop))
        while time.process_time() < deadline:
            neighbors = self.scored_neighbors(current.loop, moves)
            better = [n for n in neighbors if n.score > current.score]
            if not better:
                break
            current = max(better, key=lambda n: n.score)
        return current

    def anneal(
        self, start: Loop, deadline: float, random_state: random.Random
    ) -> Scored:
        """Softmax local search from `start`; the best loop seen."""
        started = time.process_time()
        current = best = Scored(start, self.evaluate(start))
        steps_without_gain = 0
        while (now := time.process_time()) < deadline:
            neighbors = self.scored_neighbors(current.loop, MOVES)
            if not neighbors:
                break  # nothing around this loop is valid: a dead end
            time_left = (deadline - now) / (deadline - started)
            current = softmax_draw(neighbors, current.score, time_left, random_state)
            steps_without_gain += 1
            if current.score > best.score:
                if current.score - best.score > MIN_GAIN:
                    steps_without_gain = 0
                best = current
            if steps_without_gain >= CONVERGENCE_STEP_NUM:
                break  # converged: early stop
        return best

    def build_enclosure(self) -> Construction | None:
        deadline = time.process_time() + CPU_SECONDS  # covers placing and growing too
        shape = smallest_rectangle(self.inventory)
        placements = (
            initial_solutions(shape, self.room, self.inventory, deadline)
            if shape
            else []
        )
        if not placements:
            self.note = "no rectangle fits: nothing built"
            return None

        random_state = random.Random(0)
        results = []
        for placement in placements:
            if results and time.process_time() >= deadline:
                break  # out of time; the first start is always searched
            grown = self.hill_climb(placement, (resize_face,), deadline)
            results.append(self.anneal(grown.loop, deadline, random_state))
        best = max(results, key=lambda result: result.score)
        best = self.hill_climb(best.loop, (add_gate_to_face,), math.inf)
        self.note = f"{len(placements)} initial solutions; best {best.score:.0f} ({len(best.loop.faces)} faces)"
        # A: an enclosure that scores below zero is worse than none at all
        if best.score <= 0:
            self.note += " -- below zero, so nothing is built"
            return None
        return to_construction(best.loop)
