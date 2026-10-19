"""List legal faces, then assign shared stock to a complete boundary."""

from dataclasses import dataclass

from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH
from src.pieces import ConnectorType, Piece, PieceType

from .geometry import Budget, SearchStopped


@dataclass(frozen=True)
class Assembly:
    pieces: tuple[Piece, ...]
    usage: tuple[int, ...]
    gate: bool

    @property
    def joins(self) -> int:
        return len(self.pieces) - 1


@dataclass(frozen=True)
class Allocation:
    faces: tuple[tuple[Piece, ...], ...]

    @property
    def gate_faces(self) -> int:
        return sum(any(p.is_gate for p in face) for face in self.faces)


class Stock:
    def __init__(self, inventory, budget: Budget):
        self.items = tuple(
            (Piece(kind, length), count)
            for kind, pool in (
                (PieceType.WALL, inventory.walls),
                (PieceType.GATE, inventory.gates),
            )
            for length, count in sorted(pool.items())
            if MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH and count > 0
        )
        self.counts = tuple(count for _, count in self.items)
        self.rights = inventory.connectors.get(ConnectorType.RIGHT, 0)
        self.diagonals = inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        self.straights = inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        self.gates = sum(count for piece, count in self.items if piece.is_gate)
        self.material = sum(piece.length * count for piece, count in self.items)
        self.options: dict[int, list[Assembly]] = {}
        self.complete = False
        self._allocations = {}
        usage = [0] * len(self.items)

        def visit(first, length, pieces, size):
            budget.check()
            if len(pieces) == size:
                assembly = Assembly(
                    tuple(pieces), tuple(usage), any(p.is_gate for p in pieces)
                )
                self.options.setdefault(length, []).append(assembly)
                return
            for index in range(first, len(self.items)):
                piece, available = self.items[index]
                if usage[index] == available or length + piece.length > MAX_FACE_LENGTH:
                    continue
                usage[index] += 1
                try:
                    visit(index, length + piece.length, pieces + [piece], size)
                finally:
                    usage[index] -= 1

        # Build single-piece faces first so a short deadline leaves useful stock.
        try:
            for size in range(1, min(6, self.straights + 1) + 1):
                visit(0, 0, [], size)
            self.complete = True
        except SearchStopped:
            pass
        for choices in self.options.values():
            choices.sort(key=lambda a: (a.joins, a.gate))
        self.minimum_pieces = {
            n: min(len(a.pieces) for a in choices)
            for n, choices in self.options.items()
        }

    def supports(self, shape) -> bool:
        return (
            len(shape.lengths) <= 444
            and sum(abs(t) == 2 for t in shape.turns) <= self.rights
            and sum(abs(t) == 1 for t in shape.turns) <= self.diagonals
            and all(n in self.options for n in shape.lengths)
            and sum(shape.lengths) <= self.material
        )

    def allocate(
        self, lengths, gates: int, budget: Budget, max_nodes=4444
    ) -> Allocation | None:
        """Return a joint assignment; raise SearchStopped if work is incomplete.

        Only successes are cached. A failed partial search proves nothing.
        """
        budget.check()
        if self.gates < gates:
            return None
        if any(n not in self.options for n in lengths):
            if not self.complete:
                raise SearchStopped
            return None
        order = tuple(
            sorted(
                range(len(lengths)),
                key=lambda i: (len(self.options[lengths[i]]), -lengths[i]),
            )
        )
        key = tuple(lengths[i] for i in order), gates
        if key in self._allocations:
            return self._restore(order, self._allocations[key])
        minimum = [self.minimum_pieces[lengths[i]] for i in order]
        failed, nodes = set(), 0

        def visit(position, remaining, joins, covered):
            nonlocal nodes
            budget.check()
            nodes += 1
            if nodes > max_nodes:
                raise SearchStopped
            if position == len(order):
                return () if covered >= gates else None
            state = position, remaining, joins, min(covered, gates)
            needed = sum(minimum[position:])
            missing = max(0, gates - covered)
            if (
                state in failed
                or needed > sum(remaining)
                or needed - (len(order) - position) > joins
                or missing > len(order) - position
                or missing
                > sum(n for (p, _), n in zip(self.items, remaining) if p.is_gate)
            ):
                return None
            choices = self.options[lengths[order[position]]]
            # Put gates on distinct faces early; use walls after coverage is met.
            for assembly in sorted(
                choices, key=lambda a: (a.gate != (covered < gates), a.joins)
            ):
                if assembly.joins > joins or any(
                    a > b for a, b in zip(assembly.usage, remaining)
                ):
                    continue
                rest = visit(
                    position + 1,
                    tuple(b - a for a, b in zip(assembly.usage, remaining)),
                    joins - assembly.joins,
                    covered + assembly.gate,
                )
                if rest is not None:
                    return (assembly.pieces,) + rest
            failed.add(state)
            return None

        result = visit(0, self.counts, self.straights, 0)
        if result is None:
            if not self.complete:
                raise SearchStopped
            return None
        self._allocations[key] = result
        return self._restore(order, result)

    @staticmethod
    def _restore(order, faces) -> Allocation:
        restored = [()] * len(order)
        for index, face in zip(order, faces):
            restored[index] = face
        return Allocation(tuple(restored))
