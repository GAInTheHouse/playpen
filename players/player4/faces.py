"""The ways the inventory can build one linear face: one or more pieces joined
by STRAIGHT connectors, at most MAX_FACE_LENGTH units long."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement

from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH
from src.inventory import Inventory
from src.pieces import Piece, PieceType

MAX_PIECES_PER_FACE = 4

# options kept per (length, has_gate); later ones are fallbacks for when the
# preferred ones' pieces are already used elsewhere in the loop
ALTERNATIVES = 8


@dataclass(frozen=True)
class FaceOption:
    length: int
    pieces: tuple[Piece, ...]
    uses: tuple[tuple[int, int], ...]  # (kind index, count)
    gates: int

    @property
    def straights(self) -> int:
        return len(self.pieces) - 1


class FaceCatalog:
    """Every face the full inventory could build, indexed by length and by
    whether it contains a gate. Remaining stock is a tuple of counts, one per
    entry of `kinds`."""

    def __init__(self, inventory: Inventory):
        usable = range(MIN_WALL_LENGTH, MAX_FACE_LENGTH + 1)
        self.kinds: list[Piece] = [
            Piece(PieceType.WALL, length)
            for length, n in sorted(inventory.walls.items())
            if n > 0 and length in usable
        ] + [
            Piece(PieceType.GATE, length)
            for length, n in sorted(inventory.gates.items())
            if n > 0 and length in usable
        ]
        pool = {PieceType.WALL: inventory.walls, PieceType.GATE: inventory.gates}
        self.stock = tuple(pool[k.piece_type][k.length] for k in self.kinds)

        options: dict[tuple[int, bool], list[FaceOption]] = {}
        for size in range(1, MAX_PIECES_PER_FACE + 1):
            for combo in combinations_with_replacement(range(len(self.kinds)), size):
                length = sum(self.kinds[i].length for i in combo)
                if length > MAX_FACE_LENGTH:
                    continue
                uses = tuple(sorted(Counter(combo).items()))
                if any(n > self.stock[i] for i, n in uses):
                    continue
                gates = sum(1 for i in combo if self.kinds[i].is_gate)
                option = FaceOption(
                    length=length,
                    pieces=tuple(self.kinds[i] for i in combo),
                    uses=uses,
                    gates=gates,
                )
                options.setdefault((length, gates > 0), []).append(option)

        self._options = {
            key: sorted(opts, key=lambda o: (len(o.pieces), o.gates))[:ALTERNATIVES]
            for key, opts in options.items()
        }
        self.lengths = sorted({length for length, _ in self._options})
        self.length_set = set(self.lengths)

    def pick(
        self, remaining: tuple[int, ...], length: int, with_gate: bool
    ) -> FaceOption | None:
        """The preferred face of this length the remaining stock can still build."""
        for option in self._options.get((length, with_gate), ()):
            if all(remaining[i] >= n for i, n in option.uses):
                return option
        return None

    def total_length(self, remaining: tuple[int, ...]) -> int:
        return sum(k.length * n for k, n in zip(self.kinds, remaining))

    def has_gate(self, remaining: tuple[int, ...]) -> bool:
        return any(k.is_gate and n > 0 for k, n in zip(self.kinds, remaining))

    @staticmethod
    def take(remaining: tuple[int, ...], option: FaceOption) -> tuple[int, ...]:
        left = list(remaining)
        for i, n in option.uses:
            left[i] -= n
        return tuple(left)
