"""The ways the inventory can build one linear face: on or more pieces joined
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

    def __init__(
        self,
        inventory: Inventory,
        max_pieces_per_face: int = MAX_PIECES_PER_FACE,
    ):
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
        for size in range(1, max_pieces_per_face + 1):
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

        sorted_opts = {
            key: sorted(opts, key=lambda o: (len(o.pieces), o.gates))
            for key, opts in options.items()
        }
        self._all_options = sorted_opts
        alt = ALTERNATIVES if sum(self.stock) < 80 else max(ALTERNATIVES, 24)
        self._options = {key: opts[:alt] for key, opts in sorted_opts.items()}
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

    def feasible(
        self, remaining: tuple[int, ...], length: int, with_gate: bool
    ) -> bool:
        """Whether any composition of this length still fits the remaining stock."""
        return any(
            all(remaining[i] >= n for i, n in option.uses)
            for option in self._all_options.get((length, with_gate), ())
        )

    def min_straights_for(self, length: int, with_gate: bool) -> int:
        opts = self._all_options.get((length, with_gate), ())
        if not opts:
            return 10**9
        return min(o.straights for o in opts)

    def repack(
        self,
        stock: tuple[int, ...],
        specs: list[tuple[int, bool]],
        want_two_gate_faces: bool,
    ) -> list[FaceOption] | None:
        """Joint piece allocation across all faces (inspired by other groups'
        allocate passes). Used when greedy `pick` cannot close the loop."""
        ordered = sorted(
            range(len(specs)),
            key=lambda i: (-specs[i][0], specs[i][1]),
        )
        found: list[FaceOption | None] = [None] * len(specs)

        def search(i: int, left: tuple[int, ...], gate_faces: int) -> bool:
            if i == len(ordered):
                if gate_faces == 0:
                    return False
                if want_two_gate_faces and gate_faces < 2:
                    return False
                return True
            idx = ordered[i]
            length, with_gate = specs[idx]
            for option in self._all_options.get((length, with_gate), ()):
                if not all(left[k] >= n for k, n in option.uses):
                    continue
                new_left = list(left)
                for k, n in option.uses:
                    new_left[k] -= n
                found[idx] = option
                if search(
                    i + 1,
                    tuple(new_left),
                    gate_faces + (1 if option.gates else 0),
                ):
                    return True
            return False

        if not search(0, stock, 0):
            return None
        return [found[i] for i in range(len(specs))]

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
