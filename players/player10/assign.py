"""Deciding whether the inventory can build a shape's faces, and from which pieces.

A face of length L is one or more pieces that add up to L, joined by straight
connectors. A gate is a piece like any other, so it can fill in for a wall.
"""

from dataclasses import dataclass

import src.constants as c
from src.inventory import Inventory
from src.pieces import ConnectorType

# (is_gate, length)
PieceKey = tuple[bool, int]

MAX_COMPOSITIONS = 100  # per face length, separately with and without a gate
MAX_SEARCH_NODES = 40000


@dataclass(frozen=True)
class Assignment:
    faces: list[tuple[PieceKey, ...]]  # the pieces of each face, in face order
    gate_faces: int  # how many faces hold a gate


class Assigner:
    def __init__(self, inventory: Inventory, want_second_gate: bool):
        self.stock: dict[PieceKey, int] = {}
        for is_gate, pool in ((False, inventory.walls), (True, inventory.gates)):
            for length, count in pool.items():
                # scenarios can list pieces shorter than the rules allow
                if count > 0 and length >= c.MIN_WALL_LENGTH:
                    self.stock[(is_gate, length)] = count
        self.straights = inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        self.gates = sum(n for (is_gate, _), n in self.stock.items() if is_gate)
        self.total_length = sum(key[1] * n for key, n in self.stock.items())
        self.targets = [2, 1] if want_second_gate and self.gates >= 2 else [1]

        keys = sorted(self.stock, key=lambda k: (-k[1], k[0]))
        # per face length: (compositions without a gate, compositions with one)
        self.compositions: dict[int, tuple[list, list]] = {}
        for length in range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1):
            found: list[tuple[PieceKey, ...]] = []
            self._compose(keys, 0, length, [], found)
            found.sort(key=len)
            plain = [f for f in found if not any(k[0] for k in f)]
            gated = [f for f in found if any(k[0] for k in f)]
            if plain or gated:
                self.compositions[length] = (
                    plain[:MAX_COMPOSITIONS],
                    gated[:MAX_COMPOSITIONS],
                )
        self.lengths = sorted(self.compositions)
        self._choices = {n: len(a) + len(b) for n, (a, b) in self.compositions.items()}
        self._cache: dict[tuple[int, ...], tuple | None] = {}
        self._supply_cache: dict[tuple[int, ...], bool] = {}
        self.max_nodes = MAX_SEARCH_NODES

    def _compose(self, keys, index, remaining, chosen, found):
        if remaining == 0:
            found.append(tuple(chosen))
            return
        if index == len(keys) or len(chosen) > self.straights:
            return
        key = keys[index]
        most = min(self.stock[key], remaining // key[1], self.straights + 1)
        for count in range(most, -1, -1):
            if len(chosen) + count > self.straights + 1:
                continue
            self._compose(
                keys,
                index + 1,
                remaining - count * key[1],
                chosen + [key] * count,
                found,
            )

    def assign(self, faces: tuple[int, ...]) -> Assignment | None:
        """Pieces for every face, with a gate on two different faces if that is
        wanted and possible, otherwise on one. None if the stock can't do it."""
        # the faces with the fewest ways to be built go first, so the pieces
        # only they can use are not taken by faces that had other choices
        ordered = tuple(sorted(faces, key=lambda n: (self._choices.get(n, 0), -n)))
        if ordered not in self._cache:
            self._cache[ordered] = self._solve(ordered)
        solved = self._cache[ordered]
        if solved is None:
            return None
        chosen, gate_faces = solved
        by_length: dict[int, list] = {}
        for length, pieces in zip(ordered, chosen):
            by_length.setdefault(length, []).append(pieces)
        return Assignment([by_length[length].pop() for length in faces], gate_faces)

    def _solve(self, ordered: tuple[int, ...]) -> tuple | None:
        if sum(ordered) > self.total_length:
            return None
        if any(length not in self.compositions for length in ordered):
            return None
        for target in self.targets:
            self._nodes = 0
            # states already shown to lead nowhere, so no dead end is explored twice
            self._failed: set[tuple] = set()
            chosen: list = []
            if self._search(
                ordered, 0, dict(self.stock), self.straights, 0, target, chosen
            ):
                return chosen, sum(1 for f in chosen if any(k[0] for k in f))
        return None

    def can_supply(self, faces: tuple[int, ...]) -> bool:
        """Whether the stock can supply these partial faces, without requiring a gate.

        This is a monotone feasibility filter for geometric search: an infeasible
        prefix cannot become feasible after more faces are appended.  Gate placement
        remains the responsibility of :meth:`assign` once a loop closes.
        """
        ordered = tuple(sorted(faces, key=lambda n: (self._choices.get(n, 0), -n)))
        if ordered in self._supply_cache:
            return self._supply_cache[ordered]
        if sum(ordered) > self.total_length or any(
            length not in self.compositions for length in ordered
        ):
            self._supply_cache[ordered] = False
            return False
        self._nodes = 0
        self._failed = set()
        possible = self._search(
            ordered,
            0,
            dict(self.stock),
            self.straights,
            0,
            0,
            [],
        )
        self._supply_cache[ordered] = possible
        return possible

    def _search(self, ordered, i, stock, straights, gate_faces, target, chosen) -> bool:
        if i == len(ordered):
            return gate_faces >= target
        if gate_faces + len(ordered) - i < target:
            return False
        state = (i, tuple(stock.values()), straights, min(gate_faces, target))
        if state in self._failed:
            return False
        self._nodes += 1
        if self._nodes > self.max_nodes:
            return False
        plain, gated = self.compositions[ordered[i]]
        options = gated + plain if gate_faces < target else plain + gated
        for pieces in options:
            if len(pieces) - 1 > straights:
                continue
            taken = []
            for key in pieces:
                if stock[key] <= 0:
                    break
                stock[key] -= 1
                taken.append(key)
            if len(taken) == len(pieces):
                chosen.append(pieces)
                has_gate = any(k[0] for k in pieces)
                if self._search(
                    ordered,
                    i + 1,
                    stock,
                    straights - (len(pieces) - 1),
                    gate_faces + has_gate,
                    target,
                    chosen,
                ):
                    return True
                chosen.pop()
            for key in taken:
                stock[key] += 1
        self._failed.add(state)
        return False
