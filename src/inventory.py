"""The fixed collection of walls, gates, and connectors a player has to build with.

There is no guarantee of a balanced number of components of any particular
size -- see Project 2: Playpen spec. An Inventory is handed to a player at
the start of the game and is otherwise read-only; players simulate usage
against their own copy while planning, and the simulator independently
re-checks the final construction against a fresh copy (see src/enclosure.py).
"""

import random
from dataclasses import dataclass, field

from src.pieces import ConnectorType, Piece


@dataclass
class Inventory:
    # length (units) -> count available
    walls: dict[int, int] = field(default_factory=dict)
    gates: dict[int, int] = field(default_factory=dict)
    # connector type -> count available (orientation/reflex is free, doesn't
    # consume a different pool)
    connectors: dict[ConnectorType, int] = field(default_factory=dict)

    def copy(self) -> "Inventory":
        return Inventory(
            walls=dict(self.walls),
            gates=dict(self.gates),
            connectors=dict(self.connectors),
        )

    def count_pieces(self) -> int:
        return sum(self.walls.values()) + sum(self.gates.values())

    def count_connectors(self) -> int:
        return sum(self.connectors.values())

    def has_gate(self) -> bool:
        return any(n > 0 for n in self.gates.values())

    def take_piece(self, piece: Piece) -> bool:
        """Attempt to consume one piece of the given type/length. Returns False
        (without mutating anything) if none remain."""
        pool = self.gates if piece.is_gate else self.walls
        if pool.get(piece.length, 0) <= 0:
            return False
        pool[piece.length] -= 1
        return True

    def take_connector(self, connector_type: ConnectorType) -> bool:
        if self.connectors.get(connector_type, 0) <= 0:
            return False
        self.connectors[connector_type] -= 1
        return True

    def to_dict(self) -> dict:
        return {
            "walls": {str(k): v for k, v in self.walls.items() if v > 0},
            "gates": {str(k): v for k, v in self.gates.items() if v > 0},
            "connectors": {k.value: v for k, v in self.connectors.items() if v > 0},
        }

    @staticmethod
    def from_dict(d: dict) -> "Inventory":
        return Inventory(
            walls={int(k): int(v) for k, v in d.get("walls", {}).items()},
            gates={int(k): int(v) for k, v in d.get("gates", {}).items()},
            connectors={
                ConnectorType(k): int(v) for k, v in d.get("connectors", {}).items()
            },
        )


def generate_inventory(
    seed_random: random.Random, difficulty: str = "generous"
) -> Inventory:
    """Generate a plausible, deliberately-unbalanced inventory.

    `difficulty` of "generous" produces plenty of pieces/connectors (easy to
    find *some* valid enclosure); "scarce" produces just enough to make
    construction an interesting constraint problem.
    """
    lengths = list(range(5, 16))
    walls = {}
    for length in lengths:
        if seed_random.random() < 0.85:
            hi = 8 if difficulty == "generous" else 3
            walls[length] = seed_random.randint(1, hi)

    gate_lengths = seed_random.sample(lengths, k=min(3, len(lengths)))
    gates = {}
    for length in gate_lengths:
        hi = 4 if difficulty == "generous" else 2
        gates[length] = seed_random.randint(1, hi)

    conn_hi = 40 if difficulty == "generous" else 10
    connectors = {
        ConnectorType.STRAIGHT: seed_random.randint(conn_hi // 2, conn_hi),
        ConnectorType.RIGHT: seed_random.randint(conn_hi // 2, conn_hi),
        ConnectorType.DIAGONAL: seed_random.randint(conn_hi // 2, conn_hi),
    }

    return Inventory(walls=walls, gates=gates, connectors=connectors)
