"""Search depth limits derived from the scenario inventory."""

from src.constants import MAX_FACE_LENGTH
from src.inventory import Inventory
from src.pieces import ConnectorType

MAX_FACES_DEFAULT = 14


def search_limits(inventory: Inventory) -> dict[str, int]:
    """Caps for beam search: face length and loop size."""
    straights = inventory.connectors.get(ConnectorType.STRAIGHT, 0)
    wall_lengths = {length for length, n in inventory.walls.items() if n > 0}

    max_pieces = 4
    if len(wall_lengths) == 1:
        shortest = min(wall_lengths)
        if shortest <= 5 and straights >= 15:
            max_pieces = min(MAX_FACE_LENGTH // shortest, straights + 1)

    return {
        "max_pieces_per_face": max_pieces,
        "max_faces_deep": MAX_FACES_DEFAULT,
        "min_total_turn": -4,
        "max_total_turn": 10,
    }
