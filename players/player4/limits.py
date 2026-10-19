"""Inventory-derived limits for face building."""

from src.constants import MAX_FACE_LENGTH
from src.inventory import Inventory
from src.pieces import ConnectorType


def max_pieces_per_face(inventory: Inventory) -> int:
    """How many wall/gate pieces may be joined into one linear face."""
    straights = inventory.connectors.get(ConnectorType.STRAIGHT, 0)
    wall_lengths = {length for length, n in inventory.walls.items() if n > 0}
    if len(wall_lengths) != 1:
        return 4
    shortest = min(wall_lengths)
    if shortest <= 5 and straights >= 15:
        return min(MAX_FACE_LENGTH // shortest, straights + 1)
    return 4
