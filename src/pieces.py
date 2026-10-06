"""Data model for the physical components of a playpen enclosure.

A construction is built from a chain of `Piece`s (walls or gates) joined at
their endpoints by `Connector`s. See src/enclosure.py for how a chain of
these is turned into actual geometry.
"""

from dataclasses import dataclass
from enum import Enum

import src.constants as c


class PieceType(Enum):
    WALL = "wall"
    GATE = "gate"


class ConnectorType(Enum):
    STRAIGHT = "straight"  # 180 degrees, joins two walls into one linear face
    RIGHT = "right"  # 90 degrees (or 270, flipped)
    DIAGONAL = "diagonal"  # 135 degrees (or 225, flipped)


@dataclass(frozen=True)
class Piece:
    """A single wall or gate segment of a given integer length."""

    piece_type: PieceType
    length: int

    def __post_init__(self):
        if self.length < c.MIN_WALL_LENGTH:
            raise ValueError(
                f"piece length must be >= {c.MIN_WALL_LENGTH}, got {self.length}"
            )

    @property
    def is_gate(self) -> bool:
        return self.piece_type == PieceType.GATE

    def __str__(self):
        kind = "gate" if self.is_gate else "wall"
        return f"{kind}({self.length})"


@dataclass(frozen=True)
class Connector:
    """A connector placed at one vertex of the enclosure.

    `reflex=True` uses the connector in its "flipped" orientation, so a
    `RIGHT` connector forms a 270 degree internal angle instead of 90, and a
    `DIAGONAL` connector forms 225 instead of 135. `STRAIGHT` connectors only
    have one orientation (180 either way).
    """

    connector_type: ConnectorType
    reflex: bool = False

    def internal_angle(self) -> float:
        return c.CONNECTOR_INTERNAL_ANGLES[self.connector_type.value][
            self.reflex if self.connector_type != ConnectorType.STRAIGHT else False
        ]

    def turn_angle(self) -> float:
        """Exterior turn angle (degrees) applied to heading at this vertex.

        turn = 180 - internal_angle, matching the standard turtle-graphics /
        shoelace convention: a simple CCW polygon's turns sum to +360.
        """
        return 180.0 - self.internal_angle()

    def is_straight(self) -> bool:
        return self.connector_type == ConnectorType.STRAIGHT

    def __str__(self):
        return f"{self.connector_type.value}({self.internal_angle():.0f} deg)"
