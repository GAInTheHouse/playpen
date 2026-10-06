import math
from collections.abc import Callable
from dataclasses import dataclass

from players.player0 import Player
from src.enclosure import Construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

WALL, GATE = PieceType.WALL, PieceType.GATE
RIGHT, STRAIGHT, DIAGONAL = (
    ConnectorType.RIGHT,
    ConnectorType.STRAIGHT,
    ConnectorType.DIAGONAL,
)


def _pieces(*specs: tuple[PieceType, int]) -> list[Piece]:
    return [Piece(kind, length) for kind, length in specs]


def _corners(n: int = 4) -> list[Connector]:
    return [Connector(RIGHT) for _ in range(n)]


# The room these examples are laid out in (scenarios/invalid_examples.json):
#
#   an L: a 40x16 bottom arm and a 16x40 left arm. The 24x24 "notch"
#   (x > 16 and y > 16) is empty floor -- outside the room.


def _wrong_connector_count() -> Construction:
    # four walls but only three connectors: every vertex needs exactly one
    return Construction(
        start=(6, 3),
        start_heading=0.0,
        pieces=_pieces((GATE, 10), (WALL, 8), (WALL, 10), (WALL, 8)),
        connectors=_corners(3),
    )


def _does_not_close() -> Construction:
    # opposite sides must match; the last wall is 5 instead of 8, so the loop
    # ends 3 units short of where it started
    return Construction(
        start=(6, 3),
        start_heading=0.0,
        pieces=_pieces((GATE, 10), (WALL, 8), (WALL, 10), (WALL, 5)),
        connectors=_corners(),
    )


def _wrong_closing_angle() -> Construction:
    # the walls land back on the start point, but the joint there is declared
    # as a 135-degree DIAGONAL connector when the geometry needs a 90 degree one
    return Construction(
        start=(6, 3),
        start_heading=0.0,
        pieces=_pieces((GATE, 10), (WALL, 8), (WALL, 10), (WALL, 8)),
        connectors=[Connector(DIAGONAL), *_corners(3)],
    )


def _not_enough_inventory() -> Construction:
    # a 13x8 rectangle needs two 13-unit walls; the inventory has only one
    return Construction(
        start=(5, 3),
        start_heading=0.0,
        pieces=_pieces((WALL, 13), (GATE, 8), (WALL, 13), (WALL, 8)),
        connectors=_corners(),
    )


def _no_gate() -> Construction:
    return Construction(
        start=(6, 3),
        start_heading=0.0,
        pieces=_pieces((WALL, 10), (WALL, 8), (WALL, 10), (WALL, 8)),
        connectors=_corners(),
    )


def _self_crossing() -> Construction:
    # E10, N5, W5, S10, W5, N5: closes perfectly, but the long south wall
    # crosses the first wall halfway along it
    reflex = Connector(RIGHT, reflex=True)
    return Construction(
        start=(8, 8),
        start_heading=0.0,
        pieces=_pieces(
            (GATE, 10), (WALL, 5), (WALL, 5), (WALL, 10), (WALL, 5), (WALL, 5)
        ),
        connectors=[
            reflex,
            Connector(RIGHT),
            Connector(RIGHT),
            Connector(RIGHT),
            reflex,
            reflex,
        ],
    )


def _entirely_outside() -> Construction:
    # a perfectly good rectangle, dropped into the empty notch of the L
    return Construction(
        start=(24, 24),
        start_heading=0.0,
        pieces=_pieces((GATE, 10), (WALL, 8), (WALL, 10), (WALL, 8)),
        connectors=_corners(),
    )


def _sticks_out() -> Construction:
    # starts inside the bottom arm but runs 2 units past its right-hand wall
    return Construction(
        start=(32, 4),
        start_heading=0.0,
        pieces=_pieces((GATE, 10), (WALL, 8), (WALL, 10), (WALL, 8)),
        connectors=_corners(),
    )


def _wall_cuts_across_notch() -> Construction:
    # All four corners are inside the room -- one in each arm of the L -- but
    # the 16-unit wall joining them is a straight line, and a straight line
    # between two points in different arms cuts across the notch. This can
    # only happen in a NON-convex room; in a convex one, corners inside
    # means everything between them is inside too.
    p = (12.0, 27.0)  # in the left arm
    away = math.radians(-45)
    q = (p[0] + 16 * math.cos(away), p[1] + 16 * math.sin(away))  # in the bottom arm
    return Construction(
        start=q,
        start_heading=135.0,  # walk from q back to p, keeping the rectangle on the room side
        pieces=_pieces((WALL, 16), (WALL, 6), (GATE, 16), (WALL, 6)),
        connectors=_corners(),
    )


def _face_too_long() -> Construction:
    # 12 + 12 + 8 = 32 units of wall in one straight line (STRAIGHT joints),
    # over the 30-unit limit. Top and bottom match, so the loop closes
    # and fits; the length is the only problem.
    s = Connector(STRAIGHT)
    r = Connector(RIGHT)
    return Construction(
        start=(3, 4),
        start_heading=0.0,
        pieces=_pieces(
            (WALL, 12),
            (WALL, 12),
            (WALL, 8),
            (WALL, 6),
            (GATE, 8),
            (WALL, 12),
            (WALL, 12),
            (WALL, 6),
        ),
        connectors=[r, s, s, r, r, s, s, r],
    )


@dataclass(frozen=True)
class Example:
    title: str
    expect: str  # text the simulator's rejection reason must contain
    build: Callable[[], Construction]


EXAMPLES = [
    Example(
        "Wrong number of connectors (every vertex needs exactly one)",
        "one connector per vertex",
        _wrong_connector_count,
    ),
    Example(
        "Doesn't close: the last wall is too short, leaving a gap",
        "does not close",
        _does_not_close,
    ),
    Example(
        "Closing joint has the wrong angle (the walls meet, the angle doesn't)",
        "closing connector",
        _wrong_closing_angle,
    ),
    Example(
        "Needs more of a piece than the inventory has (only one 13-unit wall)",
        "not enough",
        _not_enough_inventory,
    ),
    Example("No gate", "at least one gate", _no_gate),
    Example("Walls cross each other", "cross themselves", _self_crossing),
    Example(
        "Entirely outside the room (in the empty notch of the L)",
        "does not fit within the room",
        _entirely_outside,
    ),
    Example(
        "Sticks out through the room's right-hand wall",
        "does not fit within the room",
        _sticks_out,
    ),
    Example(
        "Every corner is inside the room, but one wall cuts across the outside",
        "does not fit within the room",
        _wall_cuts_across_notch,
    ),
    Example(
        "A straight run of walls is 32 units long (the limit is 30)",
        "linear face",
        _face_too_long,
    ),
]


class InvalidExamplesPlayer(Player):
    """Steps through EXAMPLES, one per `build_enclosure()` call -- or, in a
    subclass with `only` set, shows that one example every time (this is what
    `--player i1` ... `--player i10` are; see players/registry.py)."""

    only: int | None = None  # index of the single example to show, or None

    def __init__(self, room, inventory, weights) -> None:
        super().__init__(room, inventory, weights)
        self._next = 0

    def build_enclosure(self) -> Construction | None:
        total = len(EXAMPLES)
        if self.only is not None:
            order = [self.only]
        else:
            order = [(self._next + k) % total for k in range(total)]

        skipped = 0
        for index in order:
            example = EXAMPLES[index]
            construction = example.build()
            result = validate_construction(construction, self.room, self.inventory)
            if result.valid or example.expect not in result.reason:
                skipped += 1  # doesn't reproduce in this room/inventory
                continue

            self._next = (index + 1) % total
            self.note = f"Example {index + 1} of {total}: {example.title}"
            if skipped:
                self.note += f"  ({skipped} skipped: they don't apply to this scenario)"
            return construction

        if self.only is not None:
            self.note = (
                f"Example {self.only + 1} doesn't apply to this scenario -- "
                "try scenarios/invalid_examples.json"
            )
        else:
            self.note = (
                "None of the examples apply to this scenario -- "
                "try scenarios/invalid_examples.json"
            )
        return None
