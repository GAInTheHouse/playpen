"""Wall anchors and conversion of Player 4 beam candidates."""

import math

from shapely.geometry.polygon import orient

from players.player4.search import Anchor, Closed, connector_for
from src.constants import TOL
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType

MAX_FACES = 14
EDGE_FRACTIONS = (0.25, 0.5)


def room_anchors(polygon) -> list[Anchor]:
    """Anchors on the room's walls, each pointing along its wall so the room's
    interior is on the loop's left. Convex corners come first, then points
    partway along each wall (a corner sharper than 90 degrees can't hold a
    right-angled loop), then reflex corners; longer walls first within each."""
    ring = list(orient(polygon, 1.0).exterior.coords)[:-1]
    n = len(ring)
    ranked = []
    for i in range(n):
        prev, here, nxt = ring[i - 1], ring[i], ring[(i + 1) % n]
        dx, dy = nxt[0] - here[0], nxt[1] - here[1]
        length = math.hypot(dx, dy)
        if length < TOL:
            continue
        heading = math.degrees(math.atan2(dy, dx))
        convex = (here[0] - prev[0]) * dy - (here[1] - prev[1]) * dx > 0
        ranked.append((0 if convex else 2, -length, i, Anchor(here, heading)))
        for f in EDGE_FRACTIONS:
            point = (here[0] + f * dx, here[1] + f * dy)
            ranked.append((1, -length, i, Anchor(point, heading)))
    ranked.sort(key=lambda r: r[:3])
    return [r[3] for r in ranked]


def to_construction(anchor: Anchor, closed: Closed) -> Construction:
    pieces, connectors = [], []
    for face in closed.faces:
        for j, piece in enumerate(face.option.pieces):
            pieces.append(piece)
            connectors.append(
                connector_for(face.turn)
                if j == 0
                else Connector(ConnectorType.STRAIGHT)
            )
    return Construction(
        start=anchor.start,
        start_heading=anchor.heading_deg,
        pieces=pieces,
        connectors=connectors,
    )
