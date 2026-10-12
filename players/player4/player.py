import math
import time

import shapely
from shapely.geometry.polygon import orient

from players.player0 import Player
from players.player4.faces import FaceCatalog
from players.player4.search import Anchor, BeamSearch, Closed, connector_for
from src.constants import TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType

# well inside the simulator's 300 s CPU limit and the submission check's 60 s
SEARCH_SECONDS = 40.0
# a narrow pass over every anchor first, so a good answer exists early, then
# wider passes with whatever time is left
BEAM_WIDTHS = (40, 200)
MAX_FACES = 14
EDGE_FRACTIONS = (0.25, 0.5)


class Player4(Player):
    """Group 4: beam search over loops of faces joined by 45/90-degree turns,
    started from points on the room's walls, closed exactly with one or two
    solved faces, and ranked by the true score."""

    def build_enclosure(self) -> Construction | None:
        if not self.inventory.has_gate():
            self.note = "No gates in the inventory -- no valid enclosure exists"
            return None
        catalog = FaceCatalog(self.inventory)
        if not catalog.lengths:
            self.note = "No piece is short enough to use"
            return None

        connectors = (
            self.inventory.connectors.get(ConnectorType.STRAIGHT, 0),
            self.inventory.connectors.get(ConnectorType.RIGHT, 0),
            self.inventory.connectors.get(ConnectorType.DIAGONAL, 0),
        )
        room = self.room.polygon.buffer(TOL)
        shapely.prepare(room)

        begin = time.process_time()
        budget_end = begin + SEARCH_SECONDS
        anchors = room_anchors(self.room.polygon)
        best, best_score = None, 0.0
        for width in BEAM_WIDTHS:
            for k, anchor in enumerate(anchors):
                now = time.process_time()
                if now >= budget_end:
                    break
                deadline = now + (budget_end - now) / (len(anchors) - k)
                search = BeamSearch(
                    anchor, catalog, connectors, self.weights, room,
                    width, MAX_FACES, deadline,
                )  # fmt: skip
                try:
                    candidates = search.run()
                except Exception:
                    continue
                for candidate in candidates:
                    if candidate.score <= best_score:
                        break
                    construction = to_construction(anchor, candidate)
                    result = validate_construction(
                        construction, self.room, self.inventory
                    )
                    if result.valid:
                        best = construction
                        best_score = score_construction(result, self.weights)
                        break

        if best is None:
            self.note = "Beam search found no valid enclosure with a positive score"
            return None
        self.note = (
            f"Beam search: best score {best_score:.1f} from {len(best.pieces)} pieces"
            f" ({time.process_time() - begin:.1f}s CPU)"
        )
        return best


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
