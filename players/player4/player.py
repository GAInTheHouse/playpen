import math
import time

import shapely
from shapely.geometry.polygon import orient

from players.player0 import Player
from players.player4.faces import FaceCatalog
from players.player4.limits import max_pieces_per_face
from players.player4.restart_anchors import anchors_from_construction
from players.player4.search import Anchor, BeamSearch, Closed, connector_for
from src.constants import TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType

# well inside the simulator's 300 s CPU limit and the submission check's 60 s
SEARCH_SECONDS = 40.0
BEAM_WIDTHS = (40, 500)
MAX_FACES = 14
EDGE_FRACTIONS = (0.25, 0.5)
TOP_WIDE_ANCHORS = 12
EXTRA_WIDE_FRACTION = 0.2


class Player4(Player):
    """Group 4: beam search over loops of faces joined by 45/90-degree turns,
    started from points on the room's walls, closed exactly with one or two
    solved faces, and ranked by the true score."""

    def build_enclosure(self) -> Construction | None:
        if not self.inventory.has_gate():
            self.note = "No gates in the inventory -- no valid enclosure exists"
            return None
        catalog = FaceCatalog(
            self.inventory,
            max_pieces_per_face=max_pieces_per_face(self.inventory),
        )
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
        walls = room_anchors(self.room.polygon)

        best, best_score, ranked = self._search_phases(
            room, catalog, connectors, walls, budget_end, BEAM_WIDTHS, track_scores=True
        )
        if best is not None and time.process_time() < budget_end:
            best, best_score = self._search_phases(
                room,
                catalog,
                connectors,
                anchors_from_construction(best),
                budget_end,
                (500,),
                best,
                best_score,
            )
        now = time.process_time()
        if now < budget_end and ranked:
            ranked.sort(key=lambda r: -r[1])
            top = [walls[i] for i, _ in ranked[:TOP_WIDE_ANCHORS]]
            slice_end = min(budget_end, now + (budget_end - now) * EXTRA_WIDE_FRACTION)
            best, best_score = self._search_phases(
                room,
                catalog,
                connectors,
                top,
                slice_end,
                (500,),
                best,
                best_score,
            )

        if best is None:
            self.note = "Beam search found no valid enclosure with a positive score"
            return None
        self.note = (
            f"Beam search: best score {best_score:.1f} from {len(best.pieces)} pieces"
            f" ({time.process_time() - begin:.1f}s CPU)"
        )
        return best

    def _search_phases(
        self,
        room,
        catalog: FaceCatalog,
        connectors: tuple[int, int, int],
        anchors: list[Anchor],
        budget_end: float,
        beam_widths: tuple[int, ...],
        best: Construction | None = None,
        best_score: float = 0.0,
        track_scores: bool = False,
    ) -> (
        tuple[Construction | None, float]
        | tuple[Construction | None, float, list[tuple[int, float]]]
    ):
        per_anchor: list[tuple[int, float]] = []
        for width in beam_widths:
            for k, anchor in enumerate(anchors):
                now = time.process_time()
                if now >= budget_end:
                    break
                remaining = len(anchors) - k
                deadline = now + (budget_end - now) / remaining if remaining else now
                anchor_best = 0.0
                search = BeamSearch(
                    anchor,
                    catalog,
                    connectors,
                    self.weights,
                    room,
                    width,
                    MAX_FACES,
                    deadline,
                )
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
                        score = score_construction(result, self.weights)
                        anchor_best = max(anchor_best, score)
                        if score > best_score:
                            best = construction
                            best_score = score
                        break
                if track_scores and anchor_best > 0:
                    per_anchor.append((k, anchor_best))
        if track_scores:
            return best, best_score, per_anchor
        return best, best_score


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
