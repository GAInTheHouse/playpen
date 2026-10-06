import math

from shapely.geometry import Point

from players.player0 import Player
from src.enclosure import Construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

HEADINGS_DEG = list(range(0, 180, 15))

MAX_SHAPES_TO_TRY = 10

GRID_STEPS = 8


class ExamplePlayer(Player):
    """Looks for matching-length walls to pair into a rectangle, and places the
    biggest such rectangle it can fit somewhere inside the room."""

    def _candidate_shapes(self) -> list[tuple[float, int, int]]:
        minx, miny, maxx, maxy = self.room.polygon.bounds
        room_diagonal = math.hypot(maxx - minx, maxy - miny)

        candidates = []
        for gate_len, gate_count in self.inventory.gates.items():
            if gate_count < 1:
                continue
            for other_len, wall_count in self.inventory.walls.items():
                if wall_count < 1:
                    continue
                if math.hypot(gate_len, other_len) > room_diagonal:
                    continue
                if self._pieces_for(gate_len, other_len) is not None:
                    candidates.append((gate_len * other_len, gate_len, other_len))

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates

    def _pieces_for(self, gate_len: int, other_len: int) -> list[Piece] | None:
        """Build the four pieces of a rectangle, or None if the inventory can't actually
        supply them."""
        lengths = [gate_len, other_len, gate_len, other_len]
        scratch = self.inventory.copy()
        pieces = []
        for i, length in enumerate(lengths):
            piece_type = PieceType.GATE if i == 0 else PieceType.WALL
            piece = Piece(piece_type, length)
            if not scratch.take_piece(piece):
                return None
            pieces.append(piece)
        return pieces

    def _rectangle_at(
        self, pieces: list[Piece], center: tuple[float, float], heading_deg: float
    ) -> Construction:
        """A Construction for this rectangle, positioned so its centroid
        lands on `center` when rotated by `heading_deg`."""
        width, height = pieces[0].length, pieces[1].length
        theta = math.radians(heading_deg)
        # local centroid offset (half-width, half-height) rotated into place
        off_x = (width / 2) * math.cos(theta) - (height / 2) * math.sin(theta)
        off_y = (width / 2) * math.sin(theta) + (height / 2) * math.cos(theta)
        start = (center[0] - off_x, center[1] - off_y)
        return Construction(
            start=start,
            start_heading=heading_deg,
            pieces=pieces,
            connectors=[Connector(ConnectorType.RIGHT) for _ in range(4)],
        )

    def _find_placement(self, pieces: list[Piece]) -> Construction | None:
        """Try the room's centroid first, then fall back to a grid of anchor points
        across the room's bounding box for trickier (e.g. non-convex)rooms."""
        centroid = self.room.polygon.centroid
        for heading in HEADINGS_DEG:
            construction = self._rectangle_at(pieces, (centroid.x, centroid.y), heading)
            if validate_construction(construction, self.room, self.inventory).valid:
                return construction

        minx, miny, maxx, maxy = self.room.polygon.bounds
        if GRID_STEPS > 1:
            xs = [
                minx + (maxx - minx) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)
            ]
            ys = [
                miny + (maxy - miny) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)
            ]
        else:
            xs, ys = [(minx + maxx) / 2], [(miny + maxy) / 2]

        for x in xs:
            for y in ys:
                # cheap pre-filter: don't bother sweeping headings at a
                # point that isn't even inside the room
                if not self.room.polygon.contains(Point(x, y)):
                    continue
                for heading in HEADINGS_DEG:
                    construction = self._rectangle_at(pieces, (x, y), heading)
                    if validate_construction(
                        construction, self.room, self.inventory
                    ).valid:
                        return construction

        return None

    def build_enclosure(self) -> Construction | None:
        if self.lacks_connectors(ConnectorType.RIGHT, 4):
            return None

        for _, gate_len, other_len in self._candidate_shapes()[:MAX_SHAPES_TO_TRY]:
            pieces = self._pieces_for(gate_len, other_len)
            if pieces is None:
                continue  # shouldn't happen, _candidate_shapes already checked
            construction = self._find_placement(pieces)
            if construction is not None:
                return construction

        return None
