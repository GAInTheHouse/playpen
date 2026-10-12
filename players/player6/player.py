import math
from collections import Counter

from shapely import Polygon

from players.player0 import Player
from players.player6.polygon import find_fit
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player6(Player):
    """Group 6: <one line on your strategy -- fill this in>."""

    GATE = 0
    WALL = 1
    STEPS = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    )

    class Edge:
        """Represent one straight face, possibly assembled from multiple walls
        and gates using straight connectors."""

        def __init__(
            self, pieces: list[tuple[int, int]], length: int, straight_count: int
        ) -> None:
            self.pieces = pieces.copy()
            self.length = length
            self.straight_count = straight_count

        def __str__(self) -> str:
            return f"Edge{{pieces={self.pieces}, length={self.length:d}, straight_count={self.straight_count:d}}}"

    class Shape:
        """Store a candidate boundary in traversal order.
        turns[i] is the turn after edges[i], including the closing joint."""

        def __init__(
            self,
            edges: list["Player6.Edge"],
            turns: list[int],
            vertices: list[tuple[float, float]],
        ) -> None:
            self.edges = edges.copy()
            self.turns = turns.copy()
            self.vertices = vertices.copy()

        def __str__(self) -> str:
            edges_text: str = ", ".join(str(edge) for edge in self.edges)
            return f"Shape{{edges=[{edges_text}], turns={self.turns}, vertices={self.vertices}}}"

        def to_polygon(self) -> Polygon:
            return Polygon(self.vertices)

        def to_construction(
            self, start: tuple[float, float], start_heading: float
        ) -> Construction:
            """Expand faces into pieces and convert outgoing turns to vertex turns."""
            pieces = []
            outgoing_connectors = []
            for edge, turn in zip(self.edges, self.turns):
                for index, (kind, length) in enumerate(edge.pieces):
                    piece_type = (
                        PieceType.GATE if kind == Player6.GATE else PieceType.WALL
                    )
                    pieces.append(Piece(piece_type, length))
                    if index < len(edge.pieces) - 1:
                        outgoing_connectors.append(Connector(ConnectorType.STRAIGHT))
                    else:
                        if abs(turn) == 90:
                            connector_type = ConnectorType.RIGHT
                        elif abs(turn) == 45:
                            connector_type = ConnectorType.DIAGONAL
                        else:
                            raise ValueError(f"Unsupported corner turn: {turn}")
                        # A reflex connector turns right (-90 or -45 degrees).
                        outgoing_connectors.append(
                            Connector(connector_type, reflex=turn < 0)
                        )

            # Our turns follow each piece; Construction.connectors[i] precedes
            # piece i. The final outgoing turn belongs at the starting vertex.
            connectors = outgoing_connectors[-1:] + outgoing_connectors[:-1]
            return Construction(start, start_heading, pieces, connectors)

    class Materials:
        """Track remaining piece counts, connector counts, and total piece length."""

        def __init__(
            self, materials: list[list[int]], right_count: int, straight_count: int
        ) -> None:
            self.materials = materials
            self.total_piece_length = 0
            for material in materials:
                self.total_piece_length += material[1] * material[2]
            self.right_count = right_count
            self.straight_count = straight_count

        def __len__(self) -> int:
            return len(self.materials)

        def __getitem__(self, index: int) -> list[int]:
            return self.materials[index]

        def can_supply(self, edge: "Player6.Edge") -> bool:
            """Check aggregate piece demand and straight connectors without
            modifying the inventory."""
            if self.straight_count < edge.straight_count:
                return False

            needed = Counter(edge.pieces)
            available = Counter()

            for material_type, material_length, material_count in self.materials:
                available[(material_type, material_length)] += material_count

            return all(available[key] >= count for key, count in needed.items())

        def consume(self, edge: "Player6.Edge") -> None:
            """Deduct the pieces and straight connectors used by one edge.
            The corner connector is deducted separately during shape search."""
            if not self.can_supply(edge):
                raise ValueError("Not enough materials or straight connectors")

            needed = Counter(edge.pieces)
            for material in self.materials:
                key = (material[0], material[1])
                take = min(material[2], needed[key])

                material[2] -= take
                needed[key] -= take

            self.straight_count -= edge.straight_count
            self.total_piece_length -= edge.length

        def clone(self) -> "Player6.Materials":
            """Deep-copy the inventory so DFS branches cannot modify each other's
            remaining material counts."""
            return Player6.Materials(
                materials=[material.copy() for material in self.materials],
                right_count=self.right_count,
                straight_count=self.straight_count,
            )

    @staticmethod
    def is_simple_polygon(vertices: list[tuple[float, float]]) -> bool:
        """Require explicit closure and reject self-intersections, overlapping
        edges, and zero-area polygons. Concave polygons are allowed."""
        if len(vertices) < 4:
            return False
        if vertices[0] != vertices[-1]:
            return False

        polygon = Polygon(vertices)
        return polygon.is_valid and polygon.area > 0

    @staticmethod
    def generate_edges(materials: Materials) -> list["Player6.Edge"]:
        """Enumerate material combinations for straight faces of length at most 30.
        Use a fixed material order to avoid duplicate permutations."""
        options: list[Player6.Edge] = []

        def dfs(index: int, selected: list[tuple[int, int]], total_length: int) -> None:
            if index == len(materials):
                if selected:
                    options.append(
                        Player6.Edge(
                            pieces=selected.copy(),
                            length=total_length,
                            straight_count=len(selected) - 1,
                        )
                    )
                return

            material: list[int] = materials[index]
            material_type: int = material[0]
            material_length: int = material[1]
            material_count: int = material[2]

            # Bound by stock, the 30-unit face limit, and available straight
            # connectors: k pieces require k - 1 straight connectors.
            max_count: int = min(
                material_count,
                (30 - total_length) // material_length,
                materials.straight_count + 1 - len(selected),
            )

            for count in range(max_count + 1):
                dfs(
                    index=(index + 1),
                    selected=(selected + [(material_type, material_length)] * count),
                    total_length=(total_length + count * material_length),
                )

        dfs(index=0, selected=[], total_length=0)

        return options

    @staticmethod
    def generate_shapes(
        materials: Materials, edge_options: list[Edge], max_faces: int, max_shapes: int
    ) -> list[Shape]:
        """Enumerate closed orthogonal polygons using left and right 90-degree
        turns. Enforce inventory limits, at least one gate, and polygon validity."""
        if materials.right_count < 4:
            return []
        shapes: list[Player6.Shape] = []
        max_faces: int = min(max_faces, materials.right_count)

        def cannot_return_to_origin(position, remaining: Player6.Materials, edges_left):
            x, y = position
            minimum_distance = abs(x) + abs(y)
            maximum_distance = min(30 * edges_left, remaining.total_piece_length)
            return minimum_distance > maximum_distance

        def dfs(
            position,
            heading,
            edges,
            turns,
            vertices,
            total_turn,
            remaining: Player6.Materials,
        ) -> None:
            if len(edges) >= max_faces or len(shapes) >= max_shapes:
                return
            for edge in edge_options:
                if len(shapes) >= max_shapes:
                    return
                if not remaining.can_supply(edge):
                    continue

                if remaining.right_count == 0:
                    continue

                dx, dy = Player6.STEPS[heading]
                next_position = (
                    position[0] + dx * edge.length,
                    position[1] + dy * edge.length,
                )

                # Update remaining materials
                next_remaining = remaining.clone()
                next_remaining.consume(edge)
                next_remaining.right_count -= 1

                next_edges = edges + [edge]
                next_vertices = vertices + [next_position]

                # Headings use quarter-turn units: +1 is left, -1 is right.
                # Straight joins are already represented inside each Edge.
                for turn in [1, -1]:
                    if len(shapes) >= max_shapes:
                        return
                    next_heading = (heading + turn) % 4
                    next_total_turn = total_turn + turn
                    next_turns = turns + [turn * 90]

                    # Returns to the origin
                    if next_position == (0, 0):
                        # At least 4 edges
                        if len(next_edges) < 4:
                            continue
                        if next_heading != 0:
                            continue
                        # A simple polygon has total signed turning angle +/-360 degrees.
                        if abs(next_total_turn) != 4:
                            continue
                        # Reject self-intersections and degenerate polygons.
                        if not Player6.is_simple_polygon(next_vertices):
                            continue
                        # At least one gate
                        if not any(
                            piece_type == Player6.GATE
                            for edge in next_edges
                            for piece_type, _ in edge.pieces
                        ):
                            continue

                        shapes.append(
                            Player6.Shape(
                                edges=next_edges,
                                turns=next_turns,
                                vertices=next_vertices[:-1],
                            )
                        )
                    else:
                        if cannot_return_to_origin(
                            next_position, next_remaining, max_faces - len(next_edges)
                        ):
                            continue

                        dfs(
                            position=next_position,
                            heading=next_heading,
                            edges=next_edges,
                            turns=next_turns,
                            vertices=next_vertices,
                            total_turn=next_total_turn,
                            remaining=next_remaining,
                        )

        dfs(
            position=(0, 0),
            heading=0,
            edges=[],
            turns=[],
            vertices=[(0, 0)],
            total_turn=0,
            remaining=materials.clone(),
        )
        return shapes

    def find_candidates(self, max_shapes=10) -> list[Shape]:
        """Build the material inventory, enumerate straight-face options, and
        combine them into candidate shapes before room placement."""

        # Initialize materials
        materials_core: list[list[int]] = []
        # Ignore diagonal connectors now
        straight_count: int = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        right_count: int = self.inventory.connectors.get(ConnectorType.RIGHT, 0)

        for wall in self.inventory.walls:
            wall_count: int = self.inventory.walls[wall]
            materials_core.append([Player6.WALL, wall, wall_count])
        for gate in self.inventory.gates:
            gate_count: int = self.inventory.gates[gate]
            materials_core.append([Player6.GATE, gate, gate_count])

        materials: Player6.Materials = Player6.Materials(
            materials=materials_core,
            straight_count=straight_count,
            right_count=right_count,
        )

        # Build edges first
        edge_options: list[Player6.Edge] = Player6.generate_edges(materials=materials)

        # Reserve two slots for rectangles, which may fit narrow rooms even
        # when the first concave candidates cannot. Keep the total limit.
        rectangle_count = min(2, max_shapes)
        rectangles = Player6.generate_shapes(
            materials=materials,
            edge_options=edge_options,
            max_faces=4,
            max_shapes=rectangle_count,
        )
        shape_options: list[Player6.Shape] = Player6.generate_shapes(
            materials=materials,
            edge_options=edge_options,
            max_faces=6,
            max_shapes=max_shapes - len(rectangles),
        )

        return rectangles + shape_options

    def build_enclosure(self) -> Construction | None:
        # Set up for you by Player.__init__:
        #   self.room       the room footprint (self.room.polygon is a shapely Polygon;
        #                   self.room.contains_enclosure(polygon) checks a candidate fits)
        #   self.inventory  the walls, gates and connectors you may use
        #   self.weights    the scoring weights A, C and G
        #
        # Return a Construction describing a closed loop of walls/gates and
        # connectors, or None for "no solution".
        #
        # None scores 0; an invalid Construction scores -1000; a valid one scores
        # 1000 + A*area + C*perimeter (+ G for a second gate on another face).
        # So when you are not sure, return None.
        #
        # You get 300 s of CPU time for the whole run, including this class's
        # constructor (--cpu-limit changes it when testing). Past that the run
        # is stopped and scores 0.
        #
        # PLACEMENT_GUIDE.md covers placing a shape in the room; SIMULATOR_GUIDE.md
        # covers everything else; players/example_player.py is a worked example.
        candidates = self.find_candidates()

        best_construction = None
        best_score = 0
        for c in candidates:
            P = c.to_polygon()
            fit = find_fit(self.room.polygon, P)
            if fit is not None:
                _, (theta, tx, ty) = fit
                # Use the rigid transform directly, preserving the candidate's
                # first vertex and traversal direction rather than ring ordering.
                angle = math.radians(theta)
                x0, y0 = c.vertices[0]
                x1, y1 = c.vertices[1]
                start = (
                    x0 * math.cos(angle) - y0 * math.sin(angle) + tx,
                    x0 * math.sin(angle) + y0 * math.cos(angle) + ty,
                )
                heading = math.degrees(math.atan2(y1 - y0, x1 - x0)) + theta
                construction = c.to_construction(start, heading)
                result = validate_construction(construction, self.room, self.inventory)
                if not result.valid:
                    continue
                # Include the baseline and second-gate bonus, exactly as the simulator does.
                score = score_construction(result, self.weights)
                if score > best_score:
                    best_score = score
                    best_construction = construction

        return best_construction
