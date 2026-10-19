import math
import time
from collections import Counter
from itertools import combinations_with_replacement, product

from shapely import Polygon

from players.player0 import Player
from players.player6.diagonal_search import DiagonalSearch
from players.player6.local_search import Local45Search, LocalSearchStats
from players.player6.polygon import find_fit
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player6(Player):
    """Group 6: inventory-aware grid/beam search with local 45-degree refinement."""

    GATE = 0
    WALL = 1
    SEARCH_SECONDS = 45.0
    DIAGONAL_SECONDS = 20.0
    INSCRIBED_SECONDS = 20.0
    LOCAL_SECONDS = 2.0
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

        def __repr__(self):
            return self.__str__()

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

        def to_construction_fit(self, fit):
            _, (theta, tx, ty) = fit
            # Use the rigid transform directly, preserving the candidate's
            # first vertex and traversal direction rather than ring ordering.
            angle = math.radians(theta)
            x0, y0 = self.vertices[0]
            x1, y1 = self.vertices[1]
            start = (
                x0 * math.cos(angle) - y0 * math.sin(angle) + tx,
                x0 * math.sin(angle) + y0 * math.cos(angle) + ty,
            )
            heading = math.degrees(math.atan2(y1 - y0, x1 - x0)) + theta
            return self.to_construction(start, heading)

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

    def construct(self, polygon: Polygon, fit, c: Shape):
        _, (theta, tx, ty) = fit
        # Use the rigid transform directly, preserving the candidate's
        # first vertex and traversal direction rather than ring ordering.
        angle = math.radians(theta)
        x0, y0 = polygon.exterior.coords[0]
        x1, y1 = polygon.exterior.coords[1]
        start = (
            x0 * math.cos(angle) - y0 * math.sin(angle) + tx,
            x0 * math.sin(angle) + y0 * math.cos(angle) + ty,
        )
        heading = math.degrees(math.atan2(y1 - y0, x1 - x0)) + theta
        return c.to_construction(start, heading)

    def check_face_inventory(self, pieces):
        """Check combined demand for walls and gates without changing stock."""
        needed = Counter(tuple(piece) for piece in pieces)
        return all(
            (
                self.inventory.gates if kind == Player6.GATE else self.inventory.walls
            ).get(length, 0)
            >= count
            for (kind, length), count in needed.items()
        )

    def _out_of_time(self):
        return time.process_time() >= getattr(self, "_deadline", math.inf)

    def _score_upper_bound(self, polygon):
        """Include the baseline and the largest possible second-gate bonus."""
        bonus = self.weights.G if sum(self.inventory.gates.values()) >= 2 else 0
        return (
            1000
            + self.weights.A * polygon.area
            + self.weights.C * polygon.length
            + bonus
        )

    def _fit_geometry(self, polygon):
        """Cache only identical outlines; a failed heuristic fit proves no dominance."""
        if self._out_of_time() or polygon.area > self.room.polygon.area:
            return None
        key = tuple(polygon.exterior.coords)
        if key not in self._fit_cache:
            self._fit_cache[key] = find_fit(self.room.polygon, polygon)
        return self._fit_cache[key]

    def _consider_shape(self, shape, fit):
        """Save only simulator-validated improvements, using the actual gate faces."""
        pieces = [piece for edge in shape.edges for piece in edge.pieces]
        if not any(kind == Player6.GATE for kind, _ in pieces):
            return None
        if not self.check_face_inventory(pieces):
            return None
        straights = sum(len(edge.pieces) - 1 for edge in shape.edges)
        if straights > self.inventory.connectors.get(ConnectorType.STRAIGHT, 0):
            return None
        construction = shape.to_construction_fit(fit)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return None
        score = score_construction(result, self.weights)
        if score > self.best_score:
            self.best_score = score
            self.best_shape = shape
            self.best_fit = fit
            self.best_construction = construction
        return score

    def enumerate_faces(self, max_length=31):
        """Build length-indexed face compositions without permutation duplicates."""
        gate_sizes = {
            length
            for length, count in self.inventory.gates.items()
            if 5 <= length <= 30 and count > 0
        }
        wall_sizes = {
            length
            for length, count in self.inventory.walls.items()
            if 5 <= length <= 30 and count > 0
        }
        lengths = gate_sizes | wall_sizes
        if not lengths:
            return self.possible_faces
        start = max(self.possible_faces, default=min(lengths) - 1) + 1
        straight_count = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        for length in range(start, min(max_length, 31)):
            if self._out_of_time():
                break
            possible = []
            if length in gate_sizes:
                possible.append([(Player6.GATE, length)])
            if length in wall_sizes:
                possible.append([(Player6.WALL, length)])
            for previous_length, constructions in self.possible_faces.items():
                new_length = length - previous_length
                for constructed in constructions:
                    # Adding one piece requires len(constructed) straight joints.
                    if len(constructed) > straight_count:
                        continue
                    for kind, sizes in (
                        (Player6.GATE, gate_sizes),
                        (Player6.WALL, wall_sizes),
                    ):
                        if new_length not in sizes:
                            continue
                        piece = (kind, new_length)
                        if (new_length, kind) < (
                            constructed[-1][1],
                            constructed[-1][0],
                        ):
                            continue
                        candidate = constructed + [piece]
                        if self.check_face_inventory(candidate):
                            possible.append(candidate)
            if possible:
                self.possible_faces[length] = possible
        return self.possible_faces

    def generate_quads(self):
        """Enumerate rectangles, then lazily assign inventory-backed face pieces."""
        if self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
            return [], []
        valid_quads, invalid_quads = [], []
        lengths = sorted(self.possible_faces)
        for i in lengths:
            for j in lengths:
                if self._out_of_time():
                    return valid_quads, invalid_quads
                if j < i:
                    continue
                coords = [(0, 0), (i, 0), (i, j), (0, j)]
                polygon = Polygon(coords)
                upper = self._score_upper_bound(polygon)
                if upper <= self.best_score:
                    continue
                fit = self._fit_geometry(polygon)
                if fit is None:
                    invalid_quads.append((i, j))
                    # A timeout or limited-angle failure is not proof that
                    # larger rectangles cannot fit. Do not prune other sizes.
                    continue
                valid_quads.append((i, j))

                if i == j:
                    combinations = combinations_with_replacement(
                        self.possible_faces[i], 4
                    )
                else:
                    # Keep the Cartesian product lazy: materializing all pairs
                    # can consume substantial memory before finding a match.
                    combinations = (
                        (first[0], second[0], first[1], second[1])
                        for first in combinations_with_replacement(
                            self.possible_faces[i], 2
                        )
                        for second in combinations_with_replacement(
                            self.possible_faces[j], 2
                        )
                    )
                for compositions in combinations:
                    if self._out_of_time():
                        return valid_quads, invalid_quads
                    edges = [
                        Player6.Edge(list(pieces), length, len(pieces) - 1)
                        for pieces, length in zip(compositions, (i, j, i, j))
                    ]
                    shape = Player6.Shape(edges, [90] * 4, coords)
                    score = self._consider_shape(shape, fit)
                    if score is not None and score >= upper:
                        break
        return valid_quads, invalid_quads

    def generate_quad_shapes(self):
        """Search L outlines; fit caching never skips alternative material assignments."""
        if self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 6:
            return
        lengths = sorted(self.possible_faces)
        for i in lengths:
            i_groups = {}
            for pieces in self.possible_faces[i]:
                if len(pieces) > 1:
                    i_groups.setdefault(pieces[0][1], []).append(pieces)
            for j in lengths:
                if self._out_of_time():
                    return
                if j < i:
                    continue
                j_groups = {}
                for pieces in self.possible_faces[j]:
                    if len(pieces) > 1:
                        j_groups.setdefault(pieces[0][1], []).append(pieces)
                for split_i, i_options in i_groups.items():
                    for split_j, j_options in j_groups.items():
                        if self._out_of_time():
                            return
                        coords = [
                            (0, 0),
                            (split_i, 0),
                            (split_i, split_j),
                            (i, split_j),
                            (i, j),
                            (0, j),
                        ]
                        polygon = Polygon(coords)
                        upper = self._score_upper_bound(polygon)
                        if upper <= self.best_score:
                            continue
                        # Cache by the complete outline, including outer dimensions.
                        # A failed larger L must not reject a smaller L.
                        fit = self._fit_geometry(polygon)
                        if fit is None:
                            continue
                        combinations = product(
                            i_options,
                            j_options,
                            self.possible_faces[i],
                            self.possible_faces[j],
                        )
                        for i_sel, j_sel, i_second, j_second in combinations:
                            if self._out_of_time():
                                return
                            face_pieces = [
                                [i_sel[0]],
                                [j_sel[0]],
                                i_sel[1:],
                                j_sel[1:],
                                i_second,
                                j_second,
                            ]
                            edges = [
                                Player6.Edge(
                                    list(pieces),
                                    sum(p[1] for p in pieces),
                                    len(pieces) - 1,
                                )
                                for pieces in face_pieces
                            ]
                            shape = Player6.Shape(
                                edges, [90, -90, 90, 90, 90, 90], coords
                            )
                            score = self._consider_shape(shape, fit)
                            if score is not None and score >= upper:
                                break

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

    def _consider_inscribed(self, polygon, plan):
        """Convert a typed stock plan without consuming it or the inventory."""
        polygon = polygon.simplify(1e-8, preserve_topology=True)
        coords = list(polygon.exterior.coords)[:-1]
        pools = {length: list(items) for length, items in plan.items()}
        edges, turns = [], []
        for i, curr in enumerate(coords):
            next_point = coords[(i + 1) % len(coords)]
            after = coords[(i + 2) % len(coords)]
            dx, dy = next_point[0] - curr[0], next_point[1] - curr[1]
            length = round(math.hypot(dx, dy))
            options = pools.get(length)
            if not options:
                return None
            pieces, count = options[0]
            edges.append(Player6.Edge(list(pieces), length, len(pieces) - 1))
            if count == 1:
                options.pop(0)
            else:
                options[0] = (pieces, count - 1)
            nx, ny = after[0] - next_point[0], after[1] - next_point[1]
            turn = round(math.degrees(math.atan2(dx * ny - dy * nx, dx * nx + dy * ny)))
            if abs(turn) != 90:
                return None
            turns.append(turn)
        shape = Player6.Shape(edges, turns, coords)
        return self._consider_shape(shape, (None, (0, 0, 0)))

    def full_checks(self):
        from players.player6.inscribed_polygon_stock import solve_with_stock

        # Respect the overall orthogonal deadline, reserving the diagonal
        # branch's budget even when the grid solver hits a difficult inventory.
        deadline = min(
            getattr(self, "_deadline", math.inf),
            time.process_time() + self.INSCRIBED_SECONDS,
        )
        angle_stock = {
            kind.value: self.inventory.connectors.get(kind, 0) for kind in ConnectorType
        }
        solve_with_stock(
            self.room.polygon,
            self.inventory.walls,
            gate_stock=self.inventory.gates,
            cells=tuple(range(5, 31)),
            angle_stock=angle_stock,
            stop=lambda: time.process_time() >= deadline,
            evaluate=self._consider_inscribed,
        )

    def _improve_local(self, deadline):
        budget = min(
            self.LOCAL_SECONDS - self.local_stats.cpu_seconds,
            deadline - time.process_time(),
        )
        if (
            self.best_construction is None
            or budget <= 0
            or self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 4
        ):
            return
        search = Local45Search(
            self.room,
            self.inventory,
            self.weights,
            time_budget=budget,
            stop=lambda: time.process_time() >= deadline,
        )
        candidate = search.improve(self.best_construction)
        for name, value in vars(search.stats).items():
            setattr(self.local_stats, name, getattr(self.local_stats, name) + value)
        if candidate is not None:
            result = validate_construction(candidate, self.room, self.inventory)
            if result.valid:
                score = score_construction(result, self.weights)
                if score > self.best_score:
                    self.best_score = score
                    self.best_construction = candidate
                    self.best_shape = None
                    self.best_fit = None

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

        # Keep the supplied inventory unchanged, and return the best validated
        # construction before the submission checker's 60-second CPU limit.
        overall_deadline = time.process_time() + self.SEARCH_SECONDS
        has_diagonals = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) >= 2
        self._deadline = (
            overall_deadline - min(self.DIAGONAL_SECONDS, self.SEARCH_SECONDS)
            if has_diagonals
            else overall_deadline
        )
        self._fit_cache = {}
        self.best_shape = None
        self.best_fit = None
        self.best_score = 0
        self.best_construction = None
        self.local_stats = LocalSearchStats()

        self.possible_faces = {}
        if not self.inventory.has_gate():
            return None
        self.enumerate_faces()
        self.full_checks()

        self.generate_quads()
        self.generate_quad_shapes()
        # Use only spare orthogonal-stage time, preserving the beam budget.
        self._improve_local(self._deadline)

        # Keep the rectangle/L result as a fallback. Spend only the reserved
        # time on the other branch's exact 45-degree beam search.
        seconds_left = overall_deadline - time.process_time()
        if has_diagonals and seconds_left > 0:
            diagonal = DiagonalSearch(
                self.room,
                self.inventory,
                self.weights,
                time_budget=min(self.DIAGONAL_SECONDS, seconds_left),
                face_library=self.possible_faces or None,
            ).build_enclosure()
            if diagonal is not None:
                result = validate_construction(diagonal, self.room, self.inventory)
                if result.valid:
                    score = score_construction(result, self.weights)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_construction = diagonal

        # A later beam result may have a useful local move too. Both calls
        # share LOCAL_SECONDS and remain within the same overall deadline.
        self._improve_local(overall_deadline)
        return self.best_construction
