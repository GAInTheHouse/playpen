import math
import time
from collections import Counter
from itertools import combinations_with_replacement, product

from shapely import Polygon

from players.player0 import Player
from players.player6.diagonal_search import DiagonalSearch
from players.player6.polygon import find_fit, place_at_angle
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player6(Player):
    """Group 6: <one line on your strategy -- fill this in>."""

    GATE = 0
    WALL = 1
    # CPU-time budget per method (the hard limit is 300 s for everything):
    #   inscribed polygon solver   <= INSCRIBED_SECONDS (inside the orthogonal window)
    #   orthogonal rect/L search   <= ORTHOGONAL_SECONDS in all, inscribed included
    #   exact 45-degree beam search: everything left up to TOTAL_SECONDS
    # In testing the beam kept improving for up to 240 s on some rooms (gaps of
    # 100+ s between improvements), so an early stop would have cost score;
    # extra time on the orthogonal side or the inscribed solver changed nothing.
    ORTHOGONAL_SECONDS = 30.0
    DIAGONAL_SECONDS = 250.0  # cap; the beam really gets TOTAL_SECONDS - elapsed
    # Stop the whole search at this CPU time, leaving ~50 s of the 300 s limit
    # for validation, building the answer and slow machines.
    TOTAL_SECONDS = 250.0
    # Hard cap on the inscribed-polygon solver, which has no natural stopping
    # point (it tries up to 26 cell sizes, each with a shrink bisection).
    INSCRIBED_SECONDS = 25.0
    # Time spent turning that solution's staircases into diagonals.
    INSCRIBED_DIAGONAL_SECONDS = 5.0
    INSCRIBED_DIAGONALS = False  # staircase->diagonal pass: never improved a score
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

    def full_checks(self):
        from players.player6.inscribed_polygon_stock import (
            clean_stock,
            solve_with_stock,
        )

        combined_inventory = self.inventory.walls.copy()
        for k, v in self.inventory.gates.items():
            if k in combined_inventory:
                combined_inventory[k] += v
            else:
                combined_inventory[k] = v

        solved, plan, _why = solve_with_stock(
            self.room.polygon,
            combined_inventory,
            cells=tuple(range(5, 31)),
            must_use=list(self.inventory.gates.keys()),
            angle_stock=self.inventory.connectors,
            deadline=time.process_time() + self.INSCRIBED_SECONDS,
        )
        if solved is None:
            return

        coords = list(solved.exterior.coords)[:-1]
        n_coords = len(coords)
        lengths = []
        angles = []
        for i in range(n_coords):
            prev, curr, next = coords[i - 1], coords[i], coords[(i + 1) % n_coords]
            length = math.hypot(next[0] - curr[0], next[1] - curr[1])
            v1, v2 = (
                (curr[0] - prev[0], curr[1] - prev[1]),
                (next[0] - curr[0], next[1] - curr[1]),
            )
            turn = math.degrees(
                math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
            )
            interior = 180 - turn
            lengths.append(length)
            angles.append(interior)

        edges = []
        gates = clean_stock(self.inventory.gates.copy())
        for ln in lengths:
            stock: list = plan[round(ln)]
            sizes, inv = stock[0]

            edge_sizes = []
            for size in sizes:
                # Prioritize gates
                if size in gates:
                    edge_sizes.append((Player6.GATE, size))
                    gates[size] -= 1
                    if gates[size] == 0:
                        del gates[size]
                else:
                    edge_sizes.append((Player6.WALL, size))
            edges.append(
                Player6.Edge(
                    edge_sizes,
                    round(ln),
                    len(sizes) - 1,
                )
            )

            stock[0] = (sizes, inv - 1)
            if inv - 1 == 0:
                stock.pop(0)

        angles = [round(180 - a) for a in angles]
        angles = [180 if a == 0 else a for a in angles]
        angles = angles[1:] + angles[:1]

        shape = Player6.Shape(edges, angles, coords)

        base_score = self._consider_shape(shape, (None, (0, 0, 0)))
        if self.INSCRIBED_DIAGONALS:
            self.diagonalize_inscribed(coords, base_score)

    def diagonalize_inscribed(self, coords, base_score):
        """Replace pairs of opposite staircases of the inscribed polygon with
        diagonals (see inscribed_diagonal.py), greedily, keeping any change
        that the simulator scores higher."""
        from players.player6 import inscribed_diagonal as idg

        conn = self.inventory.connectors
        max_diag = conn.get(ConnectorType.DIAGONAL, 0)
        max_right = conn.get(ConnectorType.RIGHT, 0)
        faces = idg.to_faces(coords)
        if max_diag < 2 or faces is None:
            return
        lengths, turns, alpha = faces
        deadline = time.process_time() + self.INSCRIBED_DIAGONAL_SECONDS
        realizer = DiagonalSearch(
            self.room,
            self.inventory,
            self.weights,
            time_budget=1e9,
            face_library=self.possible_faces or None,
        )
        realizer.prepare_inventory()
        room_poly = self.room.polygon
        cur_score = base_score if base_score is not None else -math.inf

        for _round in range(6):
            best_var = None
            for w1, w2 in idg.pairs(lengths, turns):
                for diag_len in idg.diag_lengths(w1["a"]):
                    if time.process_time() >= deadline or self._out_of_time():
                        return
                    var = idg.apply_pair(lengths, turns, w1, w2, diag_len)
                    if var is None:
                        continue
                    new_len, new_turn, head0 = var
                    n_diag = sum(1 for t in new_turn if abs(t) == 1)
                    n_right = sum(1 for t in new_turn if abs(t) == 2)
                    if n_diag > max_diag or n_right > max_right:
                        continue
                    verts, closure = idg.walk(new_len, new_turn)
                    if closure != (0, 0, 0, 0):
                        continue
                    poly = Polygon(verts)
                    if not poly.is_valid or poly.area <= 0:
                        continue
                    if self._score_upper_bound(poly) <= max(self.best_score, cur_score):
                        continue
                    if any(ln not in realizer.comps for ln in new_len):
                        continue  # a face length the inventory cannot build
                    assign = realizer.realize(tuple(sorted(new_len)))
                    if assign is None:
                        continue
                    pool = {}
                    for ln, comp in zip(sorted(new_len), assign):
                        pool.setdefault(ln, []).append(comp)
                    comps = [pool[ln].pop(0) for ln in new_len]
                    edges = [
                        Player6.Edge(list(c[0]), ln, c[2])
                        for ln, c in zip(new_len, comps)
                    ]
                    shape = Player6.Shape(edges, [t * 45 for t in new_turn], verts)
                    placed = place_at_angle(room_poly, poly, alpha + 45.0 * head0)
                    if placed is None:
                        continue
                    score = self._consider_shape(shape, placed)
                    if score is not None and (best_var is None or score > best_var[0]):
                        best_var = (score, new_len, new_turn, alpha + 45.0 * head0)
            if best_var is None or best_var[0] <= cur_score:
                return
            cur_score, lengths, turns, alpha = best_var

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
        t_start = self._t_start = time.process_time()
        has_diagonals = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0) >= 2
        self._deadline = t_start + self.ORTHOGONAL_SECONDS
        self.phase_seconds = {}
        self._fit_cache = {}
        self.best_shape = None
        self.best_fit = None
        self.best_score = 0
        self.best_construction = None

        self.possible_faces = {}
        if not self.inventory.has_gate():
            return None
        self.enumerate_faces()
        t = time.process_time()
        self.full_checks()
        self.phase_seconds["inscribed"] = time.process_time() - t

        if self.best_shape:
            best_area = self.best_shape.to_polygon().area
            total_area = self.room.polygon.area
            area_ratio = best_area / total_area
            # Check how good our best shape is. If we think the solution is better than what we would get using beam search, then just return as is.
            # For now, if we cover the entire polygon or on large rooms, we cover more than half.
            if area_ratio >= 0.999 or (total_area > 10000 and area_ratio >= 0.5):
                self._report_phases()
                return self.best_construction

        t = time.process_time()
        self.generate_quads()
        self.generate_quad_shapes()
        self.phase_seconds["orthogonal"] = time.process_time() - t

        # Keep the rectangle/L result as a fallback. Spend only the reserved
        # time on the other branch's exact 45-degree beam search.
        if has_diagonals:
            t = time.process_time()
            diagonal = DiagonalSearch(
                self.room,
                self.inventory,
                self.weights,
                time_budget=max(
                    0.0,
                    min(
                        self.DIAGONAL_SECONDS,
                        self.TOTAL_SECONDS - (time.process_time() - t_start),
                    ),
                ),
                face_library=self.possible_faces or None,
            ).build_enclosure()
            self.phase_seconds["diagonal beam"] = time.process_time() - t
            if diagonal is not None:
                result = validate_construction(diagonal, self.room, self.inventory)
                if result.valid:
                    score = score_construction(result, self.weights)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_construction = diagonal

        self._report_phases()
        return self.best_construction

    def _report_phases(self):
        """One line in the GUI/console: where the CPU time went."""
        parts = ", ".join(f"{k} {v:.0f}s" for k, v in self.phase_seconds.items())
        total = time.process_time() - self._t_start
        self.note = f"CPU {total:.0f}s ({parts})"
