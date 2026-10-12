import math

from shapely.geometry import LineString, Polygon

from players.player0 import Player
from src.enclosure import Construction, explain_score, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player8(Player):
    """Generate closed shapes with backtracking, then try placing them in the room."""

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

        # ============================================================
        # SETTINGS
        # ============================================================

        
       

        # Safety limits so backtracking does not explode.
        MAX_PIECES = 8
        MAX_BRANCHES = 50000
        MAX_SHAPES = 300

        MAX_SHAPES_TO_PLACE = 50

        # Get room corners, except last since its repeated
        room_points = list(self.room.polygon.exterior.coords)[:-1]

        midpoints = []

        # Find midpoint for each neighboring points
        for i in range(len(room_points)):
            x1, y1 = room_points[i]
            x2, y2 = room_points[(i + 1) % len(room_points)]

            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2

            midpoints.append((midpoint_x, midpoint_y))

        # For every midpoint, we try multiple deterministic jumps toward the inside point to generate candidate centers.
        inside = self.room.polygon.representative_point()
        inside_x = inside.x
        inside_y = inside.y

        candidate_centers = [(inside_x, inside_y)]
        for midpoint_x, midpoint_y in midpoints:
            for fraction in [0.2, 0.4, 0.6, 0.8]:
                x = midpoint_x + fraction * (inside_x - midpoint_x)
                y = midpoint_y + fraction * (inside_y - midpoint_y)

                candidate_centers.append((x, y))

        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)

        connector_counts = self.inventory.connectors.copy()
        shapes = []

        branches_checked = 0

        def backtrack(
            pieces,
            connectors_between,
            points,
            x,
            y,
            heading,
            current_face_length,
            walls_left,
            gates_left,
            connectors_left,
            gate_count,
        ):
            nonlocal branches_checked

            branches_checked += 1

            # To prevent backtracking from exploding
            if branches_checked > MAX_BRANCHES:
                return

            # Stop once we already found enough shapes.
            if len(shapes) >= MAX_SHAPES:
                return

            # Need at least 3 pieces to enclose anything
            if len(shapes) >= MAX_SHAPES_TO_PLACE:
                return
            
            if len(pieces) >= 3:
                distance_to_start = math.hypot(x, y)

                # Check if we have closed the loop
                if distance_to_start < 0.001:
                    # Must contain at least one gate.
                    if gate_count == 0:
                        return

                    # Even if we have reached the beginning, need one final connector to connect last piece to first piece
                    needed_turn = ((-heading + 180) % 360) - 180

                    closing_connector = None

                    if abs(needed_turn - 90) < 0.001:
                        if connectors_left[ConnectorType.RIGHT] > 0:
                            closing_connector = Connector(
                                ConnectorType.RIGHT,
                                reflex=False,
                            )

                    elif abs(needed_turn + 90) < 0.001:
                        if connectors_left[ConnectorType.RIGHT] > 0:
                            closing_connector = Connector(
                                ConnectorType.RIGHT,
                                reflex=True,
                            )

                    elif abs(needed_turn - 45) < 0.001:
                        if connectors_left[ConnectorType.DIAGONAL] > 0:
                            closing_connector = Connector(
                                ConnectorType.DIAGONAL,
                                reflex=False,
                            )

                    elif abs(needed_turn + 45) < 0.001:
                        if connectors_left[ConnectorType.DIAGONAL] > 0:
                            closing_connector = Connector(
                                ConnectorType.DIAGONAL,
                                reflex=True,
                            )

                    elif abs(needed_turn) < 0.001:
                        if connectors_left[ConnectorType.STRAIGHT] > 0:
                            closing_connector = Connector(ConnectorType.STRAIGHT)

                    # If no connector can close the final corner, shape cannot work
                    if closing_connector is None:
                        return

                    connectors = [closing_connector] + connectors_between

                    # Must have exactly one connector per piece.
                    if len(connectors) != len(pieces):
                        return


                    polygon = Polygon(points[:-1])
                    if not polygon.is_valid:
                        return
                    if not polygon.exterior.is_simple:
                        return
                    if polygon.area <= 0:
                        return

                    # If we have 2 gates, add bonus to our score
                    approximate_gate_bonus = self.weights.G if gate_count >= 2 else 0

                    approximate_score = (
                        1000
                        + self.weights.A * polygon.area
                        + self.weights.C * polygon.length
                        + approximate_gate_bonus
                    )

                    shapes.append(
                        {
                            "pieces": pieces.copy(),
                            "connectors": connectors.copy(),
                            "polygon": polygon,
                            "score": approximate_score,
                        }
                    )

                    return

            if len(pieces) >= MAX_PIECES:
                return

            # Fill possible connectors 
            possible_connectors = []

            # Straight = no turn.
            if connectors_left[ConnectorType.STRAIGHT] > 0:
                possible_connectors.append(
                    (
                        Connector(ConnectorType.STRAIGHT),
                        0,
                    )
                )

            # Normal right connector = +90 degrees.
            if connectors_left[ConnectorType.RIGHT] > 0:
                possible_connectors.append(
                    (
                        Connector(
                            ConnectorType.RIGHT,
                            reflex=False,
                        ),
                        90,
                    )
                )

                # Flipped right connector = -90 degrees.
                possible_connectors.append(
                    (
                        Connector(
                            ConnectorType.RIGHT,
                            reflex=True,
                        ),
                        -90,
                    )
                )

            # Normal diagonal = +45 degrees.
            if connectors_left[ConnectorType.DIAGONAL] > 0:
                possible_connectors.append(
                    (
                        Connector(
                            ConnectorType.DIAGONAL,
                            reflex=False,
                        ),
                        45,
                    )
                )

                # Flipped diagonal = -45 degrees.
                possible_connectors.append(
                    (
                        Connector(
                            ConnectorType.DIAGONAL,
                            reflex=True,
                        ),
                        -45,
                    )
                )

            
            # Try every possible connector
            for connector, turn in possible_connectors:
                next_heading = heading + turn

                new_connectors_left = connectors_left.copy()
                new_connectors_left[connector.connector_type] -= 1

                # Fill with possible walls and gates
                possible_pieces = []

                # If we still have no gate, try gates FIRST, to reach valid shapes faster (Need to fact check this claim)
                if gate_count == 0:
                    for length, count in gates_left.items():
                        if count > 0 and length >= 5:
                            possible_pieces.append(Piece(PieceType.GATE, length))

                for length, count in walls_left.items():
                    if count > 0 and length >= 5:
                        possible_pieces.append(Piece(PieceType.WALL, length))

                # Gates are still allowed even if one has already been used.
                if gate_count > 0:
                    for length, count in gates_left.items():
                        if count > 0 and length >= 5:
                            possible_pieces.append(Piece(PieceType.GATE, length))

                # Try every possible peiece
                for piece in possible_pieces:

                    if connector.connector_type == ConnectorType.STRAIGHT:
                        new_face_length = current_face_length + piece.length
                    else:
                        new_face_length = piece.length

                    # check Length of 30 rule 
                    if new_face_length > 30:
                        continue

                    radians = math.radians(next_heading)

                    new_x = x + piece.length * math.cos(radians)
                    new_y = y + piece.length * math.sin(radians)

                    new_points = points + [(new_x, new_y)]

                    # Make sure paths dont cross
                    path = LineString(new_points)

                    if not path.is_simple:
                        continue
                    
                    # Update count of piece used
                    new_walls_left = walls_left.copy()
                    new_gates_left = gates_left.copy()

                    new_gate_count = gate_count

                    if piece.piece_type == PieceType.WALL:
                        new_walls_left[piece.length] -= 1
                    else:
                        new_gates_left[piece.length] -= 1
                        new_gate_count += 1

                    backtrack(
                        pieces + [piece],
                        connectors_between + [connector],
                        new_points,
                        new_x,
                        new_y,
                        next_heading,
                        new_face_length,
                        new_walls_left,
                        new_gates_left,
                        new_connectors_left,
                        new_gate_count,
                    )


        first_pieces = []

        # Try gates first because every valid pen needs one.
        for length, count in gates.items():
            if count > 0 and length >= 5:
                first_pieces.append(Piece(PieceType.GATE, length))

        for length, count in walls.items():
            if count > 0 and length >= 5:
                first_pieces.append(Piece(PieceType.WALL, length))

        for first_piece in first_pieces:
            new_walls = walls.copy()
            new_gates = gates.copy()

            gate_count = 0

            if first_piece.piece_type == PieceType.WALL:
                new_walls[first_piece.length] -= 1
            else:
                new_gates[first_piece.length] -= 1
                gate_count = 1

            # Heading is 0, so this first piece simply goes right.
            end_x = first_piece.length
            end_y = 0

            backtrack(
                pieces=[first_piece],
                connectors_between=[],
                points=[(0, 0), (end_x, end_y)],
                x=end_x,
                y=end_y,
                heading=0,
                current_face_length=first_piece.length,
                walls_left=new_walls,
                gates_left=new_gates,
                connectors_left=connector_counts.copy(),
                gate_count=gate_count,
            )

        # Check shapes that have the biggest score first
        shapes.sort(
            key=lambda shape: shape["score"],
            reverse=True,
        )

        best_construction = None
        best_score = -math.inf

        # Try rotations in 45-degree steps.
        headings = [
            0,
            45,
            90,
            135,
            180,
            225,
            270,
            315,
        ]

        for shape in shapes[:MAX_SHAPES_TO_PLACE]:
            polygon = shape["polygon"]

            # This is the center of the shape when it was built around (0, 0).
            shape_center = polygon.centroid

            for center_x, center_y in candidate_centers:
                for heading in headings:
                    theta = math.radians(heading)

                    # Vector from the shape center back to its start (0,0).
                    dx = -shape_center.x
                    dy = -shape_center.y

                    rotated_dx = dx * math.cos(theta) - dy * math.sin(theta)

                    rotated_dy = dx * math.sin(theta) + dy * math.cos(theta)

                    # Starting point for the Construction.
                    start = (
                        center_x + rotated_dx,
                        center_y + rotated_dy,
                    )

                    construction = Construction(
                        start=start,
                        start_heading=heading,
                        pieces=shape["pieces"],
                        connectors=shape["connectors"],
                    )

                    result = validate_construction(
                        construction,
                        self.room,
                        self.inventory,
                    )

                    if not result.valid:
                        continue

                    score = explain_score(
                        result,
                        self.weights,
                    ).total

                    if score > best_score:
                        best_score = score
                        best_construction = construction

        return best_construction
