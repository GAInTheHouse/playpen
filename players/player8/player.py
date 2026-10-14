import math
import time

from shapely.geometry import LineString, Point, Polygon

from players.player0 import Player
from src.enclosure import Construction, explain_score, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player8(Player):
    """Find irregular enclosures with beam search and a DFS fallback."""

    def build_enclosure(self) -> Construction | None:
        BEAM_WIDTH = 4000
        MAX_PIECES = 18
        MAX_SHAPES = 30000
        PLACE_LIMIT = 2500

        # CPU-time limits, measured from the start of this function.
        BEAM_SECONDS = 110.0
        FIRST_PLACEMENT_SECONDS = 150.0
        DFS_SECONDS = 225.0
        TOTAL_SECONDS = 275.0
        EPS = 1e-4

        started = time.process_time()
        beam_deadline = started + BEAM_SECONDS
        first_placement_deadline = started + FIRST_PLACEMENT_SECONDS
        dfs_deadline = started + DFS_SECONDS
        final_deadline = started + TOTAL_SECONDS

        # Updated when DFS starts, so closure lookahead
        # uses the correct deadline.
        search_deadline = beam_deadline

        room = self.room.polygon
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        connector_stock = dict(self.inventory.connectors)

        completed = []
        seen_shapes = set()

        def pieces_left(state):
            result = []

            for kind, stock in (
                (PieceType.GATE, state["gates"]),
                (PieceType.WALL, state["walls"]),
            ):
                for length, count in stock.items():
                    # The simulator requires lengths >= 5.
                    if count > 0 and length >= 5:
                        result.append(Piece(kind, length))

            return result

        def connectors_left(state):
            stock = state["available"]
            result = []

            if stock.get(ConnectorType.STRAIGHT, 0) > 0:
                result.append((Connector(ConnectorType.STRAIGHT), 0))

            if stock.get(ConnectorType.RIGHT, 0) > 0:
                result.extend(
                    [
                        (
                            Connector(
                                ConnectorType.RIGHT,
                                reflex=False,
                            ),
                            90,
                        ),
                        (
                            Connector(
                                ConnectorType.RIGHT,
                                reflex=True,
                            ),
                            -90,
                        ),
                    ]
                )

            if stock.get(ConnectorType.DIAGONAL, 0) > 0:
                result.extend(
                    [
                        (
                            Connector(
                                ConnectorType.DIAGONAL,
                                reflex=False,
                            ),
                            45,
                        ),
                        (
                            Connector(
                                ConnectorType.DIAGONAL,
                                reflex=True,
                            ),
                            -45,
                        ),
                    ]
                )

            return result

        def record(state):
            if len(completed) >= MAX_SHAPES:
                return

            if len(state["pieces"]) < 3:
                return

            if state["gate_count"] == 0:
                return

            if math.hypot(*state["points"][-1]) > EPS:
                return

            needed_turn = ((-state["heading"] + 180) % 360) - 180

            closing_connector = next(
                (
                    conn
                    for conn, turn in connectors_left(state)
                    if abs(turn - needed_turn) < EPS
                ),
                None,
            )

            if closing_connector is None:
                return

            polygon = Polygon(state["points"][:-1])

            if not polygon.is_valid or polygon.area <= EPS:
                return

            ordered_connectors = [closing_connector] + state["between"]

            signature = tuple(
                (
                    piece.piece_type,
                    piece.length,
                    conn.connector_type,
                    (
                        conn.reflex
                        if conn.connector_type != ConnectorType.STRAIGHT
                        else False
                    ),
                )
                for piece, conn in zip(
                    state["pieces"],
                    ordered_connectors,
                )
            )

            if signature in seen_shapes:
                return

            seen_shapes.add(signature)

            approximate_score = (
                1000
                + self.weights.A * polygon.area
                + self.weights.C * polygon.length
                + (self.weights.G if state["gate_count"] >= 2 else 0)
            )

            completed.append(
                (
                    approximate_score,
                    state["pieces"],
                    ordered_connectors,
                    polygon,
                )
            )

        def extend(state, conn, turn, piece):
            face_length = state["face"] + piece.length if turn == 0 else piece.length

            if face_length > 30:
                return None

            heading = (state["heading"] + turn) % 360

            x, y = state["points"][-1]
            angle = math.radians(heading)

            nx = x + piece.length * math.cos(angle)
            ny = y + piece.length * math.sin(angle)

            if math.hypot(nx, ny) < EPS:
                nx, ny = 0.0, 0.0

            points = state["points"] + [(nx, ny)]

            if not LineString(points).is_simple:
                return None

            wl = state["walls"].copy()
            gl = state["gates"].copy()

            if piece.piece_type == PieceType.GATE:
                gl[piece.length] -= 1
            else:
                wl[piece.length] -= 1

            remaining = state["available"].copy()
            remaining[conn.connector_type] -= 1

            return {
                "pieces": state["pieces"] + [piece],
                "between": state["between"] + [conn],
                "points": points,
                "heading": heading,
                "face": face_length,
                "walls": wl,
                "gates": gl,
                "available": remaining,
                "gate_count": (
                    state["gate_count"] + (piece.piece_type == PieceType.GATE)
                ),
                "diagonals": (state["diagonals"] + (abs(turn) == 45)),
            }

        def try_finish(state):
            """Try closing with one or two more pieces."""

            if time.process_time() >= search_deadline:
                return

            record(state)

            if len(state["pieces"]) >= MAX_PIECES:
                return

            x, y = state["points"][-1]
            distance = math.hypot(x, y)

            # Try closing with one piece.
            for conn, turn in connectors_left(state):
                angle = math.radians(state["heading"] + turn)

                if (
                    abs(x + distance * math.cos(angle)) > EPS
                    or abs(y + distance * math.sin(angle)) > EPS
                ):
                    continue

                for piece in pieces_left(state):
                    if abs(piece.length - distance) <= EPS:
                        candidate = extend(
                            state,
                            conn,
                            turn,
                            piece,
                        )

                        if candidate is not None:
                            record(candidate)

            if len(state["pieces"]) + 2 > MAX_PIECES:
                return

            # Try closing with two pieces.
            for conn, turn in connectors_left(state):
                if time.process_time() >= search_deadline:
                    return

                for piece in pieces_left(state):
                    penultimate = extend(
                        state,
                        conn,
                        turn,
                        piece,
                    )

                    if penultimate is None:
                        continue

                    px, py = penultimate["points"][-1]

                    if (px, py) == (0.0, 0.0):
                        record(penultimate)
                        continue

                    remaining_distance = math.hypot(px, py)

                    desired_heading = math.degrees(math.atan2(-py, -px)) % 360

                    needed_turn = (
                        (desired_heading - penultimate["heading"] + 180) % 360
                    ) - 180

                    for final_conn, final_turn in connectors_left(penultimate):
                        if abs(final_turn - needed_turn) > 0.001:
                            continue

                        for final_piece in pieces_left(penultimate):
                            if abs(final_piece.length - remaining_distance) > EPS:
                                continue

                            closed = extend(
                                penultimate,
                                final_conn,
                                final_turn,
                                final_piece,
                            )

                            if closed is not None:
                                record(closed)

                    if len(completed) >= MAX_SHAPES:
                        return

        def priority(state):
            x, y = state["points"][-1]

            heading_turn = abs(((state["heading"] + 180) % 360) - 180)

            return (
                -0.7 * math.hypot(x, y)
                + 2 * min(state["diagonals"], 3)
                + (5 if state["gate_count"] else -5)
                + min(heading_turn, 135) * 0.045
                - 0.15 * len(state["pieces"])
            )

        initial = {
            "walls": walls,
            "gates": gates,
        }

        beam = []

        for first in pieces_left(initial):
            wl = walls.copy()
            gl = gates.copy()

            if first.piece_type == PieceType.GATE:
                gl[first.length] -= 1
            else:
                wl[first.length] -= 1

            beam.append(
                {
                    "pieces": [first],
                    "between": [],
                    "points": [
                        (0.0, 0.0),
                        (float(first.length), 0.0),
                    ],
                    "heading": 0,
                    "face": first.length,
                    "walls": wl,
                    "gates": gl,
                    "available": connector_stock.copy(),
                    "gate_count": int(first.piece_type == PieceType.GATE),
                    "diagonals": 0,
                }
            )

        # Save original starting paths for the DFS fallback.
        initial_beam = beam.copy()

        for _ in range(1, MAX_PIECES):
            if (
                not beam
                or len(completed) >= MAX_SHAPES
                or time.process_time() >= search_deadline
            ):
                break

            successors = []

            for state in beam:
                if (
                    time.process_time() >= search_deadline
                    or len(completed) >= MAX_SHAPES
                ):
                    break

                try_finish(state)

                for conn, turn in connectors_left(state):
                    if time.process_time() >= search_deadline:
                        break

                    for piece in pieces_left(state):
                        candidate = extend(
                            state,
                            conn,
                            turn,
                            piece,
                        )

                        if candidate is None:
                            continue

                        if candidate["points"][-1] == (0.0, 0.0):
                            record(candidate)
                        else:
                            successors.append(candidate)

            if time.process_time() >= search_deadline:
                break

            # Group paths to preserve different types of shapes.
            groups_by_key = {}

            for state in successors:
                x, y = state["points"][-1]

                key = (
                    state["heading"] // 45,
                    max(-3, min(3, round(x / 10))),
                    max(-3, min(3, round(y / 10))),
                    min(2, state["diagonals"]),
                    min(2, state["gate_count"]),
                )

                groups_by_key.setdefault(key, []).append(state)

            for group in groups_by_key.values():
                group.sort(
                    key=priority,
                    reverse=True,
                )

            groups = sorted(
                groups_by_key.values(),
                key=lambda g: priority(g[0]),
                reverse=True,
            )

            beam = []
            seen = set()
            layer = 0

            # Round-robin selection across different groups.
            while groups and len(beam) < BEAM_WIDTH:
                if time.process_time() >= search_deadline:
                    break

                remaining_groups = []

                for group in groups:
                    if len(beam) >= BEAM_WIDTH:
                        break

                    if layer >= len(group):
                        continue

                    state = group[layer]
                    x, y = state["points"][-1]

                    signature = (
                        round(x, 3),
                        round(y, 3),
                        state["heading"],
                        state["face"],
                        state["gate_count"],
                        tuple(sorted(state["walls"].items())),
                        tuple(sorted(state["gates"].items())),
                        tuple(
                            sorted((k.value, v) for k, v in state["available"].items())
                        ),
                        tuple(
                            (round(px, 2), round(py, 2))
                            for px, py in state["points"][-2:]
                        ),
                    )

                    if signature not in seen:
                        seen.add(signature)
                        beam.append(state)

                    if layer + 1 < len(group):
                        remaining_groups.append(group)

                groups = remaining_groups
                layer += 1

        inside = room.representative_point()
        centers = [(inside.x, inside.y)]

        boundary = list(room.exterior.coords)[:-1]

        for i, (x1, y1) in enumerate(boundary):
            x2, y2 = boundary[(i + 1) % len(boundary)]

            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2

            for fraction in (0.25, 0.5, 0.75):
                centers.append(
                    (
                        midpoint_x + fraction * (inside.x - midpoint_x),
                        midpoint_y + fraction * (inside.y - midpoint_y),
                    )
                )

        minx, miny, maxx, maxy = room.bounds

        for ix in range(1, 11):
            for iy in range(1, 11):
                cx = minx + ix * (maxx - minx) / 11
                cy = miny + iy * (maxy - miny) / 11

                if room.covers(Point(cx, cy)):
                    centers.append((cx, cy))

        centers = list(dict.fromkeys((round(x, 4), round(y, 4)) for x, y in centers))

        def evaluate(shapes, deadline):
            if not shapes:
                return None, 0.0, 0

            ranked = sorted(
                shapes,
                key=lambda s: s[0],
                reverse=True,
            )

            smallest = sorted(
                shapes,
                key=lambda s: s[3].area,
            )

            diagonal = sorted(
                shapes,
                key=lambda s: sum(
                    c.connector_type == ConnectorType.DIAGONAL for c in s[2]
                ),
                reverse=True,
            )

            chosen = []
            used = set()

            for pool, limit in (
                (ranked, 1500),
                (smallest, 600),
                (diagonal, 400),
            ):
                picked = 0

                for candidate in pool:
                    if id(candidate) in used:
                        continue

                    used.add(id(candidate))
                    chosen.append(candidate)
                    picked += 1

                    if picked >= limit or len(chosen) >= PLACE_LIMIT:
                        break

            best = None
            best_score = 0.0

            for _, pieces, ordered_connectors, polygon in chosen:
                if time.process_time() >= deadline:
                    break

                center = polygon.centroid

                vertices = [
                    (x - center.x, y - center.y)
                    for x, y in list(polygon.exterior.coords)[:-1]
                ]

                for heading in range(0, 360, 15):
                    if time.process_time() >= deadline:
                        break

                    angle = math.radians(heading)
                    cs = math.cos(angle)
                    sn = math.sin(angle)

                    rotated = [
                        (
                            dx * cs - dy * sn,
                            dx * sn + dy * cs,
                        )
                        for dx, dy in vertices
                    ]

                    start_dx = -center.x * cs + center.y * sn

                    start_dy = -center.x * sn - center.y * cs

                    for cx, cy in centers:
                        if time.process_time() >= deadline:
                            break

                        placed = Polygon([(cx + dx, cy + dy) for dx, dy in rotated])

                        if not room.covers(placed):
                            continue

                        construction = Construction(
                            start=(
                                cx + start_dx,
                                cy + start_dy,
                            ),
                            start_heading=heading,
                            pieces=pieces,
                            connectors=ordered_connectors,
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
                            best = construction

            return best, best_score, len(chosen)

        beam_count = len(completed)

        best, best_score, beam_selected = evaluate(
            completed,
            first_placement_deadline,
        )

        print(
            f"Beam: {beam_count} completed, "
            f"{beam_selected} selected, "
            f"best score {best_score:.1f}, "
            f"CPU {time.process_time() - started:.1f}s"
        )

        if best is None and time.process_time() < dfs_deadline:
            print("No positive placement from beam; starting DFS fallback.")

            # Reset completed shapes so beam search does not
            # prevent DFS from saving its own candidates.
            completed.clear()
            seen_shapes.clear()

            search_deadline = dfs_deadline

            # DFS starts again from the original inventory.
            stack = initial_beam.copy()

            explored = 0

            while stack and time.process_time() < dfs_deadline:
                if len(completed) >= MAX_SHAPES:
                    break

                state = stack.pop()
                explored += 1

                try_finish(state)

                if len(state["pieces"]) >= MAX_PIECES:
                    continue

                for conn, turn in connectors_left(state):
                    if time.process_time() >= dfs_deadline:
                        break

                    for piece in pieces_left(state):
                        child = extend(
                            state,
                            conn,
                            turn,
                            piece,
                        )

                        if child is None:
                            continue

                        if child["points"][-1] == (0.0, 0.0):
                            record(child)
                        else:
                            stack.append(child)

            dfs_count = len(completed)

            best, best_score, dfs_selected = evaluate(
                completed,
                final_deadline,
            )

            print(
                f"DFS: explored {explored}, "
                f"completed {dfs_count}, "
                f"selected {dfs_selected}, "
                f"best score {best_score:.1f}, "
                f"CPU {time.process_time() - started:.1f}s"
            )

        return best
