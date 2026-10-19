import time
import math
import random
import itertools

from copy import deepcopy
from src.room import Room
from statistics import mean
from src.weights import Weights
from src.inventory import Inventory
from players.player0 import Player
from shapely.geometry import LineString
from src.pieces import Piece, Connector, PieceType, ConnectorType
from src.enclosure import (
    Construction,
    validate_construction,
    score_construction,
    compute_faces,
    walk_points,
)


class EvolvingConstruction(Construction):
    def __init__(
        self,
        start,
        start_heading,
        pieces,
        connectors,
        left_inventory,
        depth,
        child_type="",
    ):
        super().__init__(start, start_heading, pieces, connectors)
        self.left_inventory = deepcopy(left_inventory)
        self.depth = depth
        self.child_type = child_type

    def remove_self_intersecting_neighbours(self, neighbours, type_of_neighbours):
        old_len = len(neighbours)
        neighbours = [x for x in neighbours if self.has_no_self_overlap(x)]
        new_len = len(neighbours)
        if new_len != old_len and type_of_neighbours != "modify_rectangular_bumps":
            pass
        return neighbours

    def get_neighbours(self, face_vs_inventory, room):
        """
        Possible actions from one state:
        generate connector x unit x position candidates
        """
        n1 = self.modify_opposite_faces(face_vs_inventory)
        n1 = self.remove_self_intersecting_neighbours(n1, "modify_opposite_faces")
        n2 = self.translate(room)
        n3 = self.modify_rectangular_bumps(face_vs_inventory)
        n3 = self.remove_self_intersecting_neighbours(n3, "modify_rectangular_bumps")
        n4 = self.rotate()
        n5 = self.add_diagonal_connectors(face_vs_inventory)
        n5 = self.remove_self_intersecting_neighbours(n5, "add_diagonal_connectors")
        neighbours = n1 + n2 + n3 + n4 + n5
        random.shuffle(neighbours)
        return neighbours

    @staticmethod
    def has_no_self_overlap(construction):
        points = walk_points(construction)[: len(construction.pieces)]
        boundary = LineString(points + [points[0]])
        return boundary.is_simple

    def add_diagonal_connectors(self, face_vs_inventory):
        neighbours = list()
        if self.left_inventory["connectors"].get("diagonal", 0) < 4:
            return neighbours
        n = len(self.pieces)
        heading = self.start_heading
        piece_headings = [heading % 360]
        for connector in self.connectors[1:]:
            heading = heading + connector.turn_angle()
            piece_headings.append(heading % 360)

        # Select opposite directions, each followed by a +90-degree corner.
        opposite_face_indices = [
            (i1, i2)
            for i1, i2 in itertools.product(range(n), range(n))
            if (
                i1 > i2
                and math.isclose(
                    abs(piece_headings[i1] - piece_headings[i2]),
                    180,
                    abs_tol=1e-7,
                )
                and self.connectors[(i1 + 1) % n].connector_type == ConnectorType.RIGHT
                and not self.connectors[(i1 + 1) % n].reflex
                and self.connectors[(i2 + 1) % n].connector_type == ConnectorType.RIGHT
                and not self.connectors[(i2 + 1) % n].reflex
            )
        ]

        for f1_index, f2_index in opposite_face_indices:
            for length, face_inventory_options in face_vs_inventory.items():
                if not face_inventory_options:
                    continue
                inventory_left = deepcopy(self.left_inventory)
                f1_inventory, f2_inventory = random.choices(face_inventory_options, k=2)
                f1_inventory = deepcopy(f1_inventory)
                f2_inventory = deepcopy(f2_inventory)
                # Note: face_vs_inventory adds one extra straight connector for addition
                f1_inventory["connectors"]["straight"] = (
                    f1_inventory["connectors"]["straight"] - 1
                )
                f2_inventory["connectors"]["straight"] = (
                    f2_inventory["connectors"]["straight"] - 1
                )

                if not self.exists_in_inventory(inventory_left, f1_inventory):
                    continue
                f1_pieces, _ = self.get_actual_inventory(f1_inventory)
                self.update_inventory(inventory_left, f1_inventory)

                if not self.exists_in_inventory(inventory_left, f2_inventory):
                    continue
                f2_pieces, _ = self.get_actual_inventory(f2_inventory)
                self.update_inventory(inventory_left, f2_inventory)

                inventory_left["connectors"]["diagonal"] = (
                    inventory_left["connectors"]["diagonal"] - 4
                )
                inventory_left["connectors"]["right"] = (
                    inventory_left["connectors"]["right"] + 2
                )

                index_vs_new_pieces = {
                    f1_index: f1_pieces,
                    f2_index: f2_pieces,
                }
                new_pieces, new_connectors = list(), list()
                for idx in range(len(self.pieces)):
                    new_pieces.append(self.pieces[idx])
                    # connectors[idx] is BEFORE pieces[idx].
                    # Replace the old right corner after an insertion.
                    if (idx - 1) % n in index_vs_new_pieces:
                        new_connectors.append(Connector(ConnectorType.DIAGONAL, False))
                    else:
                        new_connectors.append(self.connectors[idx])
                    if idx in index_vs_new_pieces:
                        inserted_pieces = index_vs_new_pieces[idx]
                        new_pieces.extend(inserted_pieces)
                        new_connectors.append(Connector(ConnectorType.DIAGONAL, False))
                        new_connectors.extend(
                            Connector(ConnectorType.STRAIGHT)
                            for _ in range(len(inserted_pieces) - 1)
                        )

                neighbour = EvolvingConstruction(
                    start=self.start,
                    start_heading=self.start_heading,
                    pieces=new_pieces,
                    connectors=new_connectors,
                    left_inventory=inventory_left,
                    depth=self.depth + 1,
                    child_type=self.child_type + "<add_diagonal_connectors>",
                )
                neighbours.append(neighbour)
        return neighbours

    def rotate(self):
        thetas = [random.randint(-180, 180) for _ in range(6)]
        return [
            EvolvingConstruction(
                start=self.start,
                start_heading=theta,
                pieces=self.pieces,
                connectors=self.connectors,
                left_inventory=self.left_inventory,
                depth=self.depth + 1,
                child_type=self.child_type + "<rotate:{}>".format(theta),
            )
            for theta in thetas
        ]

    def translate(self, room):
        min_x, min_y, max_x, max_y = room.polygon.bounds
        step_x = (max_x - min_x) / random.randint(1, 100)
        step_y = (max_y - min_y) / random.randint(1, 100)
        deltas = [
            (dx, dy)
            for dx, dy in itertools.product(
                [-step_x, 0, step_x],
                [-step_y, 0, step_y],
            )
            if dx != 0 or dy != 0
        ]
        x, y = self.start
        return [
            EvolvingConstruction(
                start=(x + del_x, y + del_y),
                start_heading=self.start_heading,
                pieces=self.pieces,
                connectors=self.connectors,
                left_inventory=self.left_inventory,
                depth=self.depth + 1,
                child_type=self.child_type + "<translate:{}|{}>".format(del_x, del_y),
            )
            for del_x, del_y in deltas
        ]

    def modify_opposite_faces(self, face_vs_inventory):
        heading = self.start_heading
        piece_headings = [heading % 360]
        for connector in self.connectors[1:]:
            heading = heading + connector.turn_angle()
            piece_headings.append(heading % 360)

        n = len(self.pieces)
        opposite_face_indices = [
            (i1, i2)
            for i1, i2 in itertools.product(
                range(len(self.pieces)), range(len(self.pieces))
            )
            if i1 > i2 and abs(piece_headings[i1] - piece_headings[i2]) % 360 == 180
        ]
        opposite_face_indices = [
            (i1, i2)
            for i1, i2 in opposite_face_indices
            if (
                not self.connectors[(i1 + 1) % n].is_straight()
                and not self.connectors[(i2 + 1) % n].is_straight()
            )
        ]
        neighbours = list()
        for face_length, face_inventory_options in face_vs_inventory.items():
            for f1_idx, f2_idx in opposite_face_indices:
                left_inventory = deepcopy(self.left_inventory)

                f1_inventory = random.choice(face_inventory_options)
                if not self.exists_in_inventory(left_inventory, f1_inventory):
                    continue
                f1_new_pieces, f1_new_connectors = self.get_actual_inventory(
                    f1_inventory
                )
                self.update_inventory(left_inventory, f1_inventory)
                new_pieces = (
                    self.pieces[: f1_idx + 1]
                    + f1_new_pieces
                    + self.pieces[f1_idx + 1 :]
                )
                new_connectors = (
                    self.connectors[: f1_idx + 1]
                    + [Connector(ConnectorType.STRAIGHT) for _ in f1_new_pieces]
                    + self.connectors[f1_idx + 1 :]
                )

                f2_inventory = random.choice(face_inventory_options)
                if not self.exists_in_inventory(left_inventory, f2_inventory):
                    continue
                f2_new_pieces, f2_new_connectors = self.get_actual_inventory(
                    f2_inventory
                )
                self.update_inventory(left_inventory, f2_inventory)
                new_pieces = (
                    new_pieces[: f2_idx + 1] + f2_new_pieces + new_pieces[f2_idx + 1 :]
                )
                new_connectors = (
                    new_connectors[: f2_idx + 1]
                    + [Connector(ConnectorType.STRAIGHT) for _ in f2_new_pieces]
                    + new_connectors[f2_idx + 1 :]
                )

                neighbours.append(
                    EvolvingConstruction(
                        start=self.start,
                        start_heading=self.start_heading,
                        pieces=new_pieces,
                        connectors=new_connectors,
                        left_inventory=left_inventory,
                        depth=self.depth + 1,
                        child_type=self.child_type + "<mof:{}>".format(face_length),
                    )
                )
        return neighbours

    def modify_rectangular_bumps(self, face_vs_inventory, no_attempts=20):
        neighbours = list()
        n = len(self.pieces)

        if n < 4:
            return neighbours

        points = walk_points(self)
        headings = [self.start_heading % 360]

        for connector in self.connectors[1:]:
            headings.append((headings[-1] + connector.turn_angle()) % 360)

        def make_neighbour(
            start_index, old_count, replacement_pieces, replacement_headings, child_type
        ):
            # Rotate the original loop so the replaced path comes first.
            indices = [(start_index + offset) % n for offset in range(n)]

            remaining_indices = indices[old_count:]

            new_pieces = replacement_pieces + [
                self.pieces[i] for i in remaining_indices
            ]
            new_headings = replacement_headings + [
                headings[i] for i in remaining_indices
            ]

            # Rebuild connectors from the directions of adjacent pieces.
            new_connectors = list()

            for i, heading in enumerate(new_headings):
                turn = (heading - new_headings[i - 1] + 180) % 360 - 180

                if math.isclose(turn, 0, abs_tol=1e-7):
                    connector = Connector(ConnectorType.STRAIGHT)
                elif math.isclose(abs(turn), 90, abs_tol=1e-7):
                    connector = Connector(ConnectorType.RIGHT, reflex=turn < 0)
                elif math.isclose(abs(turn), 45, abs_tol=1e-7):
                    connector = Connector(ConnectorType.DIAGONAL, reflex=turn < 0)
                else:
                    # For example, a 180-degree reversal.
                    return

                new_connectors.append(connector)

            # Return the old construction's inventory, then consume the new one.
            left_inventory = deepcopy(self.left_inventory)

            for pieces, connectors, change in [
                (self.pieces, self.connectors, 1),
                (new_pieces, new_connectors, -1),
            ]:
                for piece in pieces:
                    kind = "gates" if piece.is_gate else "walls"
                    key = str(piece.length)
                    left_inventory[kind][key] = (
                        left_inventory[kind].get(key, 0) + change
                    )

                for connector in connectors:
                    key = connector.connector_type.value
                    left_inventory["connectors"][key] = (
                        left_inventory["connectors"].get(key, 0) + change
                    )

            if any(
                count < 0
                for inventory in left_inventory.values()
                for count in inventory.values()
            ):
                return

            neighbours.append(
                EvolvingConstruction(
                    start=points[start_index],
                    start_heading=new_headings[0],
                    pieces=new_pieces,
                    connectors=new_connectors,
                    left_inventory=left_inventory,
                    depth=self.depth + 1,
                    child_type=self.child_type + child_type,
                )
            )

        # ADD: replace a straight segment by three sides of a rectangle.
        # The segment can contain one or more consecutive pieces.
        lengths = list(face_vs_inventory)

        if lengths:
            for _ in range(no_attempts):
                start_index = random.randrange(n)

                # Find how many pieces remain straight from this position.
                max_count = 1

                while (
                    max_count < n - 1
                    and self.connectors[(start_index + max_count) % n].is_straight()
                ):
                    max_count += 1

                old_count = random.randint(1, max_count)
                base_indices = [
                    (start_index + offset) % n for offset in range(old_count)
                ]
                base_pieces = [self.pieces[i] for i in base_indices]
                base_heading = headings[start_index]

                joining_length = random.choice(lengths)
                options = face_vs_inventory[joining_length]

                if not options:
                    continue

                # Independent choices allow the same combination twice,
                # provided the combined inventory can supply it.
                joining_face1 = random.choice(options)
                joining_face2 = random.choice(options)

                joining_pieces1, _ = self.get_actual_inventory(joining_face1)
                joining_pieces2, _ = self.get_actual_inventory(joining_face2)

                # Try both sides: an outward bump or an inward notch.
                for turn in [-90, 90]:
                    replacement_pieces = joining_pieces1 + base_pieces + joining_pieces2

                    replacement_headings = (
                        [(base_heading + turn) % 360] * len(joining_pieces1)
                        + [base_heading] * len(base_pieces)
                        + [(base_heading - turn) % 360] * len(joining_pieces2)
                    )

                    make_neighbour(
                        start_index,
                        old_count,
                        replacement_pieces,
                        replacement_headings,
                        "<add_bump:{}|{}>".format(joining_length, turn),
                    )

        # REMOVE: flatten three consecutive faces when the first and
        # third have equal lengths and point in opposite directions.
        faces = compute_faces(self)

        for face_index in range(len(faces)):
            face1 = faces[face_index]
            face2 = faces[(face_index + 1) % len(faces)]
            face3 = faces[(face_index + 2) % len(faces)]

            if len(faces) < 4:
                break

            if not math.isclose(face1.length, face3.length, abs_tol=1e-7):
                continue

            heading1 = headings[face1.piece_indices[0]]
            heading2 = headings[face2.piece_indices[0]]
            heading3 = headings[face3.piece_indices[0]]

            opposite_gap = (heading3 - heading1) % 360
            corner_gap = (heading2 - heading1) % 360

            if not math.isclose(opposite_gap, 180, abs_tol=1e-7):
                continue

            if not any(
                math.isclose(corner_gap, angle, abs_tol=1e-7) for angle in [90, 270]
            ):
                continue

            replacement_pieces = [self.pieces[i] for i in face2.piece_indices]

            make_neighbour(
                start_index=face1.piece_indices[0],
                old_count=(
                    len(face1.piece_indices)
                    + len(face2.piece_indices)
                    + len(face3.piece_indices)
                ),
                replacement_pieces=replacement_pieces,
                replacement_headings=([heading2] * len(replacement_pieces)),
                child_type="<remove_bump>",
            )
        return neighbours

    @staticmethod
    def get_actual_inventory(face_inventory):
        pieces, connectors = list(), list()
        for length, freq in face_inventory["walls"].items():
            pieces.extend([Piece(PieceType.WALL, length)] * freq)
        for length, freq in face_inventory["gates"].items():
            pieces.extend([Piece(PieceType.GATE, length)] * freq)
        for ctype, freq in face_inventory["connectors"].items():
            connectors.extend([Connector(ConnectorType(ctype), False)] * freq)
        return pieces, connectors

    @staticmethod
    def exists_in_inventory(total_inventory, face_inventory):
        for length, freq in face_inventory["walls"].items():
            if total_inventory["walls"][str(length)] < freq:
                return False
        for length, freq in face_inventory["gates"].items():
            if total_inventory["gates"][str(length)] < freq:
                return False
        for ctype, freq in face_inventory["connectors"].items():
            if total_inventory["connectors"][ctype] < freq:
                return False
        return True

    @staticmethod
    def update_inventory(total_inventory, face_inventory):
        for length, freq in face_inventory["walls"].items():
            total_inventory["walls"][str(length)] = (
                total_inventory["walls"][str(length)] - freq
            )
        for length, freq in face_inventory["gates"].items():
            total_inventory["gates"][str(length)] = (
                total_inventory["gates"][str(length)] - freq
            )
        for ctype, freq in face_inventory["connectors"].items():
            total_inventory["connectors"][ctype] = (
                total_inventory["connectors"][ctype] - freq
            )
        return True

    def __str__(self):
        return "Depth: {} | Start: {} | Start Heading: {} | Pieces: {} | Connectors: {} | child_type: {}".format(
            self.depth,
            self.start,
            self.start_heading,
            self.pieces,
            self.connectors,
            self.child_type,
        )


class Player9(Player):
    """Group 9: <one line on your strategy -- fill this in>."""

    def __init__(self, room: Room, inventory: Inventory, weights: Weights):
        super().__init__(room, inventory, weights)
        self.face_vs_inventory = self.precompute_face_generation_possibilities()

    def precompute_face_generation_possibilities(self, max_length=30):
        inventory = self.inventory.to_dict()
        piece_types = [
            (kind, int(length), count)
            for kind in ("walls", "gates")
            for length, count in inventory.get(kind, {}).items()
            if count > 0
        ]
        piece_types.sort(key=lambda item: (item[1], item[0]))

        straight_limit = inventory.get("connectors", {}).get("straight", 0)
        catalog = {}
        chosen = []

        def search(index, total_length, piece_count, gate_count):
            if index == len(piece_types):
                if piece_count == 0:
                    return
                face = {
                    "walls": {},
                    "gates": {},
                    "connectors": {"straight": piece_count},
                    "gate_count": gate_count,
                }
                for kind, length, count in chosen:
                    face[kind][length] = count
                catalog.setdefault(total_length, []).append(face)
                return
            kind, length, available = piece_types[index]
            max_count = min(
                available,
                (max_length - total_length) // length,
                straight_limit - piece_count,
            )
            for count in range(max_count + 1):
                if count:
                    chosen.append((kind, length, count))
                search(
                    index + 1,
                    total_length + count * length,
                    piece_count + count,
                    gate_count + (count if kind == "gates" else 0),
                )
                if count:
                    chosen.pop()

        search(0, 0, 0, 0)
        return dict(sorted(catalog.items()))

    def get_smallest_rectangle(self):
        inventory = self.inventory.to_dict()
        if inventory["connectors"].get("right", 0) < 4:
            return None
        length_pairs = sorted(
            itertools.combinations_with_replacement(sorted(self.face_vs_inventory), 2),
            key=lambda pair: pair[0] * pair[1],
        )
        cx = mean(x[0] for x in self.room.get_boundary_points())
        cy = mean(x[1] for x in self.room.get_boundary_points())
        for a, b in length_pairs:
            for face1 in self.face_vs_inventory[a]:
                for face2 in self.face_vs_inventory[b]:
                    for face3 in self.face_vs_inventory[a]:
                        for face4 in self.face_vs_inventory[b]:
                            faces = [face1, face2, face3, face4]
                            remaining = deepcopy(inventory)
                            remaining["connectors"]["right"] = (
                                remaining["connectors"]["right"] - 4
                            )
                            pieces, connectors = list(), list()
                            possible = True
                            for face in faces:
                                required = deepcopy(face)
                                # Note that catalog contains an extra connector.
                                required["connectors"]["straight"] = (
                                    required["connectors"]["straight"] - 1
                                )
                                if not EvolvingConstruction.exists_in_inventory(
                                    remaining, required
                                ):
                                    possible = False
                                    break
                                EvolvingConstruction.update_inventory(
                                    remaining, required
                                )
                                face_pieces, _ = (
                                    EvolvingConstruction.get_actual_inventory(face)
                                )
                                pieces.extend(face_pieces)
                                connectors.append(Connector(ConnectorType.RIGHT))
                                connectors.extend(
                                    Connector(ConnectorType.STRAIGHT)
                                    for _ in range(len(face_pieces) - 1)
                                )

                            if possible and any(p.is_gate for p in pieces):
                                return EvolvingConstruction(
                                    start=(cx, cy),
                                    start_heading=0,
                                    pieces=pieces,
                                    connectors=connectors,
                                    left_inventory=remaining,
                                    depth=0,
                                    child_type="smallest",
                                )
        return None

    def build_enclosure(self) -> Construction | None:
        construction_1, score_1 = self.generate_enclosure_from_local_search_approach()
        best_construction, best_score = construction_1, score_1
        if best_score < 0:
            best_construction = None
        return best_construction

    def generate_enclosure_from_local_search_approach(self, no_attempts=1):
        print("Generating enclosure from local search approach...")
        results = list()
        for i in range(no_attempts):
            initial_construction = self.get_smallest_rectangle()
            if initial_construction is None:
                print("No small enclosure found to start with...")
                return None, -1
            final_construction, final_score = self.run_local_search_for_a_location(
                initial_construction
            )
            print("Attempt:", i, "Score of final construction:", final_score)
            results.append((final_construction, final_score))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        final_construction, final_score = results[0]
        return final_construction, final_score

    def run_local_search_for_a_location(self, parent, max_time_in_seconds=20):
        st = time.time()
        root_validation_result = validate_construction(
            parent, self.room, self.inventory
        )
        root_score = (
            score_construction(root_validation_result, self.weights)
            if root_validation_result.valid
            else float("-inf")
        )

        last_log_time = 0
        state_count = 0
        best_construction, best_score = parent, root_score
        while True:
            seconds_passed = time.time() - st
            temperature = math.pow(2, (max_time_in_seconds - seconds_passed))
            if seconds_passed > last_log_time:
                # print("States evaluated:", state_count, "Seconds passed:", seconds_passed, "/", max_time_in_seconds, "Temperature:", temperature)
                last_log_time = last_log_time + 1
            if seconds_passed > max_time_in_seconds:
                break
            state_count = state_count + 1
            parent_validation_result = validate_construction(
                parent, self.room, self.inventory
            )
            parent_score = score_construction(parent_validation_result, self.weights)
            neighbours = parent.get_neighbours(self.face_vs_inventory, self.room)
            random.shuffle(neighbours)
            for neighbour in neighbours:
                neighbour_validation_result = validate_construction(
                    neighbour, self.room, self.inventory
                )
                neighbour_score = (
                    score_construction(neighbour_validation_result, self.weights)
                    if neighbour_validation_result.valid
                    else float("-inf")
                )
                if neighbour_score > best_score:
                    # print("Improved score from", best_score, "to", neighbour_score, "@", state_count, " | seconds passed:", seconds_passed, "/", max_time_in_seconds)
                    best_score = neighbour_score
                    best_construction = neighbour
                gain = neighbour_score - parent_score
                accept = gain > 0 or random.random() < math.pow(
                    math.e, gain / temperature
                )
                if accept:
                    parent = neighbour
                    break
        return best_construction, best_score
