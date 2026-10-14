"""Lightweight sanity tests for the core simulator engine (no pytest needed:
run with `uv run python -m tests.test_enclosure`)."""

from shapely.geometry import Polygon

from src.enclosure import (
    Construction,
    EnclosureError,
    compute_faces,
    explain_score,
    inventory_usage,
    score_construction,
    sketch_points,
    unavailable_items,
    validate_construction,
    walk_points,
)
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room
from src.weights import Weights

BIG_ROOM = Room(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
GENEROUS_INVENTORY = Inventory(
    walls={l: 20 for l in range(5, 40)},
    gates={l: 10 for l in range(5, 40)},
    connectors={
        ConnectorType.STRAIGHT: 40,
        ConnectorType.RIGHT: 40,
        ConnectorType.DIAGONAL: 40,
    },
)
WEIGHTS = Weights(A=2.0, C=-3.0, G=100.0)


def rectangle_construction(w, h, gate_side=0, start=(10, 10), heading=0.0):
    lengths = [w, h, w, h]
    pieces = [
        Piece(PieceType.GATE if i == gate_side else PieceType.WALL, lengths[i])
        for i in range(4)
    ]
    connectors = [Connector(ConnectorType.RIGHT) for _ in range(4)]
    return Construction(
        start=start, start_heading=heading, pieces=pieces, connectors=connectors
    )


def test_valid_rectangle_closes_and_scores():
    con = rectangle_construction(10, 6)
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert result.valid, result.reason
    assert abs(result.area - 60.0) < 1e-6
    assert abs(result.perimeter - 32.0) < 1e-6
    assert result.gate_count == 1
    assert not result.second_gate_bonus
    score = score_construction(result, WEIGHTS)
    assert abs(score - (1000 + 2.0 * 60 - 3.0 * 32)) < 1e-6


def test_second_gate_on_different_face_gives_bonus():
    con = rectangle_construction(10, 6)
    con.pieces[2] = Piece(
        PieceType.GATE, con.pieces[2].length
    )  # opposite side is also a gate
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert result.valid, result.reason
    assert result.gate_count == 2
    assert result.second_gate_bonus


def test_two_gates_on_same_face_no_bonus():
    # split one side into two walls joined by a STRAIGHT connector, put a gate
    # on each half -- still one linear face, so no second-gate bonus
    pieces = [
        Piece(PieceType.GATE, 5),
        Piece(PieceType.GATE, 5),
        Piece(PieceType.WALL, 6),
        Piece(PieceType.WALL, 10),
        Piece(PieceType.WALL, 6),
    ]
    connectors = [
        Connector(ConnectorType.RIGHT),  # vertex 0: corner
        Connector(ConnectorType.STRAIGHT),  # vertex 1: joins the two gate halves
        Connector(ConnectorType.RIGHT),  # vertex 2: corner
        Connector(ConnectorType.RIGHT),  # vertex 3: corner
        Connector(ConnectorType.RIGHT),  # vertex 4: corner
    ]
    con = Construction(
        start=(10, 10), start_heading=0.0, pieces=pieces, connectors=connectors
    )
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert result.valid, result.reason
    assert result.gate_count == 2
    assert not result.second_gate_bonus


def test_no_gate_is_invalid():
    con = rectangle_construction(10, 6)
    con.pieces[0] = Piece(PieceType.WALL, con.pieces[0].length)
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert not result.valid
    assert "gate" in result.reason


def test_face_too_long_is_invalid():
    # a rectangle (4 right-angle corners) whose bottom side is split into
    # three 12-unit walls joined by STRAIGHT connectors: that face is 36
    # units, over the 30-unit limit, even though the corners still close.
    pieces = [
        Piece(PieceType.WALL, 12),  # A1 (bottom, east)
        Piece(PieceType.WALL, 12),  # A2 (bottom, east)
        Piece(PieceType.WALL, 12),  # A3 (bottom, east) -- A1+A2+A3 = 36, the long face
        Piece(PieceType.WALL, 10),  # right side (north), height
        Piece(PieceType.GATE, 36),  # top (west) -- must equal A1+A2+A3 to close
        Piece(PieceType.WALL, 10),  # left side (south), height
    ]
    connectors = [
        Connector(ConnectorType.RIGHT),  # vertex0: between left side and A1 (corner)
        Connector(ConnectorType.STRAIGHT),  # vertex1: between A1 and A2
        Connector(ConnectorType.STRAIGHT),  # vertex2: between A2 and A3
        Connector(ConnectorType.RIGHT),  # vertex3: between A3 and right side (corner)
        Connector(ConnectorType.RIGHT),  # vertex4: between right side and top (corner)
        Connector(ConnectorType.RIGHT),  # vertex5: between top and left side (corner)
    ]
    con = Construction(
        start=(10, 10), start_heading=0.0, pieces=pieces, connectors=connectors
    )
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert not result.valid
    assert "face" in result.reason


def test_enclosure_outside_room_is_invalid():
    small_room = Room(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]))
    con = rectangle_construction(10, 6, start=(0, 0))
    result = validate_construction(con, small_room, GENEROUS_INVENTORY)
    assert not result.valid
    assert "fit" in result.reason


def test_insufficient_inventory_is_invalid():
    scarce = Inventory(
        walls={10: 1, 6: 1},
        gates={10: 1},
        connectors={ConnectorType.RIGHT: 4},
    )
    con = rectangle_construction(10, 6)  # needs 2x len-10 walls, only 1 available
    result = validate_construction(con, BIG_ROOM, scarce)
    assert not result.valid
    assert "inventory" in result.reason


def test_mismatched_connector_count_raises_clean_error():
    pieces = [
        Piece(PieceType.WALL, 10),
        Piece(PieceType.GATE, 6),
        Piece(PieceType.WALL, 10),
    ]
    connectors = [Connector(ConnectorType.RIGHT), Connector(ConnectorType.RIGHT)]
    con = Construction(
        start=(0, 0), start_heading=0.0, pieces=pieces, connectors=connectors
    )
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    assert not result.valid
    assert "one connector per vertex" in result.reason


def test_compute_faces_groups_correctly():
    con = rectangle_construction(10, 6)
    faces = compute_faces(con)
    assert len(faces) == 4
    assert all(len(f.piece_indices) == 1 for f in faces)


def _four_walls(n_connectors: int, n_pieces: int = 4) -> Construction:
    pieces = [Piece(PieceType.WALL, 10 if i % 2 == 0 else 8) for i in range(n_pieces)]
    return Construction(
        start=(0, 0),
        start_heading=0.0,
        pieces=pieces,
        connectors=[Connector(ConnectorType.RIGHT) for _ in range(n_connectors)],
    )


def test_sketch_matches_the_strict_walk_when_nothing_is_wrong():
    con = rectangle_construction(10, 6)
    points, missing = sketch_points(con)
    assert missing == []
    assert points == walk_points(con)


def test_sketch_flags_the_vertex_with_no_connector_and_still_closes_a_rectangle():
    points, missing = sketch_points(_four_walls(3))  # 4 walls, 3 connectors
    assert missing == [3]
    assert len(points) == 5
    # the wall after the missing connector repeats the previous turn, so the
    # forgotten corner of this rectangle lands where the rectangle wanted it
    assert abs(points[-1][0] - points[0][0]) < 1e-9
    assert abs(points[-1][1] - points[0][1]) < 1e-9


def test_sketch_never_raises_however_broken_the_construction_is():
    for n_connectors, n_pieces in [(0, 4), (1, 4), (5, 4), (2, 2), (0, 0), (3, 1)]:
        points, missing = sketch_points(_four_walls(n_connectors, n_pieces))
        assert len(points) == n_pieces + 1
        assert missing == list(range(n_connectors, n_pieces))


def test_the_strict_walk_still_rejects_what_the_sketch_tolerates():
    for n_connectors, n_pieces in [(3, 4), (5, 4), (2, 2)]:
        con = _four_walls(n_connectors, n_pieces)
        try:
            walk_points(con)
        except EnclosureError:
            continue
        raise AssertionError(
            f"strict walk accepted {n_connectors} connectors, {n_pieces} pieces"
        )


def test_score_breakdown_terms_add_up_and_match_the_score():
    con = rectangle_construction(10, 6)
    con.pieces[2] = Piece(PieceType.GATE, con.pieces[2].length)  # 2nd gate, other face
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    b = explain_score(result, WEIGHTS)
    assert b.valid
    assert (b.baseline, b.area_term, b.perimeter_term, b.gate_bonus) == (
        1000.0,
        2.0 * 60,
        -3.0 * 32,
        100.0,
    )
    assert (
        abs(b.total - (b.baseline + b.area_term + b.perimeter_term + b.gate_bonus))
        < 1e-9
    )
    assert b.total == score_construction(result, WEIGHTS)


def test_score_breakdown_for_an_invalid_enclosure_is_just_the_penalty():
    con = rectangle_construction(10, 6)
    con.pieces[0] = Piece(PieceType.WALL, con.pieces[0].length)  # no gate
    result = validate_construction(con, BIG_ROOM, GENEROUS_INVENTORY)
    b = explain_score(result, WEIGHTS)
    assert not b.valid
    assert (b.baseline, b.area_term, b.perimeter_term, b.gate_bonus) == (0, 0, 0, 0)
    assert b.total == -1000.0 == score_construction(result, WEIGHTS)


def test_unavailable_items_lists_the_piece_the_stock_runs_out_on():
    # two 10-unit walls wanted, one in stock: the SECOND one is unavailable
    inventory = Inventory(
        walls={10: 1, 6: 2},
        gates={10: 1},
        connectors={ConnectorType.RIGHT: 4},
    )
    con = Construction(
        start=(0, 0),
        start_heading=0.0,
        pieces=[
            Piece(PieceType.GATE, 10),
            Piece(PieceType.WALL, 6),
            Piece(PieceType.WALL, 10),  # the one in stock
            Piece(PieceType.WALL, 10),  # no second one
        ],
        connectors=[Connector(ConnectorType.RIGHT) for _ in range(4)],
    )
    short_pieces, short_connectors = unavailable_items(con, inventory)
    assert short_pieces == [3]
    assert short_connectors == []
    assert inventory.walls == {10: 1, 6: 2}  # the caller's inventory is untouched


def test_unavailable_items_lists_connectors_the_stock_runs_out_on():
    inventory = Inventory(
        walls={10: 4, 6: 4},
        gates={10: 2},
        connectors={ConnectorType.RIGHT: 3},  # 4 corners wanted
    )
    short_pieces, short_connectors = unavailable_items(
        rectangle_construction(10, 6), inventory
    )
    assert short_pieces == []
    assert short_connectors == [3]


def test_unavailable_items_is_empty_when_everything_is_in_stock():
    assert unavailable_items(rectangle_construction(10, 6), GENEROUS_INVENTORY) == (
        [],
        [],
    )


def test_validation_reports_the_first_unavailable_item():
    inventory = Inventory(
        walls={6: 4}, gates={10: 2}, connectors={ConnectorType.RIGHT: 4}
    )
    result = validate_construction(rectangle_construction(10, 6), BIG_ROOM, inventory)
    assert not result.valid
    assert "wall(10)" in result.reason  # same message as before the refactor


def _rows(construction, inventory):
    return [
        (r.kind, r.key, r.available, r.used)
        for r in inventory_usage(construction, inventory)
    ]


def test_inventory_usage_lists_everything_and_marks_what_is_used():
    inventory = Inventory(
        walls={10: 6, 6: 2, 13: 1},
        gates={10: 1},
        connectors={ConnectorType.RIGHT: 4, ConnectorType.STRAIGHT: 12},
    )
    con = rectangle_construction(10, 6, gate_side=0)  # gate 10, walls 6, 10, 6; 4 RIGHT
    assert _rows(con, inventory) == [
        ("wall", 6, 2, 2),
        ("wall", 10, 6, 1),
        ("wall", 13, 1, 0),
        ("gate", 10, 1, 1),
        ("connector", "straight", 12, 0),
        ("connector", "right", 4, 4),
        ("connector", "diagonal", 0, 0),  # all three types are always listed
    ]


def test_inventory_usage_with_nothing_built_uses_nothing():
    inventory = Inventory(
        walls={10: 6}, gates={8: 1}, connectors={ConnectorType.RIGHT: 4}
    )
    assert all(r.used == 0 for r in inventory_usage(None, inventory))
    assert [(r.kind, r.key) for r in inventory_usage(None, inventory)] == [
        ("wall", 10),
        ("gate", 8),
        ("connector", "straight"),
        ("connector", "right"),
        ("connector", "diagonal"),
    ]


def test_inventory_usage_reports_shortages_and_pieces_the_inventory_lacks_entirely():
    inventory = Inventory(walls={13: 1}, gates={}, connectors={ConnectorType.RIGHT: 3})
    con = Construction(
        start=(0, 0),
        start_heading=0.0,
        pieces=[
            Piece(PieceType.WALL, 13),
            Piece(PieceType.WALL, 13),  # only one in stock
            Piece(PieceType.GATE, 8),  # no gates at all
            Piece(PieceType.WALL, 99),  # no such wall
        ],
        connectors=[Connector(ConnectorType.RIGHT)] * 4,  # one more than in stock
    )
    rows = {(r.kind, r.key): r for r in inventory_usage(con, inventory)}
    assert (
        rows["wall", 13].available,
        rows["wall", 13].used,
        rows["wall", 13].short,
    ) == (1, 2, 1)
    assert (rows["gate", 8].available, rows["gate", 8].short) == (0, 1)
    assert (rows["wall", 99].available, rows["wall", 99].short) == (0, 1)
    assert rows["connector", "right"].short == 1
    assert rows["connector", "straight"].short == 0


def test_inventory_usage_counts_a_flipped_connector_under_its_type_and_leaves_the_inventory_alone():
    inventory = Inventory(
        walls={10: 6}, gates={10: 1}, connectors={ConnectorType.RIGHT: 6}
    )
    con = rectangle_construction(10, 10)
    con.connectors[1] = Connector(ConnectorType.RIGHT, reflex=True)
    rows = {(r.kind, r.key): r for r in inventory_usage(con, inventory)}
    assert rows["connector", "right"].used == 4
    assert inventory.walls == {10: 6} and inventory.connectors == {
        ConnectorType.RIGHT: 6
    }


ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
