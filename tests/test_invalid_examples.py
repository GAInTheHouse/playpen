"""Checks that the invalid-examples demo player tells the truth: each example
really fails for the reason it claims, and the geometric claims in the
labels hold. Run with `uv run python -m tests.test_invalid_examples`."""

from pathlib import Path

from shapely.geometry import Point, Polygon

from demos.invalid_examples import EXAMPLES, InvalidExamplesPlayer
from players.registry import PLAYERS
from src.enclosure import build_geometry, validate_construction, walk_points
from src.scenario import read_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
SCENARIO = read_scenario(str(SCENARIOS / "invalid_examples.json"))


def _example(fragment: str):
    matches = [e for e in EXAMPLES if fragment in e.title]
    assert len(matches) == 1, f"{fragment!r} matched {len(matches)} examples"
    return matches[0]


def test_every_example_fails_for_its_stated_reason():
    for ex in EXAMPLES:
        result = validate_construction(ex.build(), SCENARIO.room, SCENARIO.inventory)
        assert not result.valid, ex.title
        assert ex.expect in result.reason, f"{ex.title}: got {result.reason!r}"


def test_player_cycles_through_all_examples_and_wraps():
    player = InvalidExamplesPlayer(
        SCENARIO.room.copy(), SCENARIO.inventory.copy(), SCENARIO.weights
    )
    notes = []
    for _ in range(len(EXAMPLES) + 1):
        assert player.build_enclosure() is not None
        notes.append(player.note)
    assert len(set(notes[: len(EXAMPLES)])) == len(EXAMPLES)  # all distinct
    assert notes[-1] == notes[0]  # wrapped around


def test_entirely_outside_touches_none_of_the_room():
    con = _example("Entirely outside").build()
    shape = Polygon(build_geometry(con)[0])
    assert not shape.intersects(SCENARIO.room.polygon)


def test_sticks_out_has_vertices_both_inside_and_outside():
    con = _example("Sticks out").build()
    inside = [
        SCENARIO.room.polygon.buffer(1e-4).covers(Point(v))
        for v in build_geometry(con)[0]
    ]
    assert any(inside) and not all(inside)


def test_notch_example_has_every_corner_inside_but_is_not_contained():
    con = _example("Every corner is inside").build()
    room = SCENARIO.room.polygon.buffer(1e-4)
    corners = build_geometry(con)[0]
    assert all(room.covers(Point(v)) for v in corners)  # every corner inside...
    assert not SCENARIO.room.contains_enclosure(Polygon(corners))  # ...still rejected


def test_non_closing_example_ends_away_from_where_it_started():
    points = walk_points(_example("Doesn't close").build())
    assert len(points) == 5  # 4 pieces -> 5 points
    dx, dy = points[-1][0] - points[0][0], points[-1][1] - points[0][1]
    assert abs((dx * dx + dy * dy) ** 0.5 - 3.0) < 1e-9


def test_examples_that_dont_apply_are_skipped_not_mislabeled():
    # a plain rectangle room has no notch, and an inventory without the odd
    # pieces: the notch/13-wall examples can't reproduce there, so the player
    # must skip them rather than show them under a wrong label
    other = read_scenario(str(SCENARIOS / "simple_rectangle.json"))
    player = InvalidExamplesPlayer(
        other.room.copy(), other.inventory.copy(), other.weights
    )
    shown = set()
    for _ in range(len(EXAMPLES)):
        con = player.build_enclosure()
        if con is None:
            continue
        result = validate_construction(con, other.room, other.inventory)
        assert not result.valid
        shown.add(player.note.split(":")[0])
    assert "Example 9 of 10" not in shown  # cutting across a notch needs the L


def test_each_pinned_player_always_shows_its_own_example():
    for n in range(1, len(EXAMPLES) + 1):
        cls = PLAYERS[f"i{n}"]
        player = cls(SCENARIO.room.copy(), SCENARIO.inventory.copy(), SCENARIO.weights)
        for _ in range(3):  # pinned: clicking again shows the same one
            con = player.build_enclosure()
            assert con is not None
            assert player.note.startswith(f"Example {n} of {len(EXAMPLES)}:")
            result = validate_construction(con, SCENARIO.room, SCENARIO.inventory)
            assert not result.valid and EXAMPLES[n - 1].expect in result.reason


def test_a_pinned_example_that_doesnt_apply_says_so_instead_of_faking_it():
    other = read_scenario(str(SCENARIOS / "simple_rectangle.json"))
    player = PLAYERS["i9"](other.room.copy(), other.inventory.copy(), other.weights)
    assert player.build_enclosure() is None  # the notch example needs the L
    assert "doesn't apply" in player.note


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
