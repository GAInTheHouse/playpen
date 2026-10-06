"""Players must not start a build they can't finish. Run with
`uv run python -m tests.test_players`."""

from shapely.geometry import Polygon

from players.example_player import ExamplePlayer
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights

ROOM = Room(Polygon([(0, 0), (40, 0), (40, 30), (0, 30)]))
WEIGHTS = Weights(A=2.0, C=-3.0, G=100.0)


def _inventory(right: int | None) -> Inventory:
    connectors = {ConnectorType.STRAIGHT: 20, ConnectorType.DIAGONAL: 20}
    if right is not None:
        connectors[ConnectorType.RIGHT] = right
    return Inventory(
        walls={8: 6, 10: 6, 12: 6}, gates={8: 2, 10: 2}, connectors=connectors
    )


def _count_calls(player, *method_names):
    """Wrap the named methods so we can see whether they were ever invoked."""
    calls = {name: 0 for name in method_names}
    for name in method_names:
        original = getattr(player, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        setattr(player, name, wrapper)
    return calls


def test_example_player_does_not_attempt_without_enough_right_connectors():
    for right in (None, 0, 3):
        player = ExamplePlayer(ROOM, _inventory(right), WEIGHTS)
        calls = _count_calls(
            player, "_candidate_shapes", "_pieces_for", "_find_placement"
        )
        assert player.build_enclosure() is None
        assert sum(calls.values()) == 0, f"right={right}: {calls}"
        assert "right" in player.note and "not attempting" in player.note


def test_exactly_four_right_connectors_is_enough_to_try():
    example = ExamplePlayer(ROOM, _inventory(4), WEIGHTS)
    calls = _count_calls(example, "_candidate_shapes")
    assert example.build_enclosure() is not None
    assert calls["_candidate_shapes"] == 1
    assert example.note is None


def test_plentiful_straight_and_diagonal_connectors_dont_substitute():
    # these players only build right-angle rectangles, so other connector
    # types can't make up for missing RIGHT ones (see _inventory: 20 of each)
    player = ExamplePlayer(ROOM, _inventory(3), WEIGHTS)
    assert player.inventory.connectors[ConnectorType.STRAIGHT] == 20
    assert player.build_enclosure() is None


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
