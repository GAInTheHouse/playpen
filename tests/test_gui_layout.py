"""GUI layout checks: no text overlaps other text, nothing runs off the info
panel, and the room + enclosure stay inside the room view -- at several window
sizes, for every invalid example, a valid result, and a very long piece list.

Needs a display (it opens real Tk windows); skips itself without one.
Run with `uv run python -m tests.test_gui_layout`."""

import contextlib
import io
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from demos.invalid_examples import EXAMPLES
from players.player0 import Player
from players.registry import PLAYERS
from src.args import Args
from src.constants import BAD_COLOR, GHOST_COLOR
from src.enclosure import Construction, score_construction, validate_construction
from src.game import ENCLOSURE_TAG, TOOLTIP_TAG, Game
from src.pieces import Connector, ConnectorType, Piece, PieceType

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


class _NoSolutionPlayer(Player):
    def build_enclosure(self):
        return None


def _with_players(players: dict, body) -> None:
    """Register test-only players under these names while `body` runs."""
    PLAYERS.update(players)
    try:
        body()
    finally:
        for name in players:
            del PLAYERS[name]


def _make_game(player: str, scenario: str) -> Game:
    original = tk.Tk.mainloop
    tk.Tk.mainloop = lambda self: None  # build the window, but don't block
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return Game(
                Args(
                    gui=True,
                    player=player,
                    scenario_path=str(SCENARIOS / scenario),
                    seed=1,
                    difficulty="generous",
                    export_scenario=None,
                    debug=False,
                    sandbox=False,
                )
            )
    finally:
        tk.Tk.mainloop = original


def _click(game: Game) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        game.play()
    game.root.update()


def _resize(game: Game, width: int, height: int) -> None:
    game.root.geometry(f"{width}x{height}")
    game.root.update()
    game.root.update()


def _check(game: Game, label: str) -> None:
    game.root.update()

    # 1. no two pieces of panel text overlap
    texts = [
        (game.info.itemcget(i, "text"), game.info.bbox(i))
        for i in game.info.find_all()
        if game.info.type(i) == "text"
    ]
    for a in range(len(texts)):
        for b in range(a + 1, len(texts)):
            (ta, A), (tb, B) = texts[a], texts[b]
            overlap_x = min(A[2], B[2]) - max(A[0], B[0])
            overlap_y = min(A[3], B[3]) - max(A[1], B[1])
            assert not (overlap_x > 1 and overlap_y > 1), (
                f"{label}: {ta[:30]!r} overlaps {tb[:30]!r}"
            )

    # 2. panel text stays inside the panel
    panel_w, panel_h = game.info.winfo_width(), game.info.winfo_height()
    for text, (_, _, x1, y1) in texts:
        assert x1 <= panel_w, (
            f"{label}: {text[:30]!r} runs off the right ({x1}>{panel_w})"
        )
        assert y1 <= panel_h, (
            f"{label}: {text[:30]!r} runs off the bottom ({y1}>{panel_h})"
        )

    # 3. the room view: everything drawn (room and enclosure) is inside it
    view_w, view_h = game.canvas.winfo_width(), game.canvas.winfo_height()
    for item in game.canvas.find_all():
        x0, y0, x1, y1 = game.canvas.bbox(item)
        assert x0 >= 0 and y0 >= 0 and x1 <= view_w and y1 <= view_h, (
            f"{label}: canvas item outside the {view_w}x{view_h} view: "
            f"{(x0, y0, x1, y1)}"
        )


def _with_game(player, scenario, body):
    try:
        game = _make_game(player, scenario)
    except tk.TclError as e:
        print(f"  (skipped: no display -- {e})")
        return
    try:
        body(game)
    finally:
        game.root.destroy()


def test_every_invalid_example_lays_out_cleanly():
    def body(game):
        _check(game, "before any click")
        for n in range(len(EXAMPLES)):
            _click(game)
            _check(game, f"example {n + 1}")

    _with_game("i", "invalid_examples.json", body)


def test_layout_holds_at_several_window_sizes():
    def body(game):
        for width, height in [(900, 640), (1100, 700), (1300, 760)]:
            _resize(game, width, height)
            for n in range(len(EXAMPLES)):
                _click(game)
                _check(game, f"{width}x{height}, example {n + 1}")

    _with_game("i", "invalid_examples.json", body)


def test_enclosure_is_redrawn_when_the_window_is_resized():
    def body(game):
        _click(game)
        _click(game)  # example 2: has drawing on the canvas
        before = len(game.canvas.find_withtag(ENCLOSURE_TAG))
        assert before > 0
        _resize(game, 1000, 620)
        after = len(game.canvas.find_withtag(ENCLOSURE_TAG))
        assert after == before, "resizing lost or duplicated the enclosure drawing"

    _with_game("i", "invalid_examples.json", body)


def test_a_valid_result_lays_out_cleanly():
    def body(game):
        _click(game)
        assert game.result is not None
        _check(game, "player 1 on simple_rectangle")

    _with_game("e", "simple_rectangle.json", body)


def test_a_very_long_piece_list_still_fits():
    def body(game):
        # 12 pieces (whether the loop is geometrically valid doesn't matter
        # here -- this is only about the panel coping with a long list)
        pieces = [Piece(PieceType.WALL, 6) for _ in range(11)]
        pieces.append(Piece(PieceType.GATE, 6))
        con = Construction(
            start=(3, 3),
            start_heading=0.0,
            pieces=pieces,
            connectors=[Connector(ConnectorType.RIGHT) for _ in range(12)],
        )
        game.construction = con
        game.result = validate_construction(
            con, game.scenario.room, game.scenario.inventory
        )
        game.score = score_construction(game.result, game.scenario.weights)
        game.draw_result()
        _check(game, "12-piece result")

    _with_game("e", "invalid_examples.json", body)


def test_wrong_connector_count_is_drawn_with_the_problem_marked():
    def body(game):
        _click(game)  # i1: 4 walls, only 3 connectors
        items = game.canvas.find_withtag(ENCLOSURE_TAG)
        assert items, "the attempted placement should be drawn, not left blank"
        labels = [
            game.canvas.itemcget(i, "text")
            for i in items
            if game.canvas.type(i) == "text"
        ]
        assert "no connector" in labels  # (the other text is the hover hint)
        # exactly one ghost wall (the one with no connector), in the ghost color
        ghosts = [
            i
            for i in items
            if game.canvas.type(i) == "line"
            and game.canvas.itemcget(i, "fill") == GHOST_COLOR
        ]
        assert len(ghosts) == 1
        _check(game, "i1")

    _with_game("i1", "invalid_examples.json", body)


def _hover(game: Game, x: float, y: float) -> list[str]:
    """Move the mouse to (x, y) over the room view; return the tooltip's text."""
    game._on_hover(SimpleNamespace(x=x, y=y))
    return [
        game.canvas.itemcget(i, "text")
        for i in game.canvas.find_withtag(TOOLTIP_TAG)
        if game.canvas.type(i) == "text"
    ]


def test_hovering_a_joint_shows_its_coordinates_and_connector():
    def body(game):
        _click(game)
        _click(game)  # example 2: a 10x8 loop from (6,3) that ends 3 short
        joints = {
            "vertex 0 (start): (6.00, 3.00)": "right connector, 90",
            "vertex 1: (16.00, 3.00)": "right connector, 90",
            "vertex 2: (16.00, 11.00)": "right connector, 90",
            "vertex 3: (6.00, 11.00)": "right connector, 90",
        }
        assert len(game._vertex_hits) == 5  # 4 vertices + where the last wall ends
        found = set()
        for px, py, _ in game._vertex_hits:
            (text,) = _hover(game, px, py)  # hover exactly on the joint
            found.add(text.split("\n")[0])
            for head, connector in joints.items():
                if text.startswith(head):
                    assert connector in text, text
        assert set(joints) <= found
        assert any(t.startswith("end of the last wall: (6.00, 6.00)") for t in found)

    _with_game("i", "invalid_examples.json", body)


def test_tooltip_appears_only_near_a_joint_and_goes_away():
    def body(game):
        _click(game)
        px, py, _ = game._vertex_hits[0]
        assert _hover(game, px + 5, py - 5)  # close enough
        assert _hover(game, px + 200, py + 200) == []  # far from every joint
        assert game.canvas.find_withtag(TOOLTIP_TAG) == ()
        assert _hover(game, px, py)
        game._hide_tooltip()  # what <Leave> does
        assert game.canvas.find_withtag(TOOLTIP_TAG) == ()
        assert game.canvas.bind("<Motion>") and game.canvas.bind("<Leave>")

    _with_game("i", "invalid_examples.json", body)


def test_tooltip_always_stays_inside_the_room_view():
    def body(game):
        for width, height in [(900, 660), (1300, 760)]:
            _resize(game, width, height)
            for n in range(len(EXAMPLES)):
                _click(game)
                view_w, view_h = game.canvas.winfo_width(), game.canvas.winfo_height()
                for px, py, _ in game._vertex_hits:
                    _hover(game, px, py)
                    for item in game.canvas.find_withtag(TOOLTIP_TAG):
                        x0, y0, x1, y1 = game.canvas.bbox(item)
                        assert x0 >= 0 and y0 >= 0 and x1 <= view_w and y1 <= view_h, (
                            f"{width}x{height}, example {n + 1}: tooltip at "
                            f"{(x0, y0, x1, y1)} outside {view_w}x{view_h}"
                        )

    _with_game("i", "invalid_examples.json", body)


def _panel_texts(game: Game) -> list[str]:
    return [
        game.info.itemcget(i, "text")
        for i in game.info.find_all()
        if game.info.type(i) == "text"
    ]


def test_valid_result_shows_the_score_calculation():
    def body(game):
        _click(game)
        texts = _panel_texts(game)
        weights = game.scenario.weights
        area, perimeter = game.result.area, game.result.perimeter
        assert "SCORE CALCULATION" in texts
        assert "+1000.0" in texts  # baseline
        assert f"+{weights.A * area:.1f}" in texts  # A * area
        assert f"area: {weights.A:g} \u00d7 {area:.1f}" in texts
        assert f"perimeter: \u2212{-weights.C:g} \u00d7 {perimeter:.1f}" in texts
        total = [t for t in texts if t == f"{game.score:.1f}"]
        assert total, f"the total {game.score:.1f} should appear as the SCORE row"
        _check(game, "score calculation, valid")

    _with_game("e", "simple_rectangle.json", body)


def test_invalid_result_explains_the_penalty():
    def body(game):
        _click(game)  # i1
        texts = _panel_texts(game)
        assert "SCORE CALCULATION" in texts
        assert "invalid enclosure" in texts
        assert "\u22121000.0" in texts
        assert not any(t.startswith("baseline") for t in texts)  # no other terms
        _check(game, "score calculation, invalid")

    _with_game("i1", "invalid_examples.json", body)


def test_no_solution_explains_why_zero_is_safe():
    def body(game):
        _click(game)
        texts = _panel_texts(game)
        assert any("scores 0" in t and "\u22121000" in t for t in texts)
        _check(game, "no solution")

    _with_players(
        {"_no_solution": _NoSolutionPlayer},
        lambda: _with_game("_no_solution", "simple_rectangle.json", body),
    )


def _translucent_bands(game: Game):
    """The wide, faint lines that stand for pieces the inventory can't supply."""
    return [
        i
        for i in game.canvas.find_withtag(ENCLOSURE_TAG)
        if game.canvas.type(i) == "line"
        and float(game.canvas.itemcget(i, "width")) == 9
    ]


def _canvas_labels(game: Game) -> list[str]:
    return [
        game.canvas.itemcget(i, "text")
        for i in game.canvas.find_withtag(ENCLOSURE_TAG)
        if game.canvas.type(i) == "text"
    ]


def test_a_piece_missing_from_the_inventory_is_drawn_translucent_not_placed():
    def body(game):
        _click(game)  # i4: two 13-unit walls wanted, one in stock
        bands = _translucent_bands(game)
        assert len(bands) == 1, "exactly the second 13-unit wall should be a ghost"
        # translucent = a pale mix of the alert color into the room color, not the solid color
        assert game.canvas.itemcget(bands[0], "fill").lower() != BAD_COLOR.lower()
        assert "wall 13: none left in inventory" in _canvas_labels(game)
        texts = _panel_texts(game)
        assert "WALLS PLACED (3 OF 4)" in texts
        assert "wall: 13 units \u2014 not in inventory" in texts
        _check(game, "i4")

    _with_game("i4", "invalid_examples.json", body)


def test_nothing_is_ghosted_when_the_inventory_has_everything():
    def body(game):
        for n in range(len(EXAMPLES)):
            _click(game)
            if game.player.note.startswith("Example 4 "):
                continue  # the one example that is short of a piece
            assert _translucent_bands(game) == [], game.player.note
            assert not any("none left" in t for t in _canvas_labels(game))

    _with_game(
        "i", "invalid_examples.json", body
    )  # includes i3: its connector is in stock


def test_a_connector_missing_from_the_inventory_is_ghosted_too():
    def body(game):
        # i3 puts a diagonal connector at the closing joint; take them all away
        game.scenario.inventory.connectors[ConnectorType.DIAGONAL] = 0
        _click(game)
        assert "no diagonal connector left" in _canvas_labels(game)
        assert not any(
            "none left" in t for t in _canvas_labels(game)
        )  # no wall ghosted
        # ...and hovering that joint says so
        px, py, _ = game._vertex_hits[0]
        (tip,) = _hover(game, px, py)
        assert "diagonal connector" in tip and "not in inventory" in tip
        _check(game, "i3 without diagonal connectors")

    _with_game("i3", "invalid_examples.json", body)


def test_ghosting_survives_a_window_resize():
    def body(game):
        _click(game)  # i4
        _resize(game, 1000, 700)
        assert len(_translucent_bands(game)) == 1
        assert "wall 13: none left in inventory" in _canvas_labels(game)

    _with_game("i4", "invalid_examples.json", body)


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
