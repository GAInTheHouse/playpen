"""GUI: the scrollable inventory box (every item listed, the ones the enclosure
uses highlighted), the red highlight on a negative score, and the panel scrolling
on a screen too short for it.

Needs a display (it opens real Tk windows); skips itself without one.
Run with `uv run python -m tests.test_gui_inventory`."""

import contextlib
import re
import tkinter as tk
from types import SimpleNamespace

from src.args import Args
from src.constants import BAD_COLOR, GOOD_COLOR, NEGATIVE_BG
from src.enclosure import Construction, score_construction, validate_construction
from src.game import RESULT_TAG, Game
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.weights import Weights
from tests.test_gui_layout import (
    SCENARIOS,
    _check,
    _click,
    _make_game,
    _NoSolutionPlayer,
    _with_game,
    _with_players,
)

SECTIONS = ("WALLS", "GATES", "CONNECTORS")


def _box(game: Game) -> tk.Text:
    return game.inventory_box


def _sections(game: Game) -> dict[str, list[str]]:
    """The box's text: {"WALLS": [row, ...], "GATES": [...], "CONNECTORS": [...]}."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in _box(game).get("1.0", "end").splitlines():
        if line.strip() in SECTIONS:
            current = line.strip()
            sections[current] = []
        elif line.strip() and current:
            sections[current].append(line.strip())
    return sections


def _tagged_rows(game: Game, tag: str) -> list[str]:
    box = _box(game)
    ranges = box.tag_ranges(tag)
    return [box.get(str(a), str(b)).strip() for a, b in zip(ranges[::2], ranges[1::2])]


def _row(section: list[str], name: str) -> str:
    matches = [row for row in section if row.startswith(name + " ") or row == name]
    assert len(matches) == 1, f"expected one row for {name!r} in {section}"
    return matches[0]


# ----------------------------------------------------------- the inventory box


def test_every_inventory_item_is_listed_before_anything_is_built():
    def body(game):
        inventory = game.scenario.inventory
        sections = _sections(game)
        assert set(sections) == set(SECTIONS)
        for length, count in inventory.walls.items():
            assert re.match(
                rf"x{count}\b",
                _row(sections["WALLS"], f"{length} units").split(maxsplit=2)[2],
            )
        for length, count in inventory.gates.items():
            assert f"x{count}" in _row(sections["GATES"], f"{length} units")
        for connector_type in ConnectorType:
            count = inventory.connectors.get(connector_type, 0)
            assert f"x{count}" in _row(sections["CONNECTORS"], connector_type.value)
        assert _tagged_rows(game, "chosen") == [] and _tagged_rows(game, "short") == []

    _with_game("e", "simple_rectangle.json", body)


def test_the_rows_the_enclosure_uses_are_highlighted_and_only_those():
    def body(game):
        _click(game)
        con = game.construction
        assert con is not None
        wanted_rows = set()
        for piece in con.pieces:
            kind = "GATES" if piece.is_gate else "WALLS"
            wanted_rows.add((kind, f"{piece.length} units"))
        for connector in con.connectors:
            wanted_rows.add(("CONNECTORS", connector.connector_type.value))

        sections = _sections(game)
        highlighted = set(_tagged_rows(game, "chosen"))
        for kind, rows in sections.items():
            for row in rows:
                name = (
                    " ".join(row.split()[:2])
                    if kind != "CONNECTORS"
                    else row.split()[0]
                )
                assert (row in highlighted) == ((kind, name) in wanted_rows), (
                    kind,
                    row,
                )

        # and each says how many of it are used
        for piece in {(p.is_gate, p.length) for p in con.pieces}:
            is_gate, length = piece
            n = sum(1 for p in con.pieces if (p.is_gate, p.length) == piece)
            row = _row(sections["GATES" if is_gate else "WALLS"], f"{length} units")
            assert f"using {n}" in row
        _check(game, "after highlighting")

    _with_game("e", "simple_rectangle.json", body)


def test_the_highlight_goes_away_when_there_is_no_enclosure_or_the_run_timed_out():
    def body(game):
        _click(game)
        assert _tagged_rows(game, "chosen")
        game.construction = None
        game.draw_result()
        assert _tagged_rows(game, "chosen") == [] and "using" not in _box(game).get(
            "1.0", "end"
        )

        _click(game)
        assert _tagged_rows(game, "chosen")
        game.timeout_message = "Time limit exceeded"
        game.draw_result()
        assert _tagged_rows(game, "chosen") == []

    _with_game("e", "simple_rectangle.json", body)


def test_an_item_the_enclosure_wants_more_of_than_the_inventory_has_is_red():
    def body(game):
        _click(game)  # i4: two 13-unit walls wanted, only one in stock
        short = _tagged_rows(game, "short")
        assert len(short) == 1
        assert (
            short[0].startswith("13 units")
            and "(1 short)" in short[0]
            and "using 2" in short[0]
        )
        assert "x1" in short[0]

    _with_game("i4", "invalid_examples.json", body)


def test_an_item_the_inventory_lacks_entirely_still_gets_a_red_row():
    def body(game):
        con = Construction(
            start=(5, 5),
            start_heading=0.0,
            pieces=[
                Piece(PieceType.GATE, 10),
                Piece(PieceType.WALL, 99),  # nobody has a 99-unit wall
                Piece(PieceType.WALL, 10),
                Piece(PieceType.WALL, 99),
            ],
            connectors=[Connector(ConnectorType.RIGHT)] * 4,
        )
        game.construction = con
        game.result = validate_construction(
            con, game.scenario.room, game.scenario.inventory
        )
        game.score = score_construction(game.result, game.scenario.weights)
        game.draw_result()
        short = _tagged_rows(game, "short")
        assert any(
            r.startswith("99 units") and "x0" in r and "(2 short)" in r for r in short
        ), short

    _with_game("e", "simple_rectangle.json", body)


def test_a_long_inventory_scrolls_and_the_first_highlighted_row_is_brought_into_view():
    def body(game):
        game.scenario.inventory.walls = {
            5 + i: 2 for i in range(40)
        }  # 40 rows of walls
        game._render_inventory(None)
        game.root.update()
        box = _box(game)
        first, last = box.yview()
        assert first == 0.0 and last < 0.5, (
            "the box should be showing only the top of a long list"
        )

        # the scrollbar is wired to the box
        scrollbar = next(
            w for w in box.master.winfo_children() if isinstance(w, tk.Scrollbar)
        )
        assert float(scrollbar.get()[0]) == 0.0
        box.yview_moveto(0.3)  # scroll the box...
        game.root.update()
        assert float(scrollbar.get()[0]) > 0.2  # ...and the scrollbar follows
        box.yview_moveto(0.0)
        game.root.update()

        # an enclosure whose walls are deep in the list (41 and 44 units) scrolls them into view
        con = Construction(
            start=(5, 5),
            start_heading=0.0,
            pieces=[
                Piece(PieceType.GATE, 44),
                Piece(PieceType.WALL, 41),
                Piece(PieceType.WALL, 44),
                Piece(PieceType.WALL, 41),
            ],
            connectors=[Connector(ConnectorType.RIGHT)] * 4,
        )
        game._render_inventory(con)
        game.root.update()
        assert box.yview()[0] > 0.5, "should have scrolled down to the highlighted rows"
        chosen_start = str(box.tag_ranges("chosen")[0])
        assert box.bbox(chosen_start) is not None, "the highlighted row is not visible"

        # and it comes back to the top when nothing is highlighted
        game._render_inventory(None)
        game.root.update()
        assert box.yview()[0] == 0.0

    _with_game("e", "simple_rectangle.json", body)


def test_the_box_survives_a_resize_and_stays_inside_the_panel():
    def body(game):
        for width, height in [(900, 750), (1300, 760)]:
            game.root.geometry(f"{width}x{height}")
            game.root.update()
            game.root.update()
            _click(game)
            _check(game, f"inventory box at {width}x{height}")
            assert _box(game).winfo_height() > 40

    _with_game("e", "simple_rectangle.json", body)


def test_sandbox_mode_shows_the_inventory_but_highlights_nothing():
    original = tk.Tk.mainloop
    tk.Tk.mainloop = lambda self: None
    try:
        try:
            game = Game(
                Args(
                    gui=True,
                    player=None,
                    scenario_path=str(SCENARIOS / "simple_rectangle.json"),
                    seed=1,
                    difficulty="generous",
                    export_scenario=None,
                    debug=False,
                    sandbox=True,
                )
            )
        except tk.TclError as e:
            print(f"  (skipped: no display -- {e})")
            return
    finally:
        tk.Tk.mainloop = original
    try:
        game.root.update()
        assert set(_sections(game)) == set(SECTIONS)
        assert _tagged_rows(game, "chosen") == [] and _tagged_rows(game, "short") == []
        _check(game, "sandbox")
    finally:
        game.root.destroy()


# ------------------------------------------------------ the red negative score


def _score_text_and_bands(game: Game):
    """The SCORE row's value text, and any red band drawn behind the score."""
    shown = ("−" if game.score < 0 else "") + f"{abs(game.score):.1f}"
    values = [
        i
        for i in game.info.find_withtag(RESULT_TAG)
        if game.info.type(i) == "text" and game.info.itemcget(i, "text") == shown
    ]
    value = max(
        values, key=lambda i: game.info.bbox(i)[1]
    )  # the last one: the SCORE row
    bands = [
        i
        for i in game.info.find_withtag(RESULT_TAG)
        if game.info.type(i) == "rectangle"
        and game.info.itemcget(i, "fill") == NEGATIVE_BG
    ]
    return value, bands


def test_a_valid_but_negative_score_is_highlighted_in_red():
    def body(game):
        game.scenario.weights = Weights(
            A=0.1, C=-100.0, G=0.0
        )  # a long perimeter now costs a lot
        _click(game)
        assert game.result.valid and game.score < 0
        value, bands = _score_text_and_bands(game)
        assert game.info.itemcget(value, "fill") == BAD_COLOR
        assert len(bands) == 1
        bx0, by0, bx1, by1 = game.info.bbox(bands[0])
        vx0, vy0, vx1, vy1 = game.info.bbox(value)
        assert bx0 <= vx0 and by0 <= vy0 and bx1 >= vx1 and by1 >= vy1, (
            "the band should surround the score"
        )
        order = list(game.info.find_all())
        assert order.index(bands[0]) < order.index(value), (
            "the band must sit behind the score"
        )
        _check(game, "negative score")

    _with_game("e", "simple_rectangle.json", body)


def test_an_invalid_enclosures_minus_1000_is_highlighted_in_red_too():
    def body(game):
        _click(game)
        assert game.score == -1000.0
        value, bands = _score_text_and_bands(game)
        assert game.info.itemcget(value, "fill") == BAD_COLOR and len(bands) == 1

    _with_game("i5", "invalid_examples.json", body)


def test_a_positive_score_is_green_with_no_red_band():
    def body(game):
        _click(game)
        assert game.score > 0
        value, bands = _score_text_and_bands(game)
        assert game.info.itemcget(value, "fill") == GOOD_COLOR
        assert bands == []

    _with_game("e", "simple_rectangle.json", body)


def test_no_solution_has_no_score_row_and_no_red_band():
    def body():
        game = _make_game("_no_solution", "simple_rectangle.json")
        try:
            _click(game)
            bands = [
                i
                for i in game.info.find_withtag(RESULT_TAG)
                if game.info.type(i) == "rectangle"
            ]
            assert bands == []
            assert "SCORE" not in [
                game.info.itemcget(i, "text")
                for i in game.info.find_all()
                if game.info.type(i) == "text"
            ]
        except tk.TclError as e:
            print(f"  (skipped: no display -- {e})")
        finally:
            game.root.destroy()

    _with_players({"_no_solution": _NoSolutionPlayer}, body)


def test_the_red_band_goes_away_on_the_next_positive_run():
    def body(game):
        game.scenario.weights = Weights(A=0.1, C=-100.0, G=0.0)
        _click(game)
        assert _score_text_and_bands(game)[1]
        game.scenario.weights = Weights(A=2.0, C=-3.0, G=100.0)
        game.player.weights = game.scenario.weights
        _click(game)
        assert game.score > 0 and _score_text_and_bands(game)[1] == []

    _with_game("e", "simple_rectangle.json", body)


# ---------------------------------------- a screen too short for the whole panel


@contextlib.contextmanager
def _screen_height(pixels: int):
    original = tk.Misc.winfo_screenheight
    tk.Misc.winfo_screenheight = lambda self: pixels
    try:
        yield
    finally:
        tk.Misc.winfo_screenheight = original


def test_on_a_short_screen_the_panel_scrolls_instead_of_cutting_content_off():
    try:
        with _screen_height(700):  # a small laptop: the window gets ~560px, not 750+
            game = _make_game("i", "invalid_examples.json")
    except tk.TclError as e:
        print(f"  (skipped: no display -- {e})")
        return
    try:
        game.root.update()
        assert game.info.winfo_height() < 600
        for _ in range(10):
            _click(game)
        content_bottom = max(game.info.bbox(i)[3] for i in game.info.find_all())
        assert content_bottom > game.info.winfo_height(), (
            "this window should be too short"
        )

        # the panel can be scrolled to see everything, with the mouse wheel
        assert game.info.yview() != (0.0, 1.0)
        region_bottom = float(game.info.cget("scrollregion").split()[3])
        assert region_bottom >= content_bottom
        game._on_panel_wheel(SimpleNamespace(delta=-120))
        first, _ = game.info.yview()
        assert first > 0.0
        for _ in range(30):
            game._on_panel_wheel(SimpleNamespace(delta=-120))
        assert game.info.yview()[1] == 1.0  # reached the bottom: the score is reachable
        for _ in range(60):
            game._on_panel_wheel(SimpleNamespace(delta=120))
        assert game.info.yview()[0] == 0.0
    finally:
        game.root.destroy()


def test_when_everything_fits_the_panel_does_not_scroll():
    def body(game):
        for _ in range(10):
            _click(game)
            assert game.info.yview() == (0.0, 1.0)
        game._on_panel_wheel(SimpleNamespace(delta=-120))
        assert game.info.yview() == (0.0, 1.0)  # the wheel does nothing

    _with_game("i", "invalid_examples.json", body)


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
