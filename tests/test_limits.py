"""The CPU time limit: the CpuLimit mechanism itself, and how Game uses it.
Run with `uv run python -m tests.test_limits`."""

import contextlib
import io
import signal
import time
import tkinter as tk

from players.player0 import Player
from src import limits
from src.args import Args
from src.game import Game
from src.limits import CpuLimit, PlayerTimeout
from tests.test_gui_layout import (
    _check,
    _click,
    _make_game,
    _panel_texts,
    _with_players,
)


def _burn(seconds: float) -> None:
    """Use `seconds` of CPU time."""
    end = time.process_time() + seconds
    while time.process_time() < end:
        pass


# ------------------------------------------------------------ CpuLimit itself


def test_code_within_the_limit_is_left_alone_and_its_cpu_time_recorded():
    with CpuLimit(5) as cpu:
        _burn(0.1)
    assert 0.1 <= cpu.used < 1.0


def test_a_player_that_never_stops_is_interrupted_at_the_limit():
    started = time.perf_counter()
    cpu = CpuLimit(0.3)
    try:
        with cpu:
            while True:
                pass
    except PlayerTimeout as timeout:
        assert 0.3 <= timeout.used < 1.5
        assert timeout.limit == 0.3
    else:
        raise AssertionError("an endless loop was not interrupted")
    assert time.perf_counter() - started < 3.0
    assert cpu.used >= 0.3  # recorded even though the block was interrupted


def test_a_players_except_exception_cannot_swallow_the_timeout():
    try:
        with CpuLimit(0.3):
            while True:
                try:
                    sum(i * i for i in range(1000))
                except Exception:  # noqa: BLE001, S110 -- the hostile player under test
                    pass
    except PlayerTimeout:
        return
    raise AssertionError("the timeout was swallowed")


def test_a_player_that_swallows_the_interrupt_is_still_caught_when_it_returns():
    try:
        with CpuLimit(0.3):
            try:
                while True:
                    pass
            except BaseException:  # noqa: BLE001, S110 -- the hostile player under test
                pass  # swallows the interrupt, then carries on and returns
    except PlayerTimeout as timeout:
        assert timeout.used > 0.3
        return
    raise AssertionError("an over-limit player was let through")


def test_waiting_is_not_cpu_time():
    started = time.perf_counter()
    with CpuLimit(0.2) as cpu:
        time.sleep(0.7)  # 3.5 times the limit on the clock, but no CPU
    assert time.perf_counter() - started >= 0.7
    assert cpu.used < 0.2


def test_the_limit_is_fully_disarmed_afterwards():
    before = signal.getsignal(signal.SIGPROF)
    with CpuLimit(0.2):
        pass
    assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGPROF) == before
    _burn(0.5)  # longer than the limit was: nothing is left to fire


def test_a_block_that_raises_something_else_still_records_cpu_time_and_disarms():
    cpu = CpuLimit(5)
    try:
        with cpu:
            _burn(0.05)
            raise ValueError("boom")
    except ValueError:
        pass
    assert cpu.used >= 0.05
    assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0)


def test_where_nothing_can_interrupt_an_overrun_is_still_reported_on_return():
    original = limits.CAN_INTERRUPT
    limits.CAN_INTERRUPT = False  # pretend to be a platform without SIGPROF
    finished = []
    try:
        with CpuLimit(0.1):
            _burn(0.3)
            finished.append(True)  # nothing stopped it...
    except PlayerTimeout as timeout:
        assert finished == [True]
        assert timeout.used >= 0.3  # ...but the overrun is reported afterwards
    else:
        raise AssertionError("an overrun went unreported")
    finally:
        limits.CAN_INTERRUPT = original


def test_a_huge_limit_is_accepted():
    with CpuLimit(float("inf")):
        pass


# ------------------------------------------------------------- Game uses it


class _BusyPlayer(Player):
    def build_enclosure(self):
        while True:
            pass


class _SlowToConstructPlayer(Player):
    def __init__(self, room, inventory, weights):
        super().__init__(room, inventory, weights)
        while True:
            pass

    def build_enclosure(self):
        return None


class _BusyOncePlayer(Player):
    """Hangs on its first run, then behaves."""

    runs = 0

    def build_enclosure(self):
        _BusyOncePlayer.runs += 1
        if _BusyOncePlayer.runs == 1:
            while True:
                pass


def _headless_game(player: str, cpu_limit: float) -> tuple[Game, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        game = Game(
            Args(
                gui=False,
                player=player,
                scenario_path="scenarios/simple_rectangle.json",
                seed=1,
                difficulty="generous",
                export_scenario=None,
                debug=False,
                sandbox=False,
                cpu_limit=cpu_limit,
            )
        )
    return game, out.getvalue()


def test_game_stops_a_player_that_runs_too_long_and_scores_zero():
    def body():
        started = time.perf_counter()
        game, out = _headless_game("_busy", 0.5)
        assert time.perf_counter() - started < 5.0
        assert game.construction is None and game.result is None
        assert game.score == 0.0
        assert game.timeout_message.startswith("Time limit exceeded")
        assert "No solution. SCORE: 0" in out
        assert "player CPU time: 0.5" in out and "(limit 0.5s)" in out

    _with_players({"_busy": _BusyPlayer}, body)


def test_a_slow_constructor_counts_toward_the_limit_too():
    def body():
        game, out = _headless_game("_slow_init", 0.5)
        assert game.player is None  # never finished being built
        assert game.timeout_message.startswith("Time limit exceeded")
        assert "SCORE: 0" in out

    _with_players({"_slow_init": _SlowToConstructPlayer}, body)


def test_a_normal_run_reports_its_cpu_time_and_is_not_flagged():
    game, out = _headless_game("e", 300)
    assert game.timeout_message is None
    assert game.cpu_used is not None and game.cpu_used < 5
    assert "player CPU time:" in out and "(limit 300s)" in out
    assert "valid: True" in out and "SCORE: 1072.0" in out


def test_gui_shows_the_timeout_and_the_next_click_gets_a_fresh_budget():
    def body():
        _BusyOncePlayer.runs = 0
        try:
            game = _make_game("_busy_once", "simple_rectangle.json")
        except tk.TclError as e:  # no display
            print(f"  (skipped: {e})")
            return
        try:
            game.args.cpu_limit = 0.5
            _click(game)  # runs forever -> stopped
            texts = _panel_texts(game)
            assert any(t.startswith("Time limit exceeded") for t in texts)
            assert any(t.startswith("player CPU time: 0.5") for t in texts)
            assert not any("safer than" in t for t in texts)
            _check(game, "after a timeout")

            _click(game)  # fresh budget; this run is fine and returns no solution
            texts = _panel_texts(game)
            assert not any(t.startswith("Time limit exceeded") for t in texts)
            assert "No solution generated -- score: 0" in texts
            assert game.timeout_message is None
            _check(game, "after recovering")
        finally:
            game.root.destroy()

    _with_players({"_busy_once": _BusyOncePlayer}, body)


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
