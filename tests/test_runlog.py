"""Run logs: everything a run prints goes to a log file as well as the terminal.
Run with `uv run python -m tests.test_runlog`."""

import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

from players.player0 import Player
from players.registry import PLAYERS
from src.args import Args
from src.game import Game
from src.runlog import RunLog

PROJECT = Path(__file__).resolve().parent.parent


def _logs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.log"))


def test_everything_printed_is_in_the_log_and_the_terminal_is_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        terminal, errors = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(terminal),
            contextlib.redirect_stderr(errors),
            RunLog(tmp, "label", ["a header line"]) as log,
        ):
            print("to stdout")
            print("to stderr", file=sys.stderr)
        text = log.path.read_text()
        assert "to stdout\n" in text and "to stderr\n" in text
        assert "# a header line" in text and "# started:" in text
        assert "# finished normally; wall time" in text and "process CPU time" in text
        # the terminal got exactly what was printed: no header, no footer
        assert terminal.getvalue() == "to stdout\n"
        assert errors.getvalue() == "to stderr\n"


def test_streams_are_restored_afterwards():
    with tempfile.TemporaryDirectory() as tmp:
        before = sys.stdout, sys.stderr
        with RunLog(tmp, "label", []):
            assert sys.stdout is not before[0]
        assert (sys.stdout, sys.stderr) == before


def test_lines_reach_the_file_as_they_are_printed():
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
        RunLog(tmp, "label", []) as log,
    ):
        print("still running")
        # the run hasn't ended and the file hasn't been closed, yet the line is there:
        # so a run that is killed part way still leaves its log behind
        assert "still running" in log.path.read_text()


def test_a_crash_puts_the_traceback_in_the_log_but_not_twice_on_the_terminal():
    with tempfile.TemporaryDirectory() as tmp:
        errors = io.StringIO()
        with (
            contextlib.redirect_stderr(errors),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                with RunLog(tmp, "label", []) as log:
                    raise ValueError("boom")
            except ValueError:
                pass  # it still propagates, so the interpreter prints it once itself
            else:
                raise AssertionError("the exception was swallowed")
        text = log.path.read_text()
        assert (
            "Traceback (most recent call last)" in text and "ValueError: boom" in text
        )
        assert "# crashed: ValueError: boom" in text
        assert errors.getvalue() == ""  # RunLog did not print it as well


def test_how_the_run_ended_is_recorded():
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        outcomes = {}
        for name, exception in [
            ("clean", SystemExit(0)),
            ("status", SystemExit(3)),
            ("ctrl-c", KeyboardInterrupt()),
        ]:
            try:
                with RunLog(tmp, name, []) as log:
                    raise exception
            except (SystemExit, KeyboardInterrupt):
                pass
            outcomes[name] = log.path.read_text().splitlines()[-1]
        assert outcomes["clean"].startswith("# finished normally")
        assert outcomes["status"].startswith("# exited with status 3")
        assert outcomes["ctrl-c"].startswith("# interrupted")


def test_two_runs_never_share_a_log_file():
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        paths = []
        for _ in range(3):  # same label, same second
            with RunLog(tmp, "same", []) as log:
                print(f"run {len(paths)}")
            paths.append(log.path)
        assert len(set(paths)) == 3
        for n, path in enumerate(paths):
            assert f"run {n}" in path.read_text()


def test_the_log_directory_is_created_and_odd_labels_are_made_safe():
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        directory = Path(tmp) / "does" / "not" / "exist"
        with RunLog(directory, "../../evil name/x", []) as log:
            pass
        assert log.path.parent == directory
        assert "/" not in log.path.name and " " not in log.path.name


class _ChattyPlayer(Player):
    def build_enclosure(self):
        print("chatty player: thinking out loud")


def test_a_players_own_prints_are_logged():
    PLAYERS["_chatty"] = _ChattyPlayer
    try:
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
            RunLog(tmp, "chatty", []) as log,
        ):
            Game(
                Args(
                    gui=False,
                    player="_chatty",
                    scenario_path=str(PROJECT / "scenarios/simple_rectangle.json"),
                    seed=1,
                    difficulty="generous",
                    export_scenario=None,
                    debug=False,
                    sandbox=False,
                )
            )
            text = log.path.read_text()
        assert "chatty player: thinking out loud" in text
        assert "No solution generated. SCORE: 0" in text
    finally:
        del PLAYERS["_chatty"]


# ---------------------------------------------------------- the real command


def _run_main(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_main_writes_a_log_that_matches_the_terminal():
    with tempfile.TemporaryDirectory() as tmp:
        done = _run_main(
            "--player", "e", "--scenario", "scenarios/simple_rectangle.json",
            "--seed", "1", "--log-dir", tmp,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        (log,) = _logs(Path(tmp))
        text = log.read_text()
        for line in done.stdout.splitlines():
            assert line in text, f"terminal line missing from the log: {line!r}"
        assert "SCORE: 1072.0" in done.stdout and "SCORE: 1072.0" in text
        assert "# cpu limit: 300s" in text
        assert (
            "# player: e   scenario: scenarios/simple_rectangle.json   seed: 1" in text
        )
        assert str(log) in done.stdout  # the terminal says where the log went
        assert "player CPU time:" in text


def test_main_reports_the_cpu_limit_it_was_given():
    with tempfile.TemporaryDirectory() as tmp:
        done = _run_main(
            "--player", "e", "--scenario", "scenarios/simple_rectangle.json",
            "--seed", "1", "--cpu-limit", "12.5", "--log-dir", tmp,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        assert "# cpu limit: 12.5s" in _logs(Path(tmp))[0].read_text()
        assert "(limit 12.5s)" in done.stdout


def test_main_rejects_a_bad_cpu_limit_and_writes_no_log():
    for bad in ("0", "-5", "abc", "nan"):
        with tempfile.TemporaryDirectory() as tmp:
            done = _run_main("--player", "e", "--cpu-limit", bad, "--log-dir", tmp)
            assert done.returncode == 2, (bad, done.stderr)
            assert "--cpu-limit" in done.stderr
            assert _logs(Path(tmp)) == []


def test_main_defaults_to_a_logs_folder_in_the_project():
    from src.args import DEFAULT_LOG_DIR

    assert DEFAULT_LOG_DIR == PROJECT / "logs"


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
