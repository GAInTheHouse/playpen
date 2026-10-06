"""Group submissions: finding each group's player, and the submission checker
that groups and CI run. Run with `uv run python -m tests.test_submissions`.

These tests never import or run the real players/player<N>/ code. They use
stand-in modules instead, so that a group's player -- however broken -- can
never make the simulator's own tests fail for everyone else."""

import subprocess
import sys
import tempfile
import types
from pathlib import Path

import scripts.check_submission as check
from players import registry
from players.player0 import Player
from players.registry import (
    GROUP_NAMES,
    NUM_GROUPS,
    PlayerLoadError,
    get_player_class,
    load_group_player,
    player_names,
)

PROJECT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------- finding group players


def test_every_group_name_resolves_through_its_own_module_and_class_name():
    class Stub(Player):
        def build_enclosure(self):
            return None

    stand_ins = {
        f"players.player{n}.player": _module_with(
            **{f"Player{n}": type(f"Player{n}", (Stub,), {})}
        )
        for n in range(1, NUM_GROUPS + 1)
    }
    assert (
        GROUP_NAMES == [str(n) for n in range(1, NUM_GROUPS + 1)] and NUM_GROUPS == 10
    )
    with _FakeImports(stand_ins) as imports:
        for n in range(1, NUM_GROUPS + 1):
            assert get_player_class(str(n)).__name__ == f"Player{n}"
        assert imports.requested == [f"players.player{n}.player" for n in range(1, 11)]


def test_selecting_one_group_loads_only_that_groups_module():
    stub = type("Player4", (Player,), {"build_enclosure": lambda self: None})
    with _FakeImports(
        {"players.player4.player": _module_with(Player4=stub)}
    ) as imports:
        get_player_class("4")
        get_player_class("e")  # a built-in: no group module involved
        assert imports.requested == ["players.player4.player"]


def test_importing_the_registry_imports_no_group_module():
    code = (
        "import sys, players.registry\n"
        "loaded = [m for m in sys.modules if m.startswith('players.player') and m.endswith('.player')]\n"
        "assert not loaded, loaded\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_the_names_a_user_can_pick():
    names = player_names()
    for expected in ["1", "10", "e", "i", "i1", "i10"]:
        assert expected in names
    assert (
        "r" not in names
        and "n" not in names
        and "11" not in names
        and len(set(names)) == len(names)
    )


def test_an_unknown_name_is_an_error_that_lists_the_choices():
    try:
        get_player_class("11")
    except KeyError as e:
        assert "unknown player '11'" in str(e) and "choose from" in str(e)
    else:
        raise AssertionError("an unknown player was accepted")


class _FakeImports:
    """Make the registry's import of group modules return, or raise, what we say."""

    def __init__(self, modules: dict):
        self.modules = modules
        self.requested: list[str] = []

    def __enter__(self):
        self._original = registry.importlib.import_module
        modules = self.modules

        def fake(name, *args, **kwargs):
            if name.startswith("players.player") and name.endswith(".player"):
                self.requested.append(name)
            if name in modules:
                result = modules[name]
                if isinstance(result, Exception):
                    raise result
                return result
            return self._original(name, *args, **kwargs)

        registry.importlib.import_module = fake
        return self

    def __exit__(self, *exc):
        registry.importlib.import_module = self._original


def _module_with(**attributes) -> types.ModuleType:
    module = types.ModuleType("fake")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def test_a_missing_player_file_is_explained():
    gone = ModuleNotFoundError("gone", name="players.player7.player")
    with _FakeImports({"players.player7.player": gone}):
        try:
            load_group_player(7)
        except PlayerLoadError as e:
            assert "players/player7/player.py not found" in str(e)
        else:
            raise AssertionError("no error for a missing file")


def test_a_missing_import_inside_a_groups_code_shows_its_own_error():
    inner = ModuleNotFoundError("No module named 'numpyy'", name="numpyy")
    with _FakeImports({"players.player7.player": inner}):
        try:
            load_group_player(7)
        except ModuleNotFoundError as e:
            assert e.name == "numpyy"  # not disguised as "file not found"
        except PlayerLoadError:
            raise AssertionError("a broken import in the group's code was mislabelled")


def test_a_wrong_class_name_or_wrong_base_class_is_explained():
    class NotAPlayer:
        pass

    with _FakeImports({"players.player7.player": _module_with(MyPlayer=Player)}):
        try:
            load_group_player(7)
        except PlayerLoadError as e:
            assert "must define a class Player7" in str(e)
    with _FakeImports({"players.player7.player": _module_with(Player7=NotAPlayer)}):
        try:
            load_group_player(7)
        except PlayerLoadError as e:
            assert "must subclass players.player0.Player" in str(e)


def test_one_groups_broken_module_does_not_affect_anyone_else():
    broken = SyntaxError("bad syntax")
    with _FakeImports({"players.player7.player": broken}):
        try:
            get_player_class("7")
        except SyntaxError:
            pass
        else:
            raise AssertionError("expected group 7 to fail")
        assert get_player_class("3").__name__ == "Player3"
        assert get_player_class("e").__name__ == "ExamplePlayer"


# ------------------------------------------------------------------- ownership


def test_a_pull_request_inside_one_groups_folders_is_fine():
    group, problems = check.check_ownership(
        [
            "players/player3/player.py",
            "players/player3/helpers.py",
            "players/player3/data/table.json",
            "scenarios/players/player3/tight.json",
        ]
    )
    assert (group, problems) == (3, [])


def test_files_outside_the_groups_folders_are_flagged_each():
    group, problems = check.check_ownership(
        ["players/player3/player.py", "src/game.py", "README.md", "players/player0.py"]
    )
    assert group == 3
    flagged = " ".join(problems)
    for path in ["src/game.py", "README.md", "players/player0.py"]:
        assert path in flagged
    assert "players/player3/player.py" not in flagged


def test_touching_two_groups_is_flagged_and_names_neither_as_the_group():
    group, problems = check.check_ownership(
        ["players/player3/player.py", "players/player4/player.py"]
    )
    assert group is None
    assert any(
        "more than one group" in p and "group 3" in p and "group 4" in p
        for p in problems
    )


def test_lookalike_paths_are_not_a_groups_folder():
    for path in [
        "players/player.py",  # not a group folder
        "players/player0.py",
        "players/player11/player.py",  # there are only 10 groups
        "players/player03/player.py",  # not how the folder is spelled
        "players/player3/../../src/x.py",  # wanders out of the folder
        "players/player3/./../player4/x.py",
        "players/player3",  # the folder itself isn't a file inside it
        "scenarios/player3/x.json",
        "other/players/player3/x.py",
    ]:
        assert check.group_of(path) is None, path


def test_windows_style_paths_and_an_empty_change_are_handled():
    assert check.check_ownership(["players\\player2\\player.py"]) == (2, [])
    group, problems = check.check_ownership([])
    assert group is None and problems


def test_changed_files_asks_git_for_the_diff_against_the_base():
    def git(cwd, *args):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        git(repo, "init", "-q", "-b", "main")
        git(repo, "config", "user.email", "t@example.com")
        git(repo, "config", "user.name", "t")
        (repo / "README.md").write_text("hi\n")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "base")
        git(repo, "checkout", "-q", "-b", "group3/work")
        (repo / "players" / "player3").mkdir(parents=True)
        (repo / "players" / "player3" / "player.py").write_text("x = 1\n")
        (repo / "README.md").write_text("changed\n")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "work")
        changed = check.changed_files("main", cwd=repo)
        assert sorted(changed) == ["README.md", "players/player3/player.py"]
        group, problems = check.check_ownership(changed)
        assert group == 3 and any("README.md" in p for p in problems)


# --------------------------------------------------------------- the structure


def test_check_class_accepts_a_proper_player_and_rejects_the_rest():
    class Good(Player):
        def build_enclosure(self):
            return None

    class Incomplete(Player):
        pass

    class Unrelated:
        pass

    assert check.check_class(3, Good) == []
    assert "does not implement: build_enclosure" in check.check_class(3, Incomplete)[0]
    assert "must subclass" in check.check_class(3, Unrelated)[0]


def test_check_structure_on_stand_in_modules():
    class Good(Player):
        def build_enclosure(self):
            return None

    class ExplodesOnCreation(Player):
        def __init__(self, room, inventory, weights):
            raise RuntimeError("bad constructor")

        def build_enclosure(self):
            return None

    good = {"players.player7.player": _module_with(Player7=Good)}
    with _FakeImports(good):
        assert check.check_structure(7) == []
    bad_ctor = {"players.player7.player": _module_with(Player7=ExplodesOnCreation)}
    with _FakeImports(bad_ctor):
        (problem,) = check.check_structure(7)
        assert "raised RuntimeError: bad constructor" in problem
    broken = {"players.player7.player": SyntaxError("nope")}
    with _FakeImports(broken):
        (problem,) = check.check_structure(7)
        assert "fails to import: SyntaxError" in problem
    gone = {
        "players.player7.player": ModuleNotFoundError(
            "x", name="players.player7.player"
        )
    }
    with _FakeImports(gone):
        assert "not found" in check.check_structure(7)[0]


# ------------------------------------------------------------------------ lint


def _ruff_problems(source: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "player.py"
        path.write_text(source)
        return check.run_ruff([path])


def test_clean_code_passes_the_style_check():
    assert _ruff_problems("def f(x):\n    return x + 1\n") == []


def test_everyday_student_code_is_not_held_to_the_simulators_strict_rules():
    # print, assert and a broad except are all fine in a group's code
    source = (
        "def f(x):\n"
        '    print("thinking", x)\n'
        "    assert x >= 0\n"
        "    try:\n"
        "        return 1 / x\n"
        "    except Exception:\n"
        "        return 0\n"
    )
    assert _ruff_problems(source) == []


def test_badly_formatted_code_is_reported_with_how_to_fix_it():
    problems = _ruff_problems("def f( x ):\n    return   x\n")
    assert (
        len(problems) == 1
        and "formatting" in problems[0]
        and "ruff format" in problems[0]
    )


def test_real_mistakes_are_reported():
    assert any("lint" in p and "F401" in p for p in _ruff_problems("import os\n"))
    assert any("lint" in p and "F821" in p for p in _ruff_problems("print(nope)\n"))
    assert any("lint" in p for p in _ruff_problems("def f(:\n"))


# ----------------------------------------------------------------- smoke runs


def test_runs_are_classified_from_what_they_printed():
    classify = check.classify_run
    assert classify(0, "valid: True (valid)\nSCORE: 1072.0\n", "") == ("valid", "")
    assert classify(
        0, "valid: False (enclosure walls cross themselves)\nSCORE: -1000.0\n", ""
    ) == (
        "invalid",
        "enclosure walls cross themselves",
    )
    assert classify(0, "No solution generated. SCORE: 0\n", "") == ("no solution", "")
    assert classify(
        0,
        "Time limit exceeded: stopped after 2.0s of CPU time (limit 2s). No solution. SCORE: 0\n",
        "",
    ) == ("timeout", "stopped after 2.0s of CPU time")
    status, detail = classify(
        1, "", "Traceback (most recent call last):\n  ...\nZeroDivisionError: x\n"
    )
    assert (status, detail) == ("crash", "ZeroDivisionError: x")
    assert classify(0, "something unexpected\n", "")[0] == "unknown"


def test_a_real_smoke_run_reports_score_and_cpu():
    row = check.smoke_run("e", PROJECT / "scenarios" / "simple_rectangle.json", 20)
    assert row["scenario"] == "simple_rectangle.json"
    assert (row["status"], row["score"]) == ("valid", "1072.0")
    assert row["cpu"] != ""


def test_a_groups_own_scenarios_are_run_after_the_bundled_ones():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "scenarios" / "players" / "player2").mkdir(parents=True)
        (root / "scenarios" / "players" / "player3").mkdir(parents=True)
        for name in ["b.json", "a.json"]:
            (root / "scenarios" / name).write_text("{}")
        (root / "scenarios" / "players" / "player2" / "mine.json").write_text("{}")
        (root / "scenarios" / "players" / "player3" / "theirs.json").write_text("{}")
        original = check.ROOT
        check.ROOT = root
        try:
            assert [p.name for p in check.scenarios_for(2)] == [
                "a.json",
                "b.json",
                "mine.json",
            ]
        finally:
            check.ROOT = original


# ------------------------------------------------------------------ the command


class _NothingToFix:
    """Stand in for the structure and style checks, so the command-flow tests
    below are about the flow and not about any real group's code."""

    def __enter__(self):
        self._saved = check.check_structure, check.run_ruff
        check.check_structure = lambda number: []
        check.run_ruff = lambda paths: []
        return self

    def __exit__(self, *exc):
        check.check_structure, check.run_ruff = self._saved


def test_the_command_passes_when_there_is_nothing_to_fix():
    with _NothingToFix():
        assert check.main(["4", "--no-run"]) == 0


def test_the_command_fails_on_a_structure_or_style_problem():
    with _NothingToFix():
        check.check_structure = lambda number: [
            "Player4 does not implement: build_enclosure"
        ]
        assert check.main(["4", "--no-run"]) == 1
    with _NothingToFix():
        check.run_ruff = lambda paths: ["formatting: ..."]
        assert check.main(["4", "--no-run"]) == 1


def test_the_command_finds_the_group_from_the_changes_and_enforces_ownership():
    original = check.changed_files
    try:
        with _NothingToFix():
            check.changed_files = lambda base: [
                "players/player6/player.py",
                "src/game.py",
            ]
            assert (
                check.main(["--base", "origin/main", "--no-run"]) == 1
            )  # touches shared code
            check.changed_files = lambda base: ["players/player6/player.py"]
            assert (
                check.main(["--base", "origin/main", "--no-run"]) == 0
            )  # group found: 6
            assert (
                check.main(["5", "--base", "origin/main", "--no-run"]) == 1
            )  # not group 5's
            check.changed_files = lambda base: [
                "players/player6/a.py",
                "players/player7/b.py",
            ]
            assert check.main(["--base", "origin/main", "--no-run"]) == 1  # two groups
    finally:
        check.changed_files = original


def test_the_command_can_be_run_as_a_module_from_the_project_root():
    done = subprocess.run(
        [sys.executable, "-m", "scripts.check_submission", "--help"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "group" in done.stdout and "--base" in done.stdout


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
