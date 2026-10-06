"""Check a group's submission. Groups run it before opening a pull request; CI
runs the same thing on every pull request.

    uv run python -m scripts.check_submission 3
    uv run python -m scripts.check_submission --base origin/main    # CI: finds the group itself

It checks, in order:
  1. ownership (with --base): the pull request changes only files inside ONE
     group's folders: players/player<N>/ and scenarios/players/player<N>/
  2. structure: players/player<N>/player.py defines `class Player<N>`, a
     subclass of players.player0.Player that can be created
  3. style: `ruff format --check` and a light `ruff check` (errors only)
  4. smoke runs: the player is run on every bundled scenario (and the group's
     own), and each outcome is reported

Failures (exit status 1): an ownership, structure or style problem, or a player
that crashes or hangs. Reported but not failures: timeouts, and invalid
enclosures (-1000) -- a maintainer will want to see them, but a work in
progress can legitimately have them.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from players.player0 import Player
from players.registry import NUM_GROUPS, PlayerLoadError, load_group_player
from src.scenario import read_scenario

ROOT = Path(__file__).resolve().parent.parent

# errors only: syntax errors, undefined names, unused imports, ... -- not the
# hundreds of style rules the simulator's own code is held to
LIGHT_LINT_RULES = "E4,E7,E9,F"


# ------------------------------------------------------------------ ownership


def owned_prefixes(number: int) -> list[str]:
    return [f"players/player{number}/", f"scenarios/players/player{number}/"]


def group_of(path: str) -> int | None:
    """The group whose folder `path` is in, or None if it isn't in any group's.

    Strict on purpose: the number must be written canonically (no `player03`)
    and the path must not wander out of the folder with `..`.
    """
    if ".." in path.split("/"):
        return None
    match = re.match(r"^(?:players|scenarios/players)/player([1-9][0-9]*)/", path)
    if match and int(match.group(1)) <= NUM_GROUPS:
        return int(match.group(1))
    return None


def check_ownership(changed: list[str]) -> tuple[int | None, list[str]]:
    """(the group, problems). The group is None unless exactly one group's
    folders are changed."""
    paths = [p.replace("\\", "/") for p in changed if p.strip()]
    if not paths:
        return None, ["the pull request changes no files"]
    groups = sorted({g for g in map(group_of, paths) if g is not None})
    problems = [
        f"{p}: outside your group's folders (only players/player<N>/ and "
        "scenarios/players/player<N>/ may change; ask a maintainer about shared files)"
        for p in paths
        if group_of(p) is None
    ]
    if len(groups) > 1:
        problems.append(
            "changes more than one group's folders: "
            + ", ".join(f"group {g}" for g in groups)
        )
    return (groups[0] if len(groups) == 1 else None), problems


def changed_files(base: str, cwd: Path = ROOT) -> list[str]:
    done = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(
            f"could not list changed files against {base}: {done.stderr.strip()}"
        )
    return done.stdout.splitlines()


# ------------------------------------------------------------------ structure


def check_class(number: int, cls: type) -> list[str]:
    """Problems with a group's player class, found without running it."""
    name = f"Player{number}"
    if not (isinstance(cls, type) and issubclass(cls, Player)):
        return [f"{name} must subclass players.player0.Player"]
    if getattr(cls, "__abstractmethods__", None):
        missing = ", ".join(sorted(cls.__abstractmethods__))
        return [f"{name} does not implement: {missing}"]
    return []


def check_structure(number: int) -> list[str]:
    try:
        cls = load_group_player(number)
    except PlayerLoadError as e:
        return [str(e)]
    except Exception as e:  # noqa: BLE001 -- anything in a group's module is theirs to fix
        return [
            f"players/player{number}/player.py fails to import: {type(e).__name__}: {e}"
        ]
    problems = check_class(number, cls)
    if problems:
        return problems
    scenario = read_scenario(str(ROOT / "scenarios" / "simple_rectangle.json"))
    try:
        cls(scenario.room.copy(), scenario.inventory.copy(), scenario.weights)
    except Exception as e:  # noqa: BLE001
        return [
            f"Player{number}(room, inventory, weights) raised {type(e).__name__}: {e}"
        ]
    return []


# ------------------------------------------------------------------------ lint


def run_ruff(paths: list[Path]) -> list[str]:
    """Formatting and light-lint problems in `paths`, as lines of ruff output."""
    paths = [p for p in paths if p.exists()]
    if not paths:
        return []
    targets = [str(p) for p in paths]
    problems = []
    for label, args in [
        (
            "formatting (fix with: uv run ruff format <your folder>)",
            ["format", "--check"],
        ),
        ("lint", ["check", "--select", LIGHT_LINT_RULES]),
    ]:
        done = subprocess.run(
            [sys.executable, "-m", "ruff", *args, *targets],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            output = (done.stdout + done.stderr).strip()
            problems.append(
                f"{label}:\n" + "\n".join("    " + ln for ln in output.splitlines())
            )
    return problems


# ------------------------------------------------------------------ smoke runs


def classify_run(returncode: int, stdout: str, stderr: str) -> tuple[str, str]:
    """(status, detail) for one run of `main.py`, from what it printed.
    Statuses: crash, timeout, no solution, invalid, valid, unknown."""
    if returncode != 0:
        lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        return "crash", lines[-1] if lines else f"exit status {returncode}"
    if "Time limit exceeded" in stdout:
        used = parse_line(stdout, r"stopped after ([\d.]+s) of CPU time")
        return "timeout", f"stopped after {used} of CPU time"
    if "No solution generated" in stdout:
        return "no solution", ""
    match = re.search(r"^valid: (True|False) \((.*)\)$", stdout, re.MULTILINE)
    if match:
        return (
            ("valid", "") if match.group(1) == "True" else ("invalid", match.group(2))
        )
    return "unknown", ""


def parse_line(stdout: str, pattern: str) -> str:
    match = re.search(pattern, stdout, re.MULTILINE)
    return match.group(1) if match else ""


def smoke_run(player: str | int, scenario: Path, cpu_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as logs:
        command = [
            sys.executable, "main.py",
            "--player", str(player),
            "--scenario", str(scenario),
            "--seed", "1",
            "--cpu-limit", str(cpu_limit),
            "--log-dir", logs,
        ]  # fmt: skip
        try:
            done = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=cpu_limit * 2
                + 30,  # the limit counts CPU, this guards the clock
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "scenario": scenario.name,
                "status": "hung",
                "detail": "ran far past the time limit",
                "score": "",
                "cpu": "",
            }
    status, detail = classify_run(done.returncode, done.stdout, done.stderr)
    return {
        "scenario": scenario.name,
        "status": status,
        "detail": detail,
        "score": parse_line(done.stdout, r"SCORE: (\S+)\s*$"),
        "cpu": parse_line(done.stdout, r"^player CPU time: ([\d.]+)s"),
    }


def scenarios_for(number: int) -> list[Path]:
    bundled = sorted((ROOT / "scenarios").glob("*.json"))
    own = sorted((ROOT / "scenarios" / "players" / f"player{number}").glob("*.json"))
    return bundled + own


def render_table(rows: list[dict]) -> str:
    lines = [
        "| scenario | result | score | player CPU (s) |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        result = r["status"] + (f": {r['detail']}" if r["detail"] else "")
        lines.append(f"| {r['scenario']} | {result} | {r['score']} | {r['cpu']} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a group's submission.")
    parser.add_argument(
        "group", type=int, nargs="?", help=f"your group number (1-{NUM_GROUPS})"
    )
    parser.add_argument(
        "--base",
        help="git ref the pull request is against (e.g. origin/main): "
        "also checks that only your group's folders changed, and finds the group itself",
    )
    parser.add_argument(
        "--cpu-limit",
        type=float,
        default=60,
        metavar="SECONDS",
        help="CPU limit for each smoke run (default: %(default)s)",
    )
    parser.add_argument("--no-run", action="store_true", help="skip the smoke runs")
    args = parser.parse_args(argv)  # fmt: skip

    failures: list[str] = []
    warnings: list[str] = []
    group = args.group

    if args.base:
        found, problems = check_ownership(changed_files(args.base))
        print("== ownership ==")
        print("  ok" if not problems else "\n".join("  FAIL " + p for p in problems))
        failures += problems
        if group is None:
            group = found
        elif found is not None and found != group:
            failures.append(
                f"the pull request changes group {found}'s folders, not group {group}'s"
            )
    if group is None:
        if failures:
            print("\nCannot tell which group this is, because of the problems above.")
            return 1
        parser.error(
            "give your group number, or --base so it can be found from the changes"
        )
    if not 1 <= group <= NUM_GROUPS:
        parser.error(f"group must be between 1 and {NUM_GROUPS}")

    print(f"== group {group}: structure ==")
    problems = check_structure(group)
    print("  ok" if not problems else "\n".join("  FAIL " + p for p in problems))
    failures += problems

    print("== style ==")
    problems = run_ruff([ROOT / "players" / f"player{group}"])
    print("  ok" if not problems else "\n".join("  FAIL " + p for p in problems))
    failures += problems

    if not args.no_run and not failures:
        print(f"== smoke runs (CPU limit {args.cpu_limit:g}s each) ==")
        rows = []
        for scenario in scenarios_for(group):
            row = smoke_run(group, scenario, args.cpu_limit)
            rows.append(row)
            print(
                f"  {row['scenario']:<44} {row['status']:<12} score {row['score']:<8} {row['detail']}"
            )
            if row["status"] in ("crash", "hung", "unknown"):
                failures.append(
                    f"{row['scenario']}: {row['status']} {row['detail']}".strip()
                )
            elif row["status"] in ("timeout", "invalid"):
                warnings.append(
                    f"{row['scenario']}: {row['status']} {row['detail']}".strip()
                )
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(f"### Group {group}: smoke runs\n\n{render_table(rows)}\n")
    elif failures and not args.no_run:
        print("== smoke runs ==\n  skipped until the problems above are fixed")

    print()
    for w in warnings:
        print(f"  warning: {w}")
    if failures:
        print(
            f"CHECK FAILED ({len(failures)} problem{'s' if len(failures) != 1 else ''})"
        )
        return 1
    print(
        "CHECK PASSED"
        + (
            f" (with {len(warnings)} warning{'s' if len(warnings) != 1 else ''})"
            if warnings
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
