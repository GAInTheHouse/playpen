from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from pathlib import Path
from time import time

import src.constants as c
from players.registry import PLAYER_HELP, player_names

# logs go in <project>/logs unless --log-dir says otherwise
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


@dataclass
class Args:
    gui: bool
    player: str | None  # None only in sandbox mode, which runs no player
    scenario_path: str | None
    seed: int
    difficulty: str
    export_scenario: str | None
    debug: bool
    sandbox: bool
    cpu_limit: float = c.CPU_TIME_LIMIT_SECONDS
    log_dir: Path = DEFAULT_LOG_DIR


def positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:  # also rejects nan
        raise ArgumentTypeError(f"must be greater than 0, got {text}")
    return value


def player_name(text: str) -> str:
    names = player_names()
    if text not in names:
        raise ArgumentTypeError(
            f"unknown player {text!r}; choose from: {', '.join(names)}"
        )
    return text


def sanitize_seed(org_seed: None | str) -> int:
    if org_seed is None:
        seed = int(time() * 100_000) % 1_000_000
        print(f"Generated seed: {seed}")
        return seed
    return int(org_seed)


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--gui", "-g", action="store_true", help="render GUI")
    parser.add_argument(
        "--seed", "-s", type=int, help="Seed used by the random number generator"
    )
    parser.add_argument(
        "--player",
        "-p",
        type=player_name,
        metavar="PLAYER",
        help=f"Which player to run (required, except with --sandbox): {PLAYER_HELP}",
    )
    parser.add_argument(
        "--difficulty",
        choices=["generous", "scarce"],
        default="generous",
        help="How constrained a generated scenario's inventory should be",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Display debug info")
    parser.add_argument(
        "--sandbox",
        "-x",
        action="store_true",
        help="Load a scenario in sandbox mode (view only, no player runs). Implies `--gui`.",
    )

    parser.add_argument(
        "--cpu-limit",
        type=positive_seconds,
        default=c.CPU_TIME_LIMIT_SECONDS,
        metavar="SECONDS",
        help="CPU time a player may use per run (constructing it + build_enclosure); "
        "past this its run is stopped and scores 0. Default: %(default)s",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="where each run's log file is written (everything printed to the "
        "terminal is also written there). Default: %(default)s",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scenario",
        "-i",
        dest="scenario_path",
        help="path to a scenario JSON file to load",
    )
    group.add_argument(
        "--export-scenario", "-e", help="path to save a freshly generated scenario to"
    )

    namespace = parser.parse_args()
    if namespace.player is None and not namespace.sandbox:
        parser.error("--player is required (for example: --player 1)")

    args = Args(
        gui=namespace.gui or namespace.sandbox,
        player=namespace.player,
        scenario_path=namespace.scenario_path,
        seed=sanitize_seed(namespace.seed),
        difficulty=namespace.difficulty,
        export_scenario=namespace.export_scenario,
        debug=namespace.debug,
        sandbox=namespace.sandbox,
        cpu_limit=namespace.cpu_limit,
        log_dir=namespace.log_dir,
    )

    return args
