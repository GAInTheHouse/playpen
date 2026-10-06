import platform
import shlex
import sys
from pathlib import Path

from src.args import get_args
from src.game import Game
from src.runlog import RunLog


def log_label(args) -> str:
    scenario = Path(args.scenario_path).stem if args.scenario_path else "generated"
    return f"{args.player or 'sandbox'}_{scenario}_seed{args.seed}"


if __name__ == "__main__":
    args = get_args()
    header = [
        "command: " + shlex.join(sys.argv),
        (
            f"player: {args.player}   scenario: {args.scenario_path or 'generated'}   "
            f"seed: {args.seed}   gui: {'yes' if args.gui else 'no'}"
        ),
        f"cpu limit: {args.cpu_limit:g}s",
        f"python {platform.python_version()} on {platform.platform()}",
    ]
    with RunLog(args.log_dir, log_label(args), header) as log:
        print(f"logging to {log.path}")
        game = Game(args)
