"""Run Player7 headless on a list of scenarios and collect the results in a CSV.

Run from the repo root with:

    uv run python -m players.player7.run_experiments

Each scenario gets one row (name, perimeter, area, score, run time); the last
line is `# average_score,<value>`. The CSV is written to `OUTPUT_DIR`.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

from players.player7.player import Player7
from src.enclosure import score_construction, validate_construction
from src.limits import CpuLimit, PlayerTimeout
from src.scenario import read_scenario

# Scenario files, relative to SCENARIO_ROOT, e.g. "player7/diamond_room.json".
SCENARIOS: list[str] = [
    "player1/u_corridor.json",
    "player1/B.json",
    "player2/simple_room.json",
    "player3/star.json",
    "player4/easy_rectangle_generous.json",
    "player6/cross_rotated_30.json",
    "player6/narrow.json",
    "player7/square_room_w_cutouts.json",
    "player8/curved.json",
    "player9/32_edge_concave_room.json",
    "player10/i_room.json",
    "player10/twin_octagons.json",
]

SCENARIO_ROOT = Path("scenarios/players")
OUTPUT_DIR = Path("logs/player7_experiments")
CPU_LIMIT = 300.0  # same default as the simulator's --cpu-limit

FIELDS = [
    "scenario",
    "status",
    "perimeter",
    "area",
    "score",
    "run_time_s",
    "cpu_time_s",
]


def run_scenario(name: str) -> dict:
    """Build an enclosure for one scenario and score it like the simulator does."""
    row = {"scenario": name, "perimeter": "", "area": "", "score": 0.0}
    scenario = read_scenario(str(SCENARIO_ROOT / name))
    cpu = CpuLimit(CPU_LIMIT)
    start = time.perf_counter()
    construction = None
    try:
        # one budget for the whole run: building the player + build_enclosure()
        with cpu:
            player = Player7(
                room=scenario.room.copy(),
                inventory=scenario.inventory.copy(),
                weights=scenario.weights,
            )
            construction = player.build_enclosure()
    except PlayerTimeout:
        construction = None
        row["status"] = "timeout"
    except Exception as e:  # noqa: BLE001 -- record failures and keep going
        construction = None
        row["status"] = f"error: {type(e).__name__}: {e}"
    row["run_time_s"] = round(time.perf_counter() - start, 3)
    row["cpu_time_s"] = round(cpu.used, 3)

    if "status" in row:
        return row
    if construction is None:
        row["status"] = "no solution"
        return row

    result = validate_construction(construction, scenario.room, scenario.inventory)
    row["status"] = "valid" if result.valid else f"invalid: {result.reason}"
    row["score"] = round(score_construction(result, scenario.weights), 3)
    if result.valid:
        row["perimeter"] = round(result.perimeter, 3)
        row["area"] = round(result.area, 3)
    return row


def main():
    if not SCENARIOS:
        raise SystemExit("SCENARIOS is empty; add scenario files to run_experiments.py")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"expriment_{stamp}.csv"

    rows = []
    for name in SCENARIOS:
        row = run_scenario(name)
        rows.append(row)
        print(
            f"{name}: {row['status']}; score {row['score']:g}; "
            f"{row['run_time_s']:.2f}s",
            flush=True,
        )

    average = sum(row["score"] for row in rows) / len(rows)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.write(f"# average_score,{average:.3f}\n")

    print(f"average score: {average:.3f}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
