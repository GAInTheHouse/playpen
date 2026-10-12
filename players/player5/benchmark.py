"""Benchmark the DFS; run with python -m players.player5.benchmark --help."""

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from players.player5.player import Player5
from src.enclosure import score_construction, validate_construction
from src.limits import CpuLimit, PlayerTimeout
from src.scenario import read_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--simulator-cpu-limit", type=float, default=300.0)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/player5-parallel-wall-5s.json")
    )
    parser.add_argument("--no-distance-pruning", action="store_true")
    parser.add_argument("--no-turn-pruning", action="store_true")
    parser.add_argument("--no-final-wall-pruning", action="store_true")
    args = parser.parse_args()
    scenarios = args.scenarios or sorted(
        Path("scenarios/players").glob("player*/*.json")
    )
    rows = []
    for path in scenarios:
        wall_start = time.monotonic()
        row = {
            "scenario": str(path),
            "workers": args.workers,
            "distance_pruning": not args.no_distance_pruning,
            "turn_pruning": not args.no_turn_pruning,
            "final_wall_pruning": not args.no_final_wall_pruning,
        }
        try:
            scenario = read_scenario(str(path))
            with CpuLimit(args.simulator_cpu_limit) as cpu:
                player = Player5(
                    scenario.room,
                    scenario.inventory,
                    scenario.weights,
                    workers=args.workers,
                    time_limit=args.time_limit,
                    distance_pruning=not args.no_distance_pruning,
                    turn_pruning=not args.no_turn_pruning,
                    final_wall_pruning=not args.no_final_wall_pruning,
                )
                candidate = player.build_enclosure()
            row.update(
                search_complete=player.search_complete,
                total_cpu_seconds=player.search_cpu_seconds,
                simulator_cpu_seconds=cpu.used,
                stats=asdict(player.stats),
                note=player.note,
            )
            if candidate is None:
                row.update(
                    status="no solution"
                    if player.search_complete
                    else "deadline/no solution",
                    score=0,
                )
            else:
                result = validate_construction(
                    candidate, scenario.room, scenario.inventory
                )
                row.update(
                    status="valid" if result.valid else "invalid",
                    score=score_construction(result, scenario.weights),
                    reason=result.reason,
                )
        except PlayerTimeout as exc:
            row.update(status="simulator timeout", score=0, error=str(exc))
        except Exception as exc:  # noqa: BLE001 -- record failures and continue the batch
            row.update(status="error", score=0, error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds"] = time.monotonic() - wall_start
        rows.append(row)
        print(
            f"{path}: {row['status']}; score {row['score']:g}; "
            f"total CPU {row.get('total_cpu_seconds', 0):.2f}s; "
            f"wall {row['wall_seconds']:.2f}s",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "time_limit_seconds": args.time_limit,
                    "simulator_cpu_limit_seconds": args.simulator_cpu_limit,
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Totals: {dict(Counter(row['status'] for row in rows))}", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
