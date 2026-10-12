# Parallel DFS with a fixed starting gate

The player defaults to **five seconds of elapsed time**, with up to four
worker processes exploring different subtrees concurrently. Every worker gets
**the same absolute wall-clock deadline**. Time is never divided by worker
count. Process startup is included in the five-second window; workers starting
later use the time remaining until that same deadline. Combined CPU usage can
exceed five seconds when several cores are busy.

Run an existing Player 5 scenario:

```sh
uv run main.py --player 5 --scenario scenarios/players/player5/rectangle_room.json --seed 1
```

Searches that exhaust their tree return early. The upstream Player 5 scenarios are retained alongside the additional
T-shaped scenario described below.

Each worker explores a disjoint subset of the first connector/wall choices
for each gate length. Normal DFS continues below those branches. Workers do
not repeat each other's subtrees, although each visits the initial gate root.
The assignment is static: there is no shared task queue or work stealing.
Results are merged by maximum score.

At the deadline, each worker returns its best valid polygon so far. The parent
returns the highest-scoring polygon. `search_complete` indicates whether the
entire tree was exhausted; otherwise the result is not a proven optimum.
If no valid polygon was found, return `None`. Deadline checks are cooperative:
a native geometry call and process shutdown/result collection can add a small
amount of time after the search cutoff. All workers are joined before returning.

`search_wall_seconds` reports elapsed build time. `search_cpu_seconds` reports
combined parent and child CPU on Unix using OS accounting after workers exit,
including worker startup and teardown. On platforms without `resource`, the
CPU figure is an estimate excluding worker startup and teardown. CPU is a
measurement, not the player's stopping criterion. The simulator's separate
`--cpu-limit` still limits only the parent process; its default is 300 CPU
seconds, and its printed CPU figure excludes child workers.

This version uses exactly one gate. By default it centers the gate on the
longest room edge and points along that edge. Ties use the first edge in the
room's vertex order, including the closing edge. It tries every usable gate
length, keeping each centered on the same edge midpoint. To customize:

```python
from players.player5.player import Player5

player = Player5(
    room,
    inventory,
    weights,
    start=(5, 5),
    start_heading=0,
    gate_length=5,
    workers=4,
    time_limit=5,
)
construction = player.build_enclosure()
print(player.stats, player.best_score, player.search_complete, player.note)
```

For unbounded exhaustive serial search, pass `workers=1, time_limit=None`.
Programs using multiprocessing must call the player inside an
`if __name__ == "__main__":` guard. The simulator already does this.

At each node DFS appends every available wall length with every available
connector, including both flipped orientations of right and diagonal
connectors. Identical items are counted rather than permuted as distinct
objects. Every closed chain tries the remaining closing connectors, goes
through the simulator's full validator, and is scored. Closures are evaluated
at every depth, so solutions may leave inventory unused. Inventory is copied
before use. Lengths below the simulator's five-unit minimum are unusable and
skipped, including those present in some upstream scenarios.

Three additional pruning rules are enabled by default:

- **Euclidean distance:** prune if the endpoint's distance to the start exceeds
  the maximum length of walls it can still add, reserving one connector for
  closure. The triangle inequality makes this a safe upper bound.
- **Turn feasibility:** a simple polygon must finish at total turn +360 or
  -360 degrees. A cached integer check determines whether any subset of the
  remaining connectors can supply the missing turn within the remaining wall
  and connector counts. Both winding directions and unused pieces are allowed.
- **Final wall:** if at most one more wall can be added, reject choices unless
  that wall actually reaches the start within tolerance and a remaining
  connector restores the initial heading. This check precedes expensive geometry.

The constructor switches are `distance_pruning`, `turn_pruning`, and
`final_wall_pruning`. Corresponding `stats.*_prunes` counters report rejected
nodes for distance/turn checks and rejected wall choices for final-wall checks.

Other dead branches stop when they leave the room (checking the whole wall even
in concave rooms), cross/touch earlier walls, exceed the 30-unit straight-face
limit, or lack walls/connectors. Touching the room boundary is allowed. One
connector is reserved for closure. The official validator checks faces that
wrap across the gate. A closed chain is a leaf.

Without a deadline, enumeration is exhaustive for the chosen pose and one-gate
model, not over continuous placements or constructions with extra gates.
Runtime grows exponentially; recursive traversal is intended for small
inventories. More workers do not necessarily mean more useful search on a
fixed machine, because of startup, scheduling, and static partition imbalance.

The additional scenario `scenarios/players/player5/bigger_is_worse.json` is a
T-shaped room with a 30-by-5 lower wing and a 10-by-10 central upper chamber.
Its perimeter penalty rewards compact enclosures; inventory supports every
candidate below, including the reflex turns around the full T.

| Candidate | Area | Perimeter | Score |
| :-- | --: | --: | --: |
| Full T | 250 | 90 | 935 |
| Lower 30-by-5 wing | 150 | 70 | 905 |
| Central 10-by-15 rectangle | 150 | 50 | 975 |
| Small 5-by-5 square | 25 | 20 | 955 |

These are candidate comparisons, not a proven global optimum. The centered
gate and fixed heading used by this player further restrict which candidates
it can reach.

Run tests and benchmark all player scenarios:

```sh
uv run python -m unittest players.player5.test_search
uv run python -m scripts.check_submission 5 --base upstream/main --cpu-limit 5
uv run python -m players.player5.benchmark --workers 4 --time-limit 5
```

The benchmark writes scores, completion status, node counts, combined CPU,
simulator CPU, and elapsed time to `logs/player5-parallel-wall-5s.json`.
Pass scenario paths for a subset, `--workers 64` to compare worker counts,
`--output PATH` to change the report location, or `--no-distance-pruning`,
`--no-turn-pruning`, and `--no-final-wall-pruning` for ablation experiments.
`--simulator-cpu-limit` controls the separate parent CPU guard (default 300).

[elapsed_time_evaluation.md](elapsed_time_evaluation.md) records earlier
four-worker scenario results and the corrected comparison with 64 workers.

[heuristic_evaluation.md](heuristic_evaluation.md) records the earlier
CPU-budget experiment. Its timed scenario results do not describe the current
elapsed-time deadline. Its unbounded exhaustive correctness/node-count results
remain applicable.

The simulator imports Tkinter even in headless mode. Use a Python installation
with Tkinter available, as described in the repository setup instructions.
