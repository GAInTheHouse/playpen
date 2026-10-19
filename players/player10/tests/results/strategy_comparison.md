# Player 10 strategy comparison

Compared on 46 bundled scoring scenarios with the same scenario files and the
official validator/scorer. `invalid_examples.json` was excluded. The baselines
are Rishabh V4 (`79e0cd8`) and `origin/player10-circuit` (`90c8e30`).

| Strategy | Valid | Total score | Median score | Total CPU (s) | Median CPU (s) | Unique wins |
|---|---:|---:|---:|---:|---:|---:|
| Current working tree | 43/46 | 345,840.6 | 1,981.1 | 1,106.9 | 6.6 | 3 |
| Original Rishabh V4 | 43/46 | 339,393.9 | 1,897.1 | 589.6 | 6.3 | 1 |
| Circuit branch | 43/46 | 334,107.1 | 1,916.1 | 4,927.9 | 87.5 | 11 |

## Pairwise results

| Comparison | Better | Tied | Worse | Total score difference |
|---|---:|---:|---:|---:|
| Current vs original | 6 | 39 | 1 | +6,446.7 |
| Current vs circuit | 5 | 30 | 11 | +11,733.5 |
| Circuit vs original | 12 | 30 | 4 | -5,286.8 |

The current total-score lead is dominated by the large spiral (+6,320 over the
original and +12,314 over circuit). Excluding that scenario, current is +126.7
over the original and circuit is +580.5 over current. This makes the scenario
distribution more informative than the aggregate alone.

## Main current gains over the original

- `player6/spiral`: +6,320.0
- `player2/corridor`: +424.9
- `player9/lollipop_trap`: +236.0
- `player8/curved`: +150.7
- `player7/square_room_w_cutouts`: +96.5
- `player6/hallway`: +89.9

The only current regression is `player9/crown` (-871.3). It reproduces in an
isolated run and comes from less growth time for the original template basin
after the new raster work, rather than benchmark concurrency.

## Where circuit wins

Circuit has 11 unique wins. Its largest gains over current are:

- `player9/32_edge_concave_room`: +804.3
- `player9/hallways`: +792.9
- `player1/coms4444`: +253.0
- `player9/lollipop_trap`: +244.0
- `player3/star`: +122.8
- `player4/irregular_room_second_gate`: +79.9
- `player6/n_shaped_room`: +72.5

It also improves curved by 16 points, but that result is an octagon template,
not a CP-SAT loop. Circuit's held hexagon/octagon retry is therefore separable
from its expensive lattice solver.

## Design conclusion

Keep the current hybrid as the default. It has the highest total and median
score and uses 22% of circuit's CPU. Add circuit components conditionally:

1. Retry held hexagon/octagon templates for small-stock rooms.
2. Invoke right-angled CP-SAT only for orthogonal, corridor-heavy rooms whose
   lattice fits the node/edge caps and whose current enclosure leaves a large
   score upper-bound gap.
3. Use adaptive refinement for localized gaps after a strong current solution,
   rather than running the entire circuit pipeline.
4. Preserve a separate growth budget for the best pre-raster template to remove
   the crown regression.

The merged per-scenario evidence is in `strategy_comparison.csv`; aggregate
statistics are in `strategy_comparison_summary.csv`.
