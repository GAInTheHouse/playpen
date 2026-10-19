# Player6: grid search, beam search, and local 45-degree refinement

All implementation, tests, benchmark code, and benchmark outputs for this change
are inside `players/player6/`.

## Running the tests

From the repository root:

```bash
uv run python -m unittest players.player6.test_player players.player6.test_inscribed players.player6.test_local_search players.player6.test_benchmark
uv run ruff check players/player6
uv run python -m scripts.check_submission 6 --no-run
```

The tests cover scarce/fixed-seed inputs, small rooms, insufficient inventory,
equal-length walls and gates, real gate-face bonuses, CPU-budget fallback,
rotated rooms, a construction starting in the middle of a straight face,
strict score improvements, and preservation of both input inventory and seed.

## Benchmarking

```bash
# Full comparison on eight cases, including spiral and two fixed scarce seeds.
uv run python -m players.player6.benchmark

# Repeat selected cases to assess variation under CPU deadlines.
uv run python -m players.player6.benchmark --case spiral --case scarce_seed1 --repeats 3

# A short smoke comparison with smaller, explicitly split budgets.
uv run python -m players.player6.benchmark --case tiny_room --case insufficient_stock --search-seconds 1 --grid-seconds 0.3 --beam-seconds 0.4 --local-seconds 0.1
```

The default outputs are `benchmark_results.json`, `benchmark_results.csv`, and
`benchmark_results.md` in this directory. `--output PATH` changes their location.
Each row records validity/no-solution/timeout/error, score, area, perimeter,
gate count and actual gate bonus, CPU time, wall time, grid CPU time, local-search
statistics, and whether the input inventory was preserved. Returning no solution
is expected for the deliberately insufficient-stock case and scores zero.

`baseline` runs the current grid + rectangle/L + beam strategy with local moves
disabled. `local45` enables the local pass. Both have the same 45-second overall,
20-second grid, and 20-second beam CPU budgets. Local moves share a two-second
budget, use spare orthogonal-stage time before beam search, and may use remaining
time afterward. They do not shorten the reserved beam budget. A simulator CPU
guard at the overall budget + 5 seconds detects hangs/large overruns; cooperative
checks normally stop much earlier. A single native geometry call can overshoot
the cooperative deadline until it returns.

This is an ablation of local refinement, not a comparison against the historical
pre-grid player. Deadline-limited searches can vary between runs, and improving
an incumbent locally does not establish that every complete run will beat a
separate baseline run.

The checked-in single-run report contains 16 runs across eight cases: 14 valid
solutions and two expected no-solutions for insufficient stock, with no invalid
solutions, crashes, timeouts, or inventory mutations. Final scores were equal
between variants on this set. The diagonal-room local pass accepted an earlier
improvement that beam search also found; spiral found no local improvement and
took about 1.5 extra CPU seconds. These results do not show an end-to-end score
gain yet. Set `Player6.LOCAL_SECONDS = 0` to disable the pass, or compare different
budgets with the benchmark CLI. See [benchmark_results.md](benchmark_results.md)
and the JSON/CSV files for all scores, timings, and local-search statistics.

## How the local move works

Replacing an integer-grid corner with one diagonal usually requires a noninteger
piece length. Instead, `local_search.py` translates a chain of two to four faces
and inserts equal-length, opposite diagonal faces at its ends. Their
`sqrt(2)/2` coordinate contributions cancel exactly. The old faces keep their
integer lengths and assigned pieces. Only spare pieces/straight connectors are
used for the new faces; all right/diagonal connector counts are recomputed.

The new outline is tried at its current position and a few bounding-box-based
translations without changing its orientation. The official validator checks
closure, self-intersections, containment, all inventories, the 30-unit face limit,
and gates. A move is accepted only if the official score strictly increases.
On timeout, the best validated construction is retained. The original seed and
inventory are never modified.

For example, a gated 10-by-10 square in a 15-by-15 room can use two spare
5-unit walls and four diagonal connectors. Under `A=2, C=-3, G=100`, its area
increases from 100 to approximately 170.711 and its score increases from 1080
to approximately 1191.421, with an exactly closed integer-length construction.

The pass is intentionally limited: it keeps existing material assignments,
uses only this paired-diagonal neighborhood and a few translations, and may
find no move when stock, connectors, placement, or remaining time are tight.
The general beam search remains necessary.
