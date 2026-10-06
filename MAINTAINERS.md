# Merging the groups' pull requests

(`FILES.md` explains what every file in the repository is for.)

How the repository is set up so that ten groups can each send pull requests and
you can merge them without conflicts, and what to do for each one.

## How it's built

- **One folder per group.** `players/player1/` ... `players/player10/` (and
  `scenarios/players/player1/` ... `player10/`) already exist, each with an empty
  `class Player<N>`. A group only ever changes its own folder, so two groups' pull requests
  can never conflict with each other.
- **No shared file to edit.** `--player N` finds `players/playerN/player.py` by naming
  convention (see `players/registry.py`), and loads it only when selected: a group
  whose code is broken breaks its own runs, not anyone else's.
- **CI checks every pull request** (`.github/workflows/ci.yml`):
  - *simulator tests and lint*: all the tests, and format/lint of the simulator's own code.
  - *group submission check*: runs `scripts/check_submission.py` against the pull request.
    It finds the group from the changed files, fails the pull request if it changes
    anything outside **one** group's folders, checks the player's structure and
    formatting, then runs the player on every scenario and prints a table (also shown
    on the pull request's checks page).

## One-time setup on GitHub

1. **Branch protection on `main`** (Settings > Branches > Add rule): require a pull
   request before merging; require status checks to pass, and select
   **simulator tests and lint** and **group submission check** (they appear after the first
   run); optionally require a review.
2. **Create a label named `maintainer`.** Put it on pull requests that change shared code
   (below); it skips the group submission check.
3. **Fill in `.github/CODEOWNERS`** with each group's usernames or team, so GitHub
   asks them to review changes to their folder (and you for everything else). Turn on
   "Require review from Code Owners" if you want that to be mandatory.
4. In Settings > Actions, set workflow permissions to read-only (the default is fine; nothing
   here needs write access).

## Reviewing a group's pull request

1. **Checks are green?** Both jobs. "group submission check" passing means: only that
   group's folders changed, the player loads, it's formatted, and it doesn't crash.
2. **Read the file list first.** Anything outside `players/playerN/` and
   `scenarios/players/playerN/` should not be there; the check fails it. In particular be
   suspicious of a pull request that touches `.github/` or `scripts/`: GitHub runs the
   workflow and script *from the pull request itself*, so a pull request could try to weaken
   the very check that judges it. A group submission never needs to touch them.
3. **Check the warnings** in the smoke-run table: `timeout` and `invalid` don't block a
   merge but you'll want to know. An `invalid` result scores -1000.
4. **Try it yourself** (optional, and the only way to see the GUI):

   ```bash
   gh pr checkout 123                      # or: git fetch origin pull/123/head && git switch FETCH_HEAD
   uv run python -m scripts.check_submission --base main
   uv run main.py --gui --player N --scenario scenarios/l_shaped_room.json
   ```

   Runs are logged to `logs/` (git-ignored), with the command, seed, score and CPU time.
5. **Merge.** Squash or merge commit, either is fine. There won't be conflicts between
   groups. If a group's branch is behind, ask them to merge `main` into it.

## When you change shared code

For anything outside the group folders (`src/`, `players/player0.py`, the registry, the
scenarios, CI, docs): open it as your own pull request and add the `maintainer` label.
Afterwards, tell the groups to `git merge origin/main` into their branches. If you change
what players are given or must return (the `Player` base class, `Construction`, the
scoring), say so loudly: that's the one change that can break ten players at once.
`players/player0.py` is the contract. The simulator's own tests deliberately use stand-in
players and never import a group's code, so one group's broken player can't turn the
tests red for everyone else's pull requests; the per-group check covers each group's own code.

## Running everyone's players

```bash
for n in 1 2 3 4 5 6 7 8 9 10; do
  uv run main.py --player $n --scenario scenarios/l_shaped_room.json --seed 1
done
```

Each run has its own 300 s CPU limit (`--cpu-limit` to change), prints its score, and writes
a log to `logs/`. A player that crashes only stops its own run.

## Things this does not do

- **GitHub Actions is untested here.** The workflow was written and its commands run
  locally, but it has not run on GitHub yet; check the first run, and if the window-based
  test step fails there, it is marked non-blocking on purpose.
- **Only the simulator process's CPU is counted.** A player that starts other processes
  (`multiprocessing`, `joblib`) isn't limited by them.
- **No check for what a group's code *does*** beyond running it: it can read files, use the
  network, and so on. The simulator doesn't sandbox player code.
- **The group number isn't tied to a person.** The check makes sure a pull request stays
  inside *one* group's folder, not that it's the *right* group's. CODEOWNERS and your review
  cover that.
