# Player 5 beam search

Player 5 now uses a bounded-width beam search over partial enclosure chains.
By default, the gate starts at the first endpoint of the longest room edge and
points along that edge. Each search layer adds one face: a turn followed by
one or more walls joined with straight connectors. The first added face can
also extend the gate's face with straight connectors. A bounded knapsack over wall lengths finds combinations
that fit each available straight run; the outer beam keeps a longest fit, a
strong area-and-distance candidate, and a shorter alternative. The player
estimates the area of each partial chain as if it closed directly back to the
gate start, then balances that area reward against endpoint distance:

```text
distance(endpoint, gate start) - 0.075 * estimated area
```

This heuristic rewards broad enclosures while keeping partial chains that can
still plausibly close in contention. The estimate is only a ranking signal: a
partial chain may not be able to close directly, and the best completed
construction is still selected by the official score. The search rejects walls
outside the room, crossings, and faces longer than the simulator limit.
For each valid one-gate candidate, Player 5 also tries replacing a same-length
wall on another face with an available gate, preserving the geometry and
earning the second-gate bonus when inventory allows it.
The default beam width is 2048 and the default elapsed-time limit is twenty
seconds. Beam pruning means `search_complete` is always false: the result is a
good candidate found within the bounded search, not a proven optimum.

Run a Player 5 scenario with:

```sh
uv run main.py --player 5 --scenario scenarios/players/player5/rectangle_room.json --seed 1
```

Tune the width or fix the gate pose in Python:

```python
from players.player5.player import Player5

player = Player5(room, inventory, weights, beam_width=2048, time_limit=20)
construction = player.build_enclosure()
print(player.best_score, player.search_complete, player.note)
```

The `workers` argument remains accepted for call compatibility, but this beam
search runs serially. `time_limit=None` removes the clock deadline, while the
finite beam still bounds how many states survive at each depth.
