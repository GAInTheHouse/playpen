# The Playpen Simulator, Explained

This document walks through *every* moving part of the simulator: what each
module is responsible for, exactly how a `Construction` turns into geometry,
exactly what the validator checks and in what order, and exactly what your
`Player` subclass needs to return. It assumes you've read the assignment
spec (Project 2: Playpen) and the top-level `README.md` (which covers *how
to run* the simulator); this document is about *how it works internally* and
what you, as a player author, need to know to work with it correctly.

If you only read one section, read [§7, The `Construction` object](#7-the-construction-object-the-contract-between-you-and-the-simulator)
and [§9, `validate_construction`](#9-validate_construction-the-full-rulebook-in-order) —
those two are where almost every subtle bug in a player implementation comes
from.

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [Project layout](#2-project-layout)
3. [Terminology: spec ↔ code](#3-terminology-spec--code)
4. [`Piece` and `Connector`: the physical parts](#4-piece-and-connector-the-physical-parts)
5. [`Inventory`: what you have to build with](#5-inventory-what-you-have-to-build-with)
6. [`Room`: the footprint you must fit inside](#6-room-the-footprint-you-must-fit-inside)
7. [The `Construction` object: the contract between you and the simulator](#7-the-construction-object-the-contract-between-you-and-the-simulator)
8. [`build_geometry`: turning a `Construction` into points](#8-build_geometry-turning-a-construction-into-points)
9. [`validate_construction`: the full rulebook, in order](#9-validate_construction-the-full-rulebook-in-order)
10. [`compute_faces`: grouping pieces into linear faces](#10-compute_faces-grouping-pieces-into-linear-faces)
11. [Scoring](#11-scoring)
12. [`Weights` and `Scenario`: what you're handed at the start](#12-weights-and-scenario-what-youre-handed-at-the-start)
13. [The `Player` contract](#13-the-player-contract)
14. [Worked example: reading `ExamplePlayer`](#14-worked-example-reading-exampleplayer)
15. [How `Game` wires everything together](#15-how-game-wires-everything-together)
16. [Common pitfalls](#16-common-pitfalls)
17. [Adding your own player](#17-adding-your-own-player)

---

## 1. The big picture

You are handed three things at the start of the game (bundled into a
`Scenario`):

- a **room** — a polygon describing the floor you must build inside,
- an **inventory** — a fixed, possibly-unbalanced stock of walls, gates, and
  connectors,
- a set of **weights** (`A`, `C`, `G`) describing how a valid solution will
  be scored.

Your job, implemented in `Player.build_enclosure()`, is to describe a closed
loop — walls and gates joined end-to-end by connectors — that:

- closes properly (the last piece connects back to the first),
- doesn't cross itself,
- fits entirely inside the room,
- has at least one gate,
- doesn't use more pieces/connectors than the inventory has,
- and doesn't have any single straight run of pieces longer than 30 units.

You don't build geometry yourself and hand back a polygon. Instead you hand
back a **recipe** — a `Construction`: a starting point, a starting
direction, an ordered list of pieces, and an ordered list of connectors. The
simulator (`src/enclosure.py`) is what actually walks that recipe, computes
the resulting shape, and checks every rule above. This split matters: it
means the simulator is the single source of truth for what counts as valid,
and your player never has to reimplement polygon math to know whether its
own plan is legal — it can just call `validate_construction` (or the
lighter-weight `self.room.contains_enclosure(...)`) itself before
committing.

If `build_enclosure()` returns `None`, you score **0** — this is the
deliberate "I chose not to gamble" outcome. If it returns a `Construction`
that fails validation, you score **-1000**. If it returns a valid one, you
score `1000 + A*area + C*perimeter (+ G if applicable)`. Deciding when it's
worth returning a risky-but-plausible construction versus bailing out with
`None` is a real part of the strategy — see the spec's closing question
about scarce vs. plentiful inventory.

## 2. Project layout

```
main.py                    entry point: parse args, construct Game
src/
  args.py                  CLI argument parsing
  constants.py             tunable numbers (canvas sizes, MIN_WALL_LENGTH, MAX_FACE_LENGTH, tolerances, ...)
  pieces.py                Piece, Connector, ConnectorType, PieceType
  inventory.py             Inventory (+ random inventory generator)
  room.py                  Room (+ random room-polygon generator)
  weights.py               Weights (A, C, G)
  scenario.py              Scenario = Room + Inventory + Weights, JSON (de)serialization
  enclosure.py             Construction, build_geometry, compute_faces,
                           validate_construction, score_construction
                           <- this is the file that defines "what is legal"
  game.py                  Game: the tkinter GUI + the headless run loop
players/
  player0.py               abstract Player base class (you subclass this)
  player1/ ... player10/   one folder per group; each holds player.py with an empty
                           `class Player<N>` for that group to fill in
  example_player.py        the one worked example: a deterministic rectangle-builder (see §14)
  registry.py              turns `--player <name>` into a class (finds group players itself)
demos/
  invalid_examples.py      demo player (`--player i`): cycles through deliberately invalid enclosures
scripts/
  check_submission.py      the check groups run before a pull request, and CI runs on every one
scenarios/
  players/player1/ ... player10/   each group's own test scenarios
  simple_rectangle.json     a small, generous example scenario
  l_shaped_room.json        a non-convex room, scarcer inventory
tests/
  test_enclosure.py         worked examples of every validation rule -- read these!
```

Everything under `src/` is simulator infrastructure you should treat as
fixed (you can and should read it, but you shouldn't need to modify it).
Your own code goes in your group's folder, `players/player<N>/` (see §17).

## 3. Terminology: spec ↔ code

| Spec language                                   | Code                                                              |
| :----------------------------------------------- | :----------------------------------------------------------------- |
| wall / gate                                      | `Piece` (`PieceType.WALL` / `PieceType.GATE`), in `src/pieces.py`   |
| connector, joining angle                         | `Connector`, `ConnectorType` (`STRAIGHT`/`RIGHT`/`DIAGONAL`), `src/pieces.py` |
| "used in either orientation" (270°, 225°)        | `Connector(..., reflex=True)`                                      |
| room polygon / footprint                         | `Room.polygon`, `src/room.py`                                      |
| "fits within this room" check                    | `Room.contains_enclosure(polygon)`                                  |
| the enclosure itself                             | the `shapely.Polygon` built from your `Construction`'s vertices     |
| "last piece connects back to first"               | position/heading closure checks in `build_geometry`                 |
| linear face                                      | `Face`, produced by `compute_faces`                                 |
| weights A, C, G                                  | `Weights`, `src/weights.py`                                         |
| "fixed collection of connectors/walls/gates"      | `Inventory`, `src/inventory.py`                                     |

## 4. `Piece` and `Connector`: the physical parts

From `src/pieces.py`:

```python
class PieceType(Enum):
    WALL = "wall"
    GATE = "gate"


class ConnectorType(Enum):
    STRAIGHT = "straight"  # 180 degrees
    RIGHT = "right"  # 90 degrees (or 270, flipped)
    DIAGONAL = "diagonal"  # 135 degrees (or 225, flipped)


@dataclass(frozen=True)
class Piece:
    piece_type: PieceType
    length: int  # must be >= MIN_WALL_LENGTH (5)


@dataclass(frozen=True)
class Connector:
    connector_type: ConnectorType
    reflex: bool = False
```

A `Piece` is just "wall or gate, this many units long." `Piece.__post_init__`
enforces the minimum length (5 units) the moment you construct one, so an
under-length piece fails immediately rather than surfacing as a confusing
validation error later.

A `Connector` is one of the three physical connector types from the spec,
plus a `reflex` flag for "flip it to the other orientation." Two methods
matter:

- **`internal_angle()`** — the actual interior angle this connector forms,
  looked up from `CONNECTOR_INTERNAL_ANGLES` in `constants.py`:

  | `connector_type` | `reflex=False` | `reflex=True` |
  | :--------------- | :-------------- | :-------------- |
  | `STRAIGHT`        | 180°             | *(no effect; always 180°)* |
  | `RIGHT`           | 90°              | 270°             |
  | `DIAGONAL`        | 135°             | 225°             |

  Setting `reflex=True` on a `STRAIGHT` connector is a no-op — the spec only
  describes two orientations for the angle-forming connectors (90°/270° and
  135°/225°); a straight joint is 180° either way.

- **`turn_angle()`** — the *exterior* turn this connector represents,
  `180 - internal_angle`. This is the number that actually gets used to
  walk the geometry (§8). A few worked values:

  | internal angle | turn angle | meaning                                    |
  | :-------------- | :---------- | :------------------------------------------ |
  | 180° (straight) | 0°          | keep going the same direction               |
  | 90° (right, convex) | +90°    | turn left 90° (for a CCW-traced polygon)     |
  | 270° (right, reflex) | -90°   | turn right 90° -- a concave "notch"          |
  | 135° (diagonal, convex) | +45° | gentle left turn                           |
  | 225° (diagonal, reflex) | -45° | gentle right turn (concave)                |

  Convex corners (90°, 135°) produce *positive* turns; reflex corners (270°,
  225°) produce *negative* turns. This sign convention is exactly the
  standard "turtle graphics" / shoelace convention: **walking all the way
  around a simple polygon counterclockwise, the turn angles sum to exactly
  +360°.** That single fact is what the closure check in `build_geometry`
  relies on (§8), and it's the fact you should reach for whenever you're
  trying to reason about whether a sequence of connectors can possibly close.

## 5. `Inventory`: what you have to build with

`src/inventory.py`. Three dictionaries:

```python
@dataclass
class Inventory:
    walls: dict[int, int]  # length -> count
    gates: dict[int, int]  # length -> count
    connectors: dict[ConnectorType, int]  # type -> count
```

Note connector counts are keyed by `ConnectorType` only — **not** by
orientation. A `RIGHT` connector and its reflex flip are the *same physical
part* used a different way, so they draw from the same pool. If your
inventory has 6 `RIGHT` connectors, you can use them in any mix of 90°/270°
corners, but at most 6 corners total that are "right-angle-family."

Key methods:

- **`copy()`** — a deep-enough copy (fresh dicts) so you can simulate taking
  pieces without mutating the original. **Use this liberally.** Both
  `ExamplePlayer` and `validate_construction` itself take a `copy()` before
  tentatively consuming pieces, specifically so a failed attempt doesn't
  corrupt the real inventory.
- **`take_piece(piece)`** / **`take_connector(connector_type)`** — decrement
  the relevant pool and return `True`, or return `False` (and change
  nothing) if none remain. This is written so you can use it directly as an
  "is this even feasible" check: try to take everything you plan to use
  against a scratch copy, and bail the moment something returns `False`.
- **`has_gate()`** — convenience: `True` if any gate length has `count > 0`.
  Worth checking early in your own `build_enclosure` — if it's `False`,
  there is no possible valid construction (the spec requires at least one
  gate), so you can return `None` immediately.

**Important:** the `Inventory` object your player receives (`self.inventory`
in `Player.__init__`) is *your own copy*, separate from the one the
simulator uses to score you at the end (see §15). Mutating it while you plan
is completely safe and has no effect on your final score one way or the
other — what matters is only the final `Construction` you return, which
gets checked from scratch against a *fresh* copy of the real inventory.

## 6. `Room`: the footprint you must fit inside

`src/room.py`. Thin wrapper around a `shapely.Polygon`:

```python
class Room:
    def __init__(self, polygon: Polygon): ...
    def contains_enclosure(self, enclosure_polygon: Polygon, tol=TOL) -> bool: ...
    def get_boundary_points(self) -> list[tuple[float, float]]: ...
```

`room_is_ok()` (module-level function) enforces the spec's constraints on
the room itself when a `Room` is constructed: must be a valid polygon,
non-empty, positive area, and **no holes** (`len(p.interiors) == 0`). You
won't normally need to construct a `Room` yourself — it comes from the
`Scenario` — but it's worth knowing these constraints are guaranteed to
already hold for whatever room you're handed.

**`contains_enclosure`** is the method the spec promises: *"the method to
check enclosure containment will be exposed by the simulator so that your
player can check the enclosure and avoid surprises."* It does
`self.polygon.buffer(tol).contains(enclosure_polygon)` — the small `buffer`
gives the same floating-point tolerance the real validator uses, so a
construction that grazes the room boundary within `TOL` (1e-4 units) won't
be unfairly rejected by your own pre-check but then somehow pass or fail
differently in the simulator. Call this on any candidate `shapely.Polygon`
you build (e.g. `Polygon(build_geometry(construction)[0])`) before
committing to a `Construction` — it's much cheaper than assembling a whole
`Construction`, running full validation, and finding out only then that it
doesn't fit.

Rooms are generated (`generate_room`) using a "star-shaped polygon"
construction: pick a center, sample `N` angles around the full circle, sort
them, and pick a random radius for each angle. Because the angles are
sorted and every vertex is measured from one common center, the resulting
polygon is *guaranteed* to be simple (non-self-intersecting) — no
post-hoc repair needed — and varying the radii per angle is what produces the
non-convex "bites" out of the shape.

## 7. The `Construction` object: the contract between you and the simulator

This is the one data structure you build and hand back. From
`src/enclosure.py`:

```python
@dataclass
class Construction:
    start: tuple[float, float]  # where piece 0 begins
    start_heading: float  # degrees, direction piece 0 points
    pieces: list[Piece]  # length N
    connectors: list[Connector]  # length N -- one per VERTEX, not per edge
```

The single most important thing to internalize: **`pieces` has N entries
and `connectors` also has N entries, but they don't line up the way you
might first guess.** `pieces[i]` is the *edge* from vertex `i` to vertex
`i+1`. `connectors[i]` is the turn made *at* vertex `i` — the joint between
the incoming piece (`pieces[i-1]`, wrapping around from the last piece for
`i=0`) and the outgoing piece (`pieces[i]`).

```
                  connectors[2]
                       ●───── pieces[2] ──────●
                      /                        \
          pieces[1]  /                          \  pieces[3]
                    /                              \
                   ●                                ●
             connectors[1]                    connectors[3]
                   |                                |
                   |          pieces[0]             |
                   ●──────────────────────────────●
              (start, this is  connectors[0] is the turn
               also vertex 0)  HERE, between pieces[3]
                                (incoming) and pieces[0]
                                (outgoing) -- the "closing" joint
```

So `connectors[0]` is special: it's the turn at the *start* vertex, between
the *last* piece and the *first* piece — i.e. it's the joint that "closes
the loop." It is not "the connector before piece 0 starts" in some
directionless sense; it only makes sense once the whole loop is walked and
you arrive back at vertex 0 from the other side.

### Worked example: a 10×6 rectangle with one gate

Let's build the simplest possible valid `Construction` by hand and hand-walk
the geometry, so the vertex/connector indexing is completely concrete.

```python
Construction(
    start=(0, 0),
    start_heading=0.0,  # piece 0 points along +x (east)
    pieces=[
        Piece(PieceType.GATE, 10),  # pieces[0]: bottom, 10 units, this is our gate
        Piece(PieceType.WALL, 6),  # pieces[1]: right side, 6 units
        Piece(PieceType.WALL, 10),  # pieces[2]: top, 10 units
        Piece(PieceType.WALL, 6),  # pieces[3]: left side, 6 units
    ],
    connectors=[
        Connector(
            ConnectorType.RIGHT
        ),  # connectors[0]: closing joint (left-side -> bottom)
        Connector(ConnectorType.RIGHT),  # connectors[1]: bottom -> right side
        Connector(ConnectorType.RIGHT),  # connectors[2]: right side -> top
        Connector(ConnectorType.RIGHT),  # connectors[3]: top -> left side
    ],
)
```

Every connector here is a plain (non-reflex) `RIGHT`, so every `turn_angle()`
is +90°. Walking it (this is exactly what `build_geometry` does, see §8):

| i | heading before piece i | piece i (length, direction)         | vertex after piece i |
| - | :---------------------- | :----------------------------------- | :--------------------- |
| 0 | 0°  (start_heading)     | 10 units east                        | (10, 0)                 |
| 1 | 0° + 90° = 90°          | 6 units north                        | (10, 6)                 |
| 2 | 90° + 90° = 180°        | 10 units west                        | (0, 6)                  |
| 3 | 180° + 90° = 270°       | 6 units south                        | (0, 0)  ✓ back to start |

Closing check: after piece 3 the heading is 270°; applying `connectors[0]`'s
turn (+90°) gives 270°+90° = 360° ≡ 0°, which matches `start_heading`
exactly — the loop closes both in position *and* heading. This rectangle
has area 60, perimeter 32, one gate, and (since every connector is `RIGHT`,
none `STRAIGHT`) four separate one-piece faces, each trivially under the
30-unit limit.

This exact example (with numeric assertions) lives in
`tests/test_enclosure.py::test_valid_rectangle_closes_and_scores` — run it,
put a `print(build_geometry(con))` in the middle, and watch the vertices
fall out.

### A concave (reflex) example

To make an L-shaped enclosure you need at least one reflex corner. Swapping
one `Connector(ConnectorType.RIGHT)` for
`Connector(ConnectorType.RIGHT, reflex=True)` gives that vertex a 270°
internal angle and a **-90°** turn instead of +90° — the path turns the
*other* way, carving a notch instead of a corner. Whenever you're
constructing something non-convex, this is the flag you need.

## 8. `build_geometry`: turning a `Construction` into points

This is the function that does the walk from §7, in code
(`src/enclosure.py`):

```python
def build_geometry(construction):
    n = len(construction.pieces)
    if n != len(construction.connectors):
        raise EnclosureError(...)  # one connector per vertex, always
    if n < MIN_ENCLOSURE_PIECES:  # need at least 3 pieces to enclose anything
        raise EnclosureError(...)

    headings = [0.0] * n
    headings[0] = construction.start_heading
    for i in range(1, n):
        headings[i] = headings[i - 1] + construction.connectors[i].turn_angle()

    vertices = [construction.start]
    for i in range(n):
        theta = radians(headings[i])
        dx = construction.pieces[i].length * cos(theta)
        dy = construction.pieces[i].length * sin(theta)
        vertices.append((vertices[-1][0] + dx, vertices[-1][1] + dy))

    closing_turn = construction.connectors[0].turn_angle()
    expected_final_heading = headings[-1] + closing_turn
    heading_gap = angle_gap(expected_final_heading, headings[0])
    position_gap = distance(vertices[-1], vertices[0])

    return vertices[:-1], position_gap, heading_gap
```

Notice `connectors[0]` is deliberately *not* used in the `headings[1..n-1]`
loop — it only shows up at the very end, to check the closing joint. That
matches §7: `connectors[0]` is the turn "at the start," which only means
something once you've walked all the way back around to it.

The function returns three things:

- **`vertices`** — the `n` actual corner points of the shape (note: the
  `n+1`-th point, which should coincide with `vertices[0]`, is dropped —
  it's redundant once you know the shape closes).
- **`position_gap`** — how far off (in units) the walk actually is from
  closing. For a perfect construction this is `0.0` (up to floating point
  noise); `validate_construction` requires it to be within `TOL` (1e-4).
- **`heading_gap`** — how far off (in degrees) the closing connector's
  declared turn is from what the geometry actually needs. This check exists
  because it's entirely possible to declare a `Construction` whose *edges*
  happen to land back at the start point by coincidence, while the
  `connectors[0]` you specified doesn't actually describe the angle that
  closure requires — i.e., you got the position right but lied about (or
  miscalculated) the angle of the final joint. `validate_construction`
  requires this within `ANGLE_TOL` (1e-3 degrees).

**Why two separate checks instead of one?** Because they catch different
mistakes. Getting `position_gap` right but `heading_gap` wrong usually means
an arithmetic slip in choosing `connectors[0]`'s type/orientation. Getting
`heading_gap` right (turns genuinely sum to a multiple of 360°) but
`position_gap` wrong usually means your edge lengths don't actually satisfy
the vector-sum-to-zero condition a closed polygon needs (e.g., you turned
by exactly 360° total but used mismatched opposite-side lengths).

`unavailable_items(construction, inventory)` answers "which pieces and
connectors can the inventory not supply?" by taking everything in construction
order (all pieces, then all connectors), so it lists the *second* 13-unit wall
when only one is in stock. `validate_construction` uses it for its inventory
check, and the GUI uses it to draw those items as translucent ghosts instead
of placing them.

`walk_points(construction)` exposes the same walk as all `n+1` points, including where the last piece *actually* ends — which is what the GUI draws, so a loop that doesn't close visibly doesn't (with a marked gap).

`sketch_points(construction)` is the never-raising version the GUI actually uses: for a
structurally broken construction (say, fewer connectors than vertices) it still lays out
every wall it can, and reports which vertices have no connector so they can be marked.
The wall after such a vertex has no real direction, so it is drawn as a dashed ghost
using the previous turn — a forgotten corner of a rectangle shows up where the
rectangle wanted it.

`build_geometry` only raises `EnclosureError` for *structural* impossibility
— wrong list lengths, or too few pieces to enclose anything. It does **not**
raise just because the shape doesn't close; that's a normal, expected
outcome that `validate_construction` reports as a regular "invalid, here's
why" result rather than an exception. Exceptions are for "this
`Construction` couldn't even be interpreted," not for "this construction is
a legal-shaped `Construction` that turns out to be a bad enclosure."

## 9. `validate_construction`: the full rulebook, in order

`validate_construction(construction, room, inventory, tol=TOL)` runs every
check the spec requires, **in this exact order**, and returns a
`ValidationResult` the moment one fails. Knowing this order is the single
fastest way to debug "why is my construction invalid": read the `reason`
string, and you now also know every earlier check already passed.

1. **Structural sanity** (`build_geometry` doesn't raise) — matching
   `pieces`/`connectors` lengths, at least 3 pieces.
   → *reason comes straight from the `EnclosureError` message.*
2. **Position closure**: `position_gap <= TOL`.
   → `"enclosure does not close: last piece ends X units from the start"`
3. **Heading closure**: `heading_gap <= ANGLE_TOL`.
   → `"closing connector's angle is inconsistent with the geometry (off by X degrees)"`
4. **Inventory sufficiency**: every piece, then every connector, must be
   `take`-able from a **scratch copy** of `inventory` (the real one is never
   mutated by validation).
   → `"not enough wall(10) in inventory to build this construction"` (or gate/connector-flavored variants)
5. **At least one gate**: `gate_count >= 1`.
   → `"enclosure must have at least one gate"`
6. **Simple polygon**: `shapely` says the resulting `Polygon` is valid and
   its exterior ring doesn't self-intersect.
   → `"enclosure walls cross themselves"`
7. **Positive area** (guards against degenerate zero-area polygons slipping
   past the simplicity check).
   → `"enclosure has zero area"`
8. **Room containment**: `room.contains_enclosure(polygon)`.
   → `"enclosure does not fit within the room"`
9. **Face length**: every `Face` from `compute_faces` is `<= MAX_FACE_LENGTH`
   (30 units, plus tolerance).
   → `"a linear face is X units long, exceeds the 30-unit limit"`
10. Only if **all** of the above pass: compute `second_gate_bonus` (§10) and
    return `valid=True` along with `area`, `perimeter`, `gate_count`,
    `faces`, and the `polygon` itself.

Two things worth calling out explicitly:

- **Order is a debugging tool.** If you get `"enclosure does not fit within
  the room"`, you already know your shape closes correctly, uses inventory
  you actually have, and has a gate — the *only* remaining problem is
  position. Don't go re-checking earlier-stage things a later-stage error
  message already implies are fine.
- **The inventory check uses a scratch copy** (`inventory.copy()` inside
  `validate_construction`), so calling this function never has side effects
  on the `inventory` object you pass in. You can call it as many times as
  you like while exploring candidate constructions without worrying about
  bookkeeping.

## 10. `compute_faces`: grouping pieces into linear faces

The spec: *"Every linear face of the enclosure must be no longer than 30
units. A linear face may include several walls joined by 180-degree
connectors."* In code, a face is a maximal run of pieces where every
connector *between* them (not at the ends of the run) is `STRAIGHT`.

The implementation (`src/enclosure.py`) uses the vertex/connector
convention from §7 directly: **piece `i` starts a new face (relative to
piece `i-1`) exactly when `connectors[i]` is not straight** — because
`connectors[i]` is precisely the joint between those two pieces.

```python
starts = [i for i in range(n) if not construction.connectors[i].is_straight()]
```

Then it walks the circular list of pieces, cutting a new `Face` at each
index in `starts`. For the rectangle example in §7 (all four connectors are
`RIGHT`, none `STRAIGHT`), `starts = [0, 1, 2, 3]` — every piece starts its
own face, so you get four one-piece faces.

### Worked example: a split side

Suppose the bottom side of a rectangle is built from *three* 8-unit walls
joined by `STRAIGHT` connectors instead of one 24-unit wall (maybe that's
all the inventory had):

```python
pieces = [wall(8), wall(8), wall(8), wall(10), gate(24), wall(10)]
connectors = [RIGHT, STRAIGHT, STRAIGHT, RIGHT, RIGHT, RIGHT]
#             ^vtx0                        ^vtx3     ^vtx4     ^vtx5
```

Here `starts = [0, 3, 4, 5]` (indices 1 and 2 are `STRAIGHT`, so they don't
start new faces). Walking from each start to the next gives faces
`[0, 1, 2]` (length 8+8+8 = 24), `[3]` (length 10), `[4]` (length 24),
`[5]` (length 10) — four faces, the first and third each under the 30-unit
cap even though they're each built from a single "logical" side of the
rectangle. This exact shape (with a face that *does* exceed 30, to show the
rejection) is `tests/test_enclosure.py::test_face_too_long_is_invalid`.

### The second-gate bonus, precisely

`validate_construction` computes:

```python
gate_faces = {id(f) for f in faces if f.has_gate}
second_gate_bonus = len(gate_faces) >= 2
```

This is **not** "are there two gates." It's "are there gates on two
*different* faces." Two gates on the same straight run (e.g. both halves of
a side that's been split, joined by a `STRAIGHT` connector between them)
still count as one face and **do not** earn the bonus — see
`tests/test_enclosure.py::test_two_gates_on_same_face_no_bonus` versus
`test_second_gate_on_different_face_gives_bonus`. If you're specifically
chasing the `G` bonus, make sure your second gate sits behind a real
(non-straight) corner from the first one.

## 11. Scoring

`score_construction(result, weights)` (which is just the total of
`explain_score(result, weights)`, the per-term breakdown the GUI displays):

```python
if not result.valid:
    return -1000.0
score = 1000.0
score += weights.A * result.area
score += weights.C * result.perimeter  # weights.C is <= 0, so this subtracts
if result.second_gate_bonus:
    score += weights.G
return score
```

Directly off the spec: 1000 baseline, `+A*area` (bigger is better),
`+C*perimeter` where `C` is non-positive (so more/longer pieces cost you),
and `+G` for a second gate on a different face. If `build_enclosure()`
returns `None`, `Game.play()` never calls `score_construction` at all — it
short-circuits to a score of `0.0` directly (see `src/game.py`, the
`if construction is None:` branch).

## 12. `Weights` and `Scenario`: what you're handed at the start

`Weights` (`src/weights.py`) is just the three numbers, with sanity checks
baked into construction: `C` must be `<= 0` and `G` must be `>= 0`
(`__post_init__` raises `ValueError` otherwise) — so if you ever see a
`Weights` object in a scenario, you can rely on those signs without
re-checking them yourself.

`Scenario` (`src/scenario.py`) bundles a `Room` + `Inventory` + `Weights`
and is what actually gets (de)serialized to the JSON files in `scenarios/`.
Look at `scenarios/simple_rectangle.json` for the shape:

```json
{
  "room": [[0, 0], [40, 0], [40, 30], [0, 30]],
  "inventory": {
    "walls": {"5": 6, "8": 4},
    "gates": {"6": 2},
    "connectors": {"straight": 12, "right": 12, "diagonal": 8}
  },
  "weights": {"A": 2.0, "C": -3.0, "G": 100.0}
}
```

If no `--scenario` file is passed on the command line, `generate_scenario`
builds a random one instead (`generate_room` + `generate_inventory`, both
seeded from `--seed` so runs are reproducible).

## 13. The `Player` contract

`players/player0.py`:

```python
class Player(ABC):
    def __init__(self, room: Room, inventory: Inventory, weights: Weights) -> None:
        self.room = room
        self.inventory = inventory
        self.weights = weights

    @abstractmethod
    def build_enclosure(self) -> Construction | None:
        pass
```

That's the entire interface. `self.room`, `self.inventory`, and
`self.weights` are populated before `build_enclosure` is ever called, and
they are **your own private copies** (see §15 for exactly how `Game`
constructs them) — freely mutate `self.inventory` for your own bookkeeping
as you plan; it has no effect on scoring.

Your `build_enclosure` should return:

- a `Construction` you believe is valid (it will be independently
  re-checked — see §15 — so there's no benefit to "cheating" here, only
  risk if you're wrong), or
- `None` if you don't believe a valid one exists (or you're choosing not to
  risk one) — this scores 0, safely, no matter what.

`PlayerException` is available if you want to signal "something about my
own logic is broken" distinctly from a bare `Exception`, though
`Game.play()` (§15) treats both the same way: it prints a short message and
**re-raises**. A player that throws crashes the whole run — there is no
silent "exception means 0 points" fallback. If there's any code path where
you're unsure whether you have a valid answer, `return None` rather than
letting an exception escape.

### The time limit

Constructing your player and `build_enclosure()` together get **300 seconds of
CPU time** per run (`--cpu-limit` changes it). Past that your player is stopped
mid-run and scores 0, as if it had returned `None`. Things worth knowing:

- It is **CPU** time, not time on the clock: waiting costs nothing, several
  busy threads add up. Each run prints `player CPU time: ...` so you can see
  how close you are.
- Your constructor counts. Precomputing in `__init__` doesn't buy extra time.
- The interrupt is a `PlayerTimeout`, a `BaseException` (like
  `KeyboardInterrupt`), so your own `except Exception:` won't swallow it. Don't
  catch `BaseException` or use a bare `except:` around your search; if you do
  swallow it you are still caught when you return, but you've wasted the run.
- Only the simulator's own process is counted: CPU used by other processes you
  start (`multiprocessing`, `joblib`) is not.
- Return `None` early rather than searching on when you know you can't win:
  it scores 0, an invalid answer scores -1000, and a timeout scores 0.

## 14. Worked example: reading `ExamplePlayer`

`players/example_player.py` is the one worked example in the repo. It only ever
builds rectangles, but it's real, runnable code that touches every piece of
machinery described above. Run it with `--player e`. The idea: work out *what*
the inventory can build, then search for *where* in the room it fits.

```python
def build_enclosure(self) -> Construction | None:
    if self.lacks_connectors(ConnectorType.RIGHT, 4):
        return None

    for _, gate_len, other_len in self._candidate_shapes()[:MAX_SHAPES_TO_TRY]:
        pieces = self._pieces_for(gate_len, other_len)
        if pieces is None:
            continue  # shouldn't happen, _candidate_shapes already checked
        construction = self._find_placement(pieces)
        if construction is not None:
            return construction

    return None
```

The first thing it does is refuse to start if the inventory has fewer than 4
`RIGHT` connectors: every rectangle needs exactly four, so without them nothing
below can succeed. `lacks_connectors` (on the `Player` base class, worth using in
your own player) returns `True` in that case *and* sets `self.note` to say why,
which the GUI shows and the console prints, so a refusal is explained rather than
just silently scoring 0. Then it loops over candidate shapes, biggest first, and
returns the first one that can be placed.

```python
def _pieces_for(self, gate_len: int, other_len: int) -> list[Piece] | None:
    lengths = [gate_len, other_len, gate_len, other_len]
    scratch = self.inventory.copy()
    pieces = []
    for i, length in enumerate(lengths):
        piece_type = PieceType.GATE if i == 0 else PieceType.WALL
        piece = Piece(piece_type, length)
        if not scratch.take_piece(piece):
            return None
        pieces.append(piece)
    return pieces
```

This is the "simulate against a scratch copy" pattern from §5: build the four
`Piece`s, try to `take_piece` each one from `scratch` (a throwaway copy, so
`self.inventory` is never touched), and give up the moment the inventory can't
support the combination. The sides are `[gate, other, gate, other]` because opposite
sides of a rectangle must be equal (§7's worked example: that is what makes a
4-piece, all-`RIGHT`-connector loop close). If a length is needed three times, say
for a square, the third `take_piece` is the one that fails, with no special case.

`_candidate_shapes` runs `_pieces_for` over every (gate length, wall length) pair
the inventory has, skips any rectangle whose diagonal is longer than the room's
bounding box (it can't fit at any rotation), and sorts what's left by area, largest
first: `A` is never negative, so more area never hurts.

`_rectangle_at` and `_find_placement` are the geometry half: turning "a rectangle of
this size, centered here, turned this way" into a `Construction`, and searching the
room's centroid and then a grid of positions, at a sweep of rotations, for one that
`validate_construction` accepts. That is explained step by step in
[PLACEMENT_GUIDE.md](PLACEMENT_GUIDE.md).

Two things worth noticing. It asks `validate_construction` -- *the exact same function
the simulator scores you with* (§9) -- whether each candidate fits, so it can never
return something the simulator rejects; and if nothing fits it returns `None`, a safe
0 rather than a -1000 guess.

It is deliberately simple. A player beyond it would consider non-rectangular and
non-convex enclosures (reflex connectors, §7), and think harder about the `A`/`C`/`G`
weights when choosing between valid candidates rather than stopping at the first (or
biggest) one found.

## 15. How `Game` wires everything together

`src/game.py`, `Game.__init__`:

```python
self.scenario = self.load_scenario()  # from --scenario file, or generated from --seed

if not self.args.sandbox:
    self.player = (self.get_player_class())(
        room=self.scenario.room.copy(),
        inventory=self.scenario.inventory.copy(),
        weights=self.scenario.weights,
    )
```

This is the detail from §5/§13 made concrete: **the player is constructed
with `.copy()`s of the room and inventory**, not the same objects `Game`
holds in `self.scenario`. Whatever your player does to `self.inventory`
internally during planning is invisible to the rest of the system.

`Game.play()` is what actually runs a turn:

```python
def play(self):
    construction = (
        self.player.build_enclosure()
    )  # (wrapped in try/except that re-raises)
    self.construction = construction

    if construction is None:
        self.score = 0.0
        return

    self.result = validate_construction(
        construction,
        self.scenario.room,
        self.scenario.inventory,  # <- the ORIGINALS, not the player's copies
    )
    self.score = score_construction(self.result, self.scenario.weights)
```

This is the other half of the same detail: the construction your player
returns is validated against `self.scenario.room` / `self.scenario.inventory`
— the pristine originals — regardless of what state your player's own
copies ended up in. There is exactly one source of truth for scoring, and
it's untouched by anything your player did along the way.

In GUI mode (`--gui`, or `--sandbox` which implies it), `Game` also builds a
tkinter window: a canvas on the left showing the room (and, once you play,
your enclosure drawn on top of it — green if valid, red if invalid, gates
drawn with a visible gap), and an info panel on the right showing the
inventory, weights, and a live score breakdown after each run. `--sandbox`
mode skips instantiating a player entirely and just shows you the room and
inventory for a scenario — useful for eyeballing what you're working with
before writing any strategy.

## 16. Common pitfalls

- **Miscounting connectors.** `len(connectors) == len(pieces)` always,
  because every piece has exactly one vertex "before" it with a connector.
  A common mistake is providing one connector per *turn you think you're
  making*, forgetting that `connectors[0]` — the closing joint — still has
  to be there and has to be correct (§7, §8).
- **Forgetting `reflex`.** If your shape needs to turn the "other way" at
  some vertex (any non-convex enclosure), you need
  `Connector(type, reflex=True)`, not a different `ConnectorType`. Using a
  plain `RIGHT` where you needed a reflex `RIGHT` will silently produce a
  self-intersecting or non-closing shape rather than an obvious error.
- **Two gates, same face, expecting the bonus.** The bonus requires two
  *different* faces (§10) — splitting one wall into two gate-halves with a
  `STRAIGHT` connector between them does not qualify.
- **Treating inventory counts as per-orientation.** `RIGHT` and its reflex
  flip share one pool; don't budget for them as if they were separate
  resources.
- **Not checking containment early.** `validate_construction` does real
  polygon work (simplicity, containment, face-length checks) — if you're
  searching over many candidate placements, `self.room.contains_enclosure`
  on a quick `Polygon(build_geometry(construction)[0])` is a much cheaper
  first filter than running full validation on every candidate.
- **Letting exceptions escape.** `Game.play()` re-raises anything your
  player throws; wrap risky logic and return `None` on failure rather than
  letting a crash end the run.
- **Assuming your inventory bookkeeping affects your score.** It doesn't
  (§15) — only the final `Construction` you return matters, checked from
  scratch.

## 17. Adding your own player

Your group's folder, `players/player<N>/`, already holds an empty player:

1. Open `players/player<N>/player.py` and fill in `build_enclosure` of
   `class Player<N>`. Add more files to the same folder if you like, and import
   them by full path (`from players.player3.helpers import ...`).
2. Run it: `uv run main.py --gui --player <N> --scenario scenarios/simple_rectangle.json`.
   Nothing needs registering; `--player <N>` finds `Player<N>` on its own.
3. Before a pull request: `uv run python -m scripts.check_submission <N>`.
   README.md has the submission rules.
4. Run `uv run main.py --gui --player i --scenario scenarios/invalid_examples.json`
   and click "Build enclosure" repeatedly (or pick one mistake directly with
   `--player i1` ... `--player i10`; README has the list): each click shows a different way to
   get a construction rejected (doesn't close, outside the room, walls cross,
   wall cuts across a notch in a non-convex room, ...) and the exact reason
   the validator gives. `demos/invalid_examples.py` shows how each
   one is built, one broken rule at a time.
5. Read `tests/test_enclosure.py` for more worked examples of exactly what
   passes and fails validation, and consider adding your own tests there in
   the same style while developing — it's much faster to iterate against
   `validate_construction` directly in a unit test than to click "Build
   enclosure" in the GUI repeatedly.
