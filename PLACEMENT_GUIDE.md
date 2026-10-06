# Placing an enclosure in the room: getting started

This guide is about the geometry half of the project: **given a shape you want
to build, where in the room can it go?** It starts from nothing, and every code
block runs as written, in order, so you can paste them into one file and play with
them. (`SIMULATOR_GUIDE.md` covers the rest of the simulator: inventory, scoring, the
rules. `players/example_player.py` is a complete strategy to read afterwards.)

By the end you will be able to:

- turn a `Construction` into the actual corner coordinates, by hand and in code;
- build a rectangle anywhere, at any angle, and test whether it fits the room;
- explain why "all my corners are inside the room" is not enough in a non-convex room;
- search for a placement systematically instead of by luck;
- build a shape that isn't a rectangle.

## 1. What you are actually producing

Your player doesn't return a polygon. It returns a **recipe**: a starting point, a
starting direction, the walls and gates in order, and the connector at each corner.

```text
Construction(start=(x, y), start_heading=degrees, pieces=[...], connectors=[...])
```

The simulator walks the recipe to get the corners, then checks everything. So the
geometry question is: *which `start` and `start_heading` make the shape land inside
the room?* The simulator is the judge, and you can call the same judge yourself
(§4), so you never have to guess.

## 2. Coordinates and angles

- Positions are in **units**, the same units as wall lengths. A 10-unit wall is 10
  units long in the room.
- `x` grows to the right and `y` grows **up**; an angle is in **degrees**, measured
  counter-clockwise from the +x direction. Heading 0 points along +x, 90 along +y,
  180 along -x, 270 along -y. Python's `math` functions use **radians**, so convert
  with `math.radians(...)`.
- The GUI draws the room with `y` growing *downward* on the screen (that's how
  windows work), so a picture can look flipped compared to a graph. Don't trust your
  eyes for direction; **hover over a joint** in the GUI to see its true coordinates.
- A connector is a *turn*. A `RIGHT` connector (90°) turns the path 90° **left**
  (counter-clockwise); flipped (`reflex=True`) it turns 90° the other way. A
  `DIAGONAL` turns 45°, and `STRAIGHT` doesn't turn at all.

## 3. From a Construction to corners

Walk a 10 x 6 rectangle by hand. Start at (5, 5) heading 0 (east). Each `RIGHT`
connector turns 90° left before the next wall:

| wall | heading | direction | ends at |
| :-- | :-- | :-- | :-- |
| 0: 10 units | 0° | east | (15, 5) |
| 1: 6 units | 90° | north | (15, 11) |
| 2: 10 units | 180° | west | (5, 11) |
| 3: 6 units | 270° | south | (5, 5), back where it started |

Now the same thing in code. `build_geometry` does the walk for you:

```python
import math

from shapely.geometry import Point, Polygon

from src.enclosure import Construction, build_geometry, validate_construction
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room

RIGHT = ConnectorType.RIGHT
room = Room(Polygon([(0, 0), (40, 0), (40, 30), (0, 30)]))  # a 40 x 30 room

rect = Construction(
    start=(5, 5),
    start_heading=0,
    pieces=[
        Piece(PieceType.GATE, 10),
        Piece(PieceType.WALL, 6),
        Piece(PieceType.WALL, 10),
        Piece(PieceType.WALL, 6),
    ],
    connectors=[Connector(RIGHT)] * 4,
)

vertices, position_gap, heading_gap = build_geometry(rect)
print([(round(x, 3), round(y, 3)) for x, y in vertices])
# [(5, 5), (15.0, 5.0), (15.0, 11.0), (5.0, 11.0)]
print(position_gap < 1e-9, heading_gap)  # True 0.0  -> the loop closes
```

`vertices` are the corners. The two gaps say how far the loop is from closing; the
simulator rejects a loop that doesn't. The position gap here is about 2e-15, not exactly
0: that's floating-point rounding in the sines and cosines, which the simulator
tolerates up to 1e-4 units. (It's also why the corners above are printed rounded: the
last one is really `11.000000000000002`.) Never compare floats with `==`.

## 4. Is it OK? Three questions, from cheap to complete

Make a shape out of the corners, and ask the room whether it fits:

```python
def shape_of(construction):
    """The shapely Polygon a Construction would build."""
    return Polygon(build_geometry(construction)[0])


print(room.contains_enclosure(shape_of(rect)))  # True
```

`room.contains_enclosure` answers one question only: does the shape lie inside the
room? It is cheap, so use it inside your search loops. Walls may lie *exactly along*
the room's walls: it allows a tiny tolerance (1e-4 units), so a flush fit counts.

The complete judge is `validate_construction`. It checks everything the simulator
will check (the loop closes, the inventory has the pieces, there is a gate, the walls
don't cross, the shape fits the room, no straight run is over 30 units) and tells you
**why** when it says no:

```python
inventory = Inventory(walls={6: 4, 10: 4}, gates={10: 1}, connectors={RIGHT: 8})

result = validate_construction(rect, room, inventory)
print(result.valid, result.area, result.perimeter)  # True 60.0 32.0

# take away the gate and ask again
no_gate = Construction(
    rect.start,
    rect.start_heading,
    [Piece(PieceType.WALL, p.length) for p in rect.pieces],
    rect.connectors,
)
print(validate_construction(no_gate, room, inventory).reason)
# enclosure must have at least one gate
```

When a candidate is rejected and you don't know why, print `.reason`. Run
`uv run main.py --gui --player i --scenario scenarios/invalid_examples.json` and click through
ten different ways to be invalid, each with the reason the validator gives.

## 5. Putting a rectangle where you want it

A rectangle has width `w` (piece 0) and height `h` (piece 1). If it starts at `start`
with heading `theta`, let `u` be the direction piece 0 points and `v` the direction
piece 1 points (90° to the left of `u`):

```text
u = (cos θ, sin θ)        v = (-sin θ, cos θ)
corners:  start,   start + w·u,   start + w·u + h·v,   start + h·v
```

Putting the *centre* where you want it is the same formula turned around:
`start = centre - (w·u + h·v) / 2`. As code:

```python
def rectangle(start, heading_deg, width, height):
    """A rectangle: a gate of `width`, then walls of `height`, `width`, `height`."""
    pieces = [
        Piece(PieceType.GATE, width),
        Piece(PieceType.WALL, height),
        Piece(PieceType.WALL, width),
        Piece(PieceType.WALL, height),
    ]
    return Construction(start, heading_deg, pieces, [Connector(RIGHT)] * 4)


def rectangle_at(center, heading_deg, width, height):
    """The same rectangle, positioned by its centre instead of its first corner."""
    t = math.radians(heading_deg)
    ux, uy = math.cos(t), math.sin(t)
    vx, vy = -uy, ux
    start = (
        center[0] - (width * ux + height * vx) / 2,
        center[1] - (width * uy + height * vy) / 2,
    )
    return rectangle(start, heading_deg, width, height)


tilted = shape_of(
    rectangle_at((20, 15), 30, 12, 8)
)  # centred in the room, turned 30 degrees
print(tilted.centroid.coords[0], tilted.area)  # (20.0, 15.0) 96.0
print(room.contains_enclosure(tilted))  # True

print(
    room.contains_enclosure(shape_of(rectangle((0, 0), 0, 12, 8)))
)  # True: flush in the corner
print(
    room.contains_enclosure(shape_of(rectangle((35, 5), 0, 12, 8)))
)  # False: pokes out of the right side
```

Don't round the coordinates you compute. The loop has to close to within 1e-4 units,
and rounding `start` to a couple of decimals can break that. Let the floats be floats.

## 6. "All my corners are inside" is not enough

In a plain rectangular room, if every corner of your shape is inside the room, the
whole shape is. A room with a notch in it (an L, a U, anything non-convex) breaks that:
a wall can join two corners that are both inside and still cut across the outside.

```python
# an L-shaped room: a 40 x 15 bottom arm and a 15 x 40 left arm. The square
# above the bottom arm and right of the left arm (x > 15, y > 15) is NOT floor.
l_room = Room(Polygon([(0, 0), (40, 0), (40, 15), (15, 15), (15, 40), (0, 40)]))

bridge = Polygon(
    [(8, 33), (33, 8), (36, 11), (11, 36)]
)  # a slanted bar from one arm to the other

print([l_room.polygon.contains(Point(p)) for p in bridge.exterior.coords[:-1]])
# [True, True, True, True]   <- every corner is inside the room
print(l_room.contains_enclosure(bridge))
# False                      <- but the bar crosses the notch
```

So **always test the whole shape** with `contains_enclosure` (or `validate_construction`),
never just its corners. This is example 9 in the invalid-enclosure demo
(`--player i9 --scenario scenarios/invalid_examples.json`).

## 7. Where to look: searching for a placement

You can't try every position and angle (there are infinitely many), so a placement
search tries a sensible finite set. From simplest to smartest:

### 7.1 Start at the centre, and know its trap

For a convex room, the centroid is inside, so "put the shape's centre on the room's
centroid and turn it until it fits" works. For a non-convex room the centroid can lie
**outside** the room:

```python
u_room = Room(
    Polygon(
        [(0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (10, 10), (10, 30), (0, 30)]
    )
)

centroid = u_room.polygon.centroid
print((round(centroid.x, 3), round(centroid.y, 3)), u_room.polygon.contains(centroid))
# (15.0, 13.571) False         <- in the gap between the U's two arms

inside = u_room.polygon.representative_point()
print((round(inside.x, 3), round(inside.y, 3)), u_room.polygon.contains(inside))
# (5.0, 20.0) True             <- shapely's "a point guaranteed to be inside"
```

`polygon.representative_point()` is the quick fix when you just need *a* point that is
in the room. Filtering candidate points with `polygon.contains(Point(x, y))` before
trying anything at them is a cheap way to skip hopeless ones.

### 7.2 Sweep angles and positions

`players/example_player.py` tries the centroid at every 15° and then a grid of points
across the room's bounding box, again at every 15°. It's easy to write and fine for
roomy rooms. Its weakness is the *step*: if the only angle that works falls between
two of your steps, the sweep misses it.

### 7.3 Use the room's own geometry

A good placement almost always **touches the room's walls and lines up with them**. So
instead of sweeping angles blindly, take the angles from the room's own edges, and
instead of a grid, put one corner of your shape on each corner of the room:

```python
def edge_headings(poly):
    """The distinct edge directions of a polygon, in degrees, modulo 90."""
    points = list(poly.exterior.coords)
    return sorted(
        {
            round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90, 9)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        }
    )


def corner_placements(room, width, height):
    """Every (start, heading) that puts one corner of a width x height rectangle on a
    corner of the room, lined up with one of the room's edges."""
    room_corners = list(room.polygon.exterior.coords)[:-1]
    for base in edge_headings(room.polygon):
        for heading in (base, base + 90, base + 180, base + 270):
            t = math.radians(heading)
            ux, uy, vx, vy = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
            # where each of the rectangle's four corners sits relative to `start`
            offsets = [
                (0, 0),
                (width * ux, width * uy),
                (width * ux + height * vx, width * uy + height * vy),
                (height * vx, height * vy),
            ]
            for px, py in room_corners:
                for ox, oy in offsets:
                    yield (px - ox, py - oy), heading


def find_placement(room, width, height):
    for start, heading in corner_placements(room, width, height):
        candidate = rectangle(start, heading, width, height)
        if room.contains_enclosure(shape_of(candidate)):
            return candidate
    return None
```

Here it is on a room turned 33°, with a rectangle that only just fits (a 41 x 31 room
and a 40 x 30 rectangle, half a unit to spare each way):

```python
from shapely import affinity

tilted_room = Room(
    affinity.rotate(Polygon([(0, 0), (41, 0), (41, 31), (0, 31)]), 33, origin=(0, 0))
)

print(edge_headings(tilted_room.polygon))  # [33.0]
found = find_placement(tilted_room, 40, 30)
print(found.start_heading)  # 33

centre = tilted_room.polygon.centroid
sweep = lambda step: [
    h
    for h in range(0, 180, step)
    if tilted_room.contains_enclosure(
        shape_of(rectangle_at((centre.x, centre.y), h, 40, 30))
    )
]
print(sweep(15))  # []            <- a 15-degree sweep finds nothing
print(sweep(1))  # [32, 33, 34]  <- even a 1-degree sweep only just does
```

Reading the room's edges finds the answer on the first try; sweeping has to get lucky.
The same idea scales up: also try shifting the shape along a wall, or growing it until
it stops fitting.

### 7.4 Which shape, and how big?

The shape you place comes from the inventory (what lengths, and how many of each, do
you have?). Try the biggest area first, since the score rewards area, and skip shapes
that can't fit at all: a rectangle whose diagonal is longer than the room's bounding
box diagonal fits nowhere, at any angle. `SIMULATOR_GUIDE.md` §14 walks through how
`players/example_player.py` does exactly this.

## 8. Beyond rectangles

Rectangles are the easy case: four `RIGHT` connectors and two pairs of equal sides. The
loop can be any simple polygon. **Every** corner needs a connector, and a closed loop's
turns always add up to 360° (a reflex corner counts as -90°). For an L-shape:

```text
(10,30) ←10← (20,30)
   │             ↓ 10
  20             (20,20) ←10← (30,20)
   │                              ↓ 10
(10,10) ───────── 20 ──────────→ (30,10)
```

```python
reflex = Connector(RIGHT, reflex=True)  # a 270° inside corner: turns the other way

l_shape = Construction(
    start=(10, 10),
    start_heading=0,
    pieces=[
        Piece(PieceType.GATE, 20),  # east
        Piece(PieceType.WALL, 10),  # north
        Piece(PieceType.WALL, 10),  # west
        Piece(PieceType.WALL, 10),  # north  <- turns right, away from the usual left
        Piece(PieceType.WALL, 10),  # west
        Piece(PieceType.WALL, 20),  # south
    ],
    # connectors[i] is the turn at the START of piece i (between piece i-1 and piece i)
    connectors=[
        Connector(RIGHT),
        Connector(RIGHT),
        Connector(RIGHT),
        reflex,
        Connector(RIGHT),
        Connector(RIGHT),
    ],
)

print(sum(c.turn_angle() for c in l_shape.connectors))  # 360.0
print([(round(x, 3), round(y, 3)) for x, y in build_geometry(l_shape)[0]])
# [(10, 10), (30.0, 10.0), (30.0, 20.0), (20.0, 20.0), (20.0, 30.0), (10.0, 30.0)]

stock = Inventory(walls={10: 6, 20: 4}, gates={20: 1}, connectors={RIGHT: 8})
result = validate_construction(l_shape, room, stock)
print(result.valid, result.area)  # True 300.0
```

An L-shaped enclosure fits an L-shaped room where a rectangle wastes the corner, which is
where a higher score comes from. Other ideas, all using what the simulator already
checks for you:

- **Long sides from several walls.** Join walls with `STRAIGHT` connectors to make one
  straight face (up to 30 units long) when no single wall is the right length.
- **45° corners.** `DIAGONAL` connectors turn 45°, so eight turns make an octagon; useful
  for squeezing into awkward corners.
- **A second gate on a different face** earns the `G` bonus (not two gates on the same
  straight run).

## 9. Debugging a placement

- **`validate_construction(c, room, inventory).reason`** says what's wrong, in order.
- **The GUI.** `uv run main.py --gui --player N --scenario ...` draws the result. Hover a
  corner for its coordinates and connector. A wall you don't have is drawn as a pale
  ghost, and a loop that doesn't close shows a red gap.
- **`--debug`** shows each wall's length on the drawing.
- **`self.note = "..."`** in your player puts a line in the GUI's result panel and the console.
- **`print`** is fine; everything printed also goes to the run's log in `logs/`.
- **Test offline, with no window.** This runs your placement code on any scenario:

```python
from src.scenario import read_scenario

scenario = read_scenario("scenarios/l_shaped_room.json")
print(scenario.room.polygon.bounds)  # (minx, miny, maxx, maxy) of this room
print(
    find_placement(scenario.room, 12, 8) is not None
)  # does a 12 x 8 rectangle fit anywhere?
```

## 10. Common mistakes

| Mistake | What happens |
| :-- | :-- |
| Degrees where `math.sin` wants radians (or the reverse) | Shapes at the wrong angle; closing errors |
| Rounding computed coordinates | The loop stops closing (tolerance is 1e-4) |
| Testing only the corners against the room | Passes, then fails on a non-convex room (§6) |
| Assuming the centroid is inside the room | Fails for U, L and C shaped rooms (§7.1) |
| Reading the GUI's y direction as up | A shape that looks flipped (hover for real coordinates) |
| One connector per *turn* instead of per *corner* | "need exactly one connector per vertex" |
| Opposite sides of a rectangle not equal | "enclosure does not close" |
| Returning your best guess instead of `None` | An invalid guess is -1000; `None` is 0 |

## 11. A suggested path

1. A rectangle placed in a rectangular room (`scenarios/simple_rectangle.json`).
2. The same, in the non-convex room (`scenarios/l_shaped_room.json`).
3. Choose the *biggest* rectangle your inventory allows that fits.
4. Use a pair of walls to make a longer side when no single wall fits.
5. Try shapes that follow the room's notches (§8).
6. Add the second gate, and weigh `A`, `C` and `G` against each other.

Run `--player e` to see where a plain rectangle search gets you, then beat it.
