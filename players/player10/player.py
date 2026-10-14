import heapq
import math
import time

import src.constants as c
from players.player0 import Player
from players.player10.assign import Assigner, Assignment
from players.player10.freeform import Faces, Grower, Loop
from players.player10.placement import RoomSearch
from players.player10.shapes import (
    CORNERS_NEEDED,
    HEADING_STEP_DEG,
    HEXAGON,
    L_SHAPE,
    MEASURES,
    OCTAGON,
    RECTANGLE,
    SYMMETRY_DEG,
    U_SHAPE,
    Shape,
    contains_smaller,
    free_shape,
    headed_faces,
    inner_radius,
    inner_rectangles,
    make_shape,
)
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

# CPU seconds the search may use. Nothing can interrupt a player on Windows, so
# the player has to stop itself well inside the simulator's limit.
TIME_BUDGET = 42.0
# how far into the budget each stage of the search may run; growing gets the rest
RECTANGLE_SHARE = 0.12
L_SHARE = 0.25
OCTAGON_SHARE = 0.42
U_SHARE = 0.55
# growing asks for pieces for thousands of loops, so each ask gets less effort
GROW_ASSIGN_NODES = 3000
MAX_SEEDS = 6
# the best rectangle is also grown from other places it fits, this many at most
EXTRA_SPOTS = 2
SPOTS_TRIED = 24
FALLBACK_BUDGET = 3.0
FALLBACK_SHAPES = 40
RECENT_FAILURES = 400
SHRINK_STEPS = 4

Candidate = tuple[float, str, tuple[int, ...]]


class Player10(Player):
    """Group 10: the best-scoring rectangle, L, U, hexagon or octagon the inventory
    can build, placed against the room's own walls, then grown face by face into
    a free-form loop that fills the room."""

    def build_enclosure(self) -> Construction | None:
        self._best: Construction | None = None
        self._best_score = 0.0  # a valid enclosure can score below 0; None is 0
        try:
            self._build()
        except Exception as e:  # noqa: BLE001 -- a crash would end the whole run
            self.note = f"search stopped by {type(e).__name__}: {e}"
        return self._best

    def _build(self) -> None:
        started = time.process_time()
        weights = self.weights
        self._assigner = Assigner(self.inventory, want_second_gate=weights.G > 0)
        if self._assigner.gates == 0:
            self.note = "no usable gate in the inventory -- not attempting a build"
            return
        self._search_room = RoomSearch(self.room)
        self._failures: dict[str, list[tuple[int, ...]]] = {}
        self._fitting_rectangles: list[tuple[int, int]] = []
        self._seeds: list[Loop] = []
        self._rectangle: tuple | None = None
        bonus = weights.G if len(self._assigner.targets) == 2 else 0.0
        self._bonus = bonus
        stock = self.inventory.connectors
        families = [
            family
            for family, needs in CORNERS_NEEDED.items()
            if all(stock.get(kind, 0) >= n for kind, n in needs.items())
        ]

        # cheapest families first, so there is always something to fall back on
        # and a score for the dearer searches to beat. Octagons are slow to rule
        # out; U-shapes are many, and only pay off where nothing else fits.
        stages = [
            ([RECTANGLE], RECTANGLE_SHARE),
            ([L_SHAPE], L_SHARE),
            ([OCTAGON, HEXAGON], OCTAGON_SHARE),
            ([U_SHAPE], U_SHARE),
        ]
        for stage, share in stages:
            wanted = [f for f in stage if f in families]
            if wanted:
                self._deadline = started + TIME_BUDGET * share
                self._search([x for f in wanted for x in self._candidates(f, bonus)])
        self._grow(started + TIME_BUDGET)
        if self._best is None and not self.note:
            self.note = "nothing the inventory can build fits and scores above 0"

    # ------------------------------------------------------------------ grow

    def _grow(self, deadline: float) -> None:
        """Grow the shapes found so far into free-form loops (see freeform.py)."""
        if not self._seeds:
            return
        stock = self.inventory.connectors
        grower = Grower(
            region=self._search_room.region,
            room_corners=self._search_room.corners,
            buildable=set(self._assigner.lengths),
            rights=stock.get(ConnectorType.RIGHT, 0),
            diagonals=stock.get(ConnectorType.DIAGONAL, 0),
            total_length=self._assigner.total_length,
            score=lambda area, perimeter: self._score(area, perimeter) + self._bonus,
            accept=self._accept_loop,
            can_build=lambda lengths: self._assigner.assign(lengths) is not None,
        )
        self._assigner.max_nodes = GROW_ASSIGN_NODES
        seeds = sorted(self._seeds, key=lambda loop: -loop.score)[:MAX_SEEDS]
        seeds += self._other_spots()
        # the charge for a side over the limit starts below what a unit of
        # growth earns, so spreading out comes first and repairs second
        grower.grow(seeds, deadline, first_charge=max(self.weights.A, 0.1))

    def _other_spots(self) -> list[Loop]:
        """The best rectangle again, at other places it fits: a room with
        several open areas gets a seed growing in each."""
        if self._rectangle is None:
            return []
        shape, first, score = self._rectangle
        a, b = shape.params

        def centre(start, heading):
            t = math.radians(heading)
            return (
                start[0] + (a * math.cos(t) - b * math.sin(t)) / 2.0,
                start[1] + (a * math.sin(t) + b * math.cos(t)) / 2.0,
            )

        taken = [centre(*first)]
        spots = []
        for start, heading in self._placements(shape, limit=SPOTS_TRIED):
            here = centre(start, heading)
            if all(math.dist(here, other) >= max(a, b) for other in taken):
                taken.append(here)
                spots.append(Loop(start, heading, headed_faces(shape), score))
                if len(spots) >= EXTRA_SPOTS:
                    break
        return spots

    def _accept_loop(
        self, start: tuple[float, float], heading: float, faces: Faces
    ) -> float | None:
        """The true score of a loop, if the inventory can build it and the
        simulator accepts it; the best one seen is kept."""
        shape = free_shape(faces)
        assignment = self._assigner.assign(shape.faces)
        if assignment is None:
            return None
        # a loop's faces are headed from the loop's own heading, and its first
        # face need not point along it
        first = heading + 45.0 * faces[0][0]
        construction = self._construction(shape, assignment, start, first)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return None
        score = score_construction(result, self.weights)
        if score > self._best_score:
            self._best, self._best_score = construction, score
            self.note = f"{shape.describe()}, score {score:.1f}"
        return score

    # ------------------------------------------------------------ candidates

    def _candidates(self, family: str, bonus: float) -> list[Candidate]:
        """Every shape of a family worth trying, with the most it could score."""
        lengths = self._assigner.lengths
        usable = set(lengths)
        if family == RECTANGLE:
            params = [(a, b) for a in lengths for b in lengths if b <= a]
        elif family == OCTAGON:
            # p is the longest side: the other choices are the same octagon turned
            params = [
                (p, r, q, t)
                for p in lengths
                for r in lengths
                if r <= p
                for q in lengths
                if q <= p
                for t in lengths
                if t <= p
            ]
        elif family == HEXAGON:
            params = [(a, d1, d2) for a in lengths for d1 in lengths for d2 in lengths]
        elif family == U_SHAPE:
            params = [
                (left, gap, right, height, leg)
                for left in lengths
                for gap in lengths
                for right in lengths
                if left + gap + right in usable
                for height in lengths
                for leg in lengths
                if height - leg >= c.MIN_WALL_LENGTH
            ]
        else:
            params = [
                (a, b, cut, d)
                for a in lengths
                for cut in lengths
                if a - cut in usable
                for b in lengths
                for d in lengths
                if b + d in usable
            ]
        measure = MEASURES[family]
        out = []
        for p in params:
            area, perimeter = measure(*p)
            if area > self._search_room.area + c.TOL:
                continue
            score = self._score(area, perimeter) + bonus
            if score > self._best_score:
                out.append((score, family, p))
        return out

    def _score(self, area: float, perimeter: float) -> float:
        return c.BASELINE_SCORE + self.weights.A * area + self.weights.C * perimeter

    # ---------------------------------------------------------------- search

    def _search(self, candidates: list[Candidate]) -> None:
        """Try candidates from the highest score down; the first that can be built
        and placed is the best of them."""
        # (-score, tie-break, family, params, score is exact)
        heap = [(-s, i, f, p, False) for i, (s, f, p) in enumerate(candidates)]
        heapq.heapify(heap)
        counter = len(heap)
        timed_out = False

        while heap:
            if time.process_time() > self._deadline:
                timed_out = True
                break
            negative, _, family, params, exact = heapq.heappop(heap)
            if -negative <= self._best_score:
                return
            if self._cannot_fit(family, params):
                continue
            shape = make_shape(family, params)
            assignment = self._assigner.assign(shape.faces)
            if assignment is None:
                continue
            if not exact:
                score = self._score(shape.area, shape.perimeter)
                if assignment.gate_faces >= 2:
                    score += self.weights.G
                if score < -negative - 1e-9:
                    # no second gate after all: it is worth less than it was
                    # ranked at, so let it wait its turn
                    counter += 1
                    heapq.heappush(heap, (-score, counter, family, params, True))
                    continue
            if self._try(shape, assignment):
                return
            self._note_failure(family, params)

        if timed_out and self._best is None:
            # ran out of time on big shapes: settle for the smallest ones
            deadline = time.process_time() + FALLBACK_BUDGET
            for _, _, family, params, _ in heapq.nlargest(FALLBACK_SHAPES, heap):
                if time.process_time() > deadline:
                    break
                shape = make_shape(family, params)
                assignment = self._assigner.assign(shape.faces)
                if assignment is not None:
                    self._try(shape, assignment)

    def _cannot_fit(self, family: str, params: tuple[int, ...]) -> bool:
        """Cheap reasons a shape can't be placed, tried before the real search."""
        if inner_radius(family, params) > self._search_room.inradius:
            return True
        recent = self._failures.get(family, [])[-RECENT_FAILURES:]
        if any(contains_smaller(family, params, small) for small in recent):
            return True
        return not all(
            self._rectangle_fits(max(r), min(r))
            for r in inner_rectangles(family, params)
        )

    def _rectangle_fits(self, a: int, b: int) -> bool:
        """Whether an a x b rectangle (a >= b) fits anywhere in the room, whether
        or not the inventory can build it. Remembered, and inferred from other
        rectangles where possible."""
        if any(fa >= a and fb >= b for fa, fb in self._fitting_rectangles):
            return True
        failures = self._failures.setdefault(RECTANGLE, [])
        if any(a >= fa and b >= fb for fa, fb in failures):
            return False
        if inner_radius(RECTANGLE, (a, b)) > self._search_room.inradius:
            return False
        fits = self._placements(make_shape(RECTANGLE, (a, b)), limit=1) != []
        (self._fitting_rectangles if fits else failures).append((a, b))
        return fits

    def _try(self, shape: Shape, assignment: Assignment) -> bool:
        """Place the shape; keep it if it is valid and beats the best so far."""
        for start, heading in self._placements(shape):
            construction = self._construction(shape, assignment, start, heading)
            # the simulator's own judge: only what it accepts is ever returned
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                continue
            score = score_construction(result, self.weights)
            self._seeds.append(Loop(start, heading, headed_faces(shape), score))
            if shape.family == RECTANGLE:
                self._rectangle = (shape, (start, heading), score)
            if score > self._best_score:
                self._best, self._best_score = construction, score
                self.note = f"{shape.describe()}, score {score:.1f}"
            return True
        return False

    def _placements(self, shape: Shape, limit: int = 3) -> list:
        return self._search_room.place(
            shape.vertices(),
            HEADING_STEP_DEG[shape.family],
            SYMMETRY_DEG[shape.family],
            limit,
        )

    def _note_failure(self, family: str, params: tuple[int, ...]) -> None:
        """Remember that a shape fits nowhere. Shapes are tried biggest first, so a
        failure on its own rules out little; a much smaller version that also
        fails rules out everything that could hold it."""
        failures = self._failures.setdefault(family, [])
        smallest = params
        if family in (RECTANGLE, OCTAGON):
            fitting, failing = 0.0, 1.0
            for _ in range(SHRINK_STEPS):
                scale = (fitting + failing) / 2.0
                shrunk = tuple(max(c.MIN_WALL_LENGTH, round(scale * p)) for p in params)
                if shrunk == smallest:
                    break
                if family == RECTANGLE:
                    fits = self._rectangle_fits(max(shrunk), min(shrunk))
                else:
                    fits = self._placements(make_shape(family, shrunk), 1) != []
                if fits:
                    fitting = scale
                else:
                    failing, smallest = scale, shrunk
        if smallest not in failures:
            failures.append(smallest)

    @staticmethod
    def _construction(
        shape: Shape,
        assignment: Assignment,
        start: tuple[float, float],
        heading: float,
    ) -> Construction:
        pieces, connectors = [], []
        for face, keys in enumerate(assignment.faces):
            for i, (is_gate, length) in enumerate(keys):
                kind = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(kind, length))
                connectors.append(
                    shape.corner(face) if i == 0 else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(start, heading, pieces, connectors)
