"""Scheduled Player 4 boundary search, global assignment, and growth/repair.

Production search implementation. No raster search.
"""

import math
import time

import numpy as np
import shapely

from players.player0 import Player
from players.player4 import anchors as original
from players.player4.assignment import Assigner
from players.player4.faces import FaceCatalog
from players.player4.growth import Grower, Loop
from players.player4.limits import max_pieces_per_face
from players.player4.macro import MacroBeamSearch, buildable_lengths, should_use_macro
from players.player4.realize import expand_macro_edge, realize_closed
from players.player4.restart_anchors import anchors_from_construction
from players.player4.search import BeamSearch, connector_for
from src.constants import BASELINE_SCORE, TOL
from src.enclosure import (
    Construction,
    compute_faces,
    score_construction,
    validate_construction,
    walk_points,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType


def spread_anchors(anchors, limit=32):
    """Farthest-point sampling in position/heading space, starting at best wall."""
    if len(anchors) <= limit:
        return list(anchors)
    scale = max(math.dist(a.start, b.start) for a in anchors for b in anchors) or 1

    def distance(a, b):
        angle = abs((a.heading_deg - b.heading_deg + 180) % 360 - 180) / 180
        return math.dist(a.start, b.start) / scale + 0.25 * angle

    selected, remaining = [anchors[0]], list(anchors[1:])
    while remaining and len(selected) < limit:
        chosen = max(remaining, key=lambda a: min(distance(a, b) for b in selected))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def construction_loop(construction, score):
    """Express an existing construction in the Grower's local 45-degree frame."""
    faces = compute_faces(construction)
    points = walk_points(construction)
    headings = [construction.start_heading]
    for connector in construction.connectors[1:]:
        headings.append(headings[-1] + connector.turn_angle())
    first = faces[0].piece_indices[0]
    base = headings[first]
    headed = tuple(
        (round((headings[f.piece_indices[0]] - base) / 45) % 8, round(f.length))
        for f in faces
    )
    return Loop(points[first], base, headed, score)


class SearchPlayer(Player):
    """Player 4 boundary search with explicit stage budgets and optional repair."""

    def __init__(self, room, inventory, weights, variant="grown", budget=40.0):
        super().__init__(room, inventory, weights)
        if variant not in ("scheduled", "assigned", "grown"):
            raise ValueError(
                f"experimental variant must be one of {('scheduled', 'assigned', 'grown')}"
            )
        self.variant, self.budget = variant, budget
        self.best, self.best_score = None, 0.0
        self.seeds = []
        self.stages = []
        self.assignment = None

    def accept(self, construction):
        if construction is None:
            return None
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return None
        score = score_construction(result, self.weights)
        if score > 0:
            self.seeds.append(construction_loop(construction, score))
            self.seeds = sorted(self.seeds, key=lambda loop: -loop.score)[:24]
        if score > self.best_score:
            self.best, self.best_score = construction, score
        return score

    def assign_loop(self, start, heading, faces):
        assignment = self.assignment.assign(tuple(length for _, length in faces))
        if assignment is None:
            return None
        pieces, connectors = [], []
        for face, keys in enumerate(assignment.faces):
            for i, (is_gate, length) in enumerate(keys):
                pieces.append(
                    Piece(PieceType.GATE if is_gate else PieceType.WALL, length)
                )
                connectors.append(
                    connector_for((faces[face][0] - faces[face - 1][0] + 3) % 8 - 3)
                    if i == 0
                    else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(start, heading + faces[0][0] * 45, pieces, connectors)

    def macro(self, anchors, deadline):
        lengths = tuple(self.catalog.lengths)
        realizable = buildable_lengths(lengths)
        for i, anchor in enumerate(anchors):
            now = time.process_time()
            if now >= deadline:
                break
            search = MacroBeamSearch(
                anchor,
                self.catalog,
                self.connectors,
                self.weights,
                self.region,
                40,
                now + (deadline - now) / (len(anchors) - i),
                realizable,
                min(lengths),
                min(lengths),
            )
            for closed in search.run():
                if time.process_time() >= deadline:
                    break
                if self.assignment is None:
                    construction = realize_closed(
                        anchor,
                        closed,
                        self.catalog,
                        self.connectors,
                        self.weights,
                        self.region,
                    )
                else:
                    legs = []
                    for edge in closed.edges:
                        expanded = expand_macro_edge(
                            edge,
                            lengths,
                            self.connectors[1],
                            min(lengths),
                            min(lengths),
                        )
                        if expanded is None:
                            break
                        legs.extend(
                            (leg.heading, leg.length)
                            for leg in expanded
                            if leg.length > 0
                        )
                    else:
                        self.accept(
                            self.assign_loop(anchor.start, anchor.heading_deg, legs)
                        )
                    continue
                self.accept(construction)

    def beam(self, anchors, width, deadline):
        ranked = []
        for i, anchor in enumerate(anchors):
            now = time.process_time()
            if now >= deadline:
                break
            search = BeamSearch(
                anchor,
                self.catalog,
                self.connectors,
                self.weights,
                self.region,
                width,
                original.MAX_FACES,
                now + (deadline - now) / (len(anchors) - i),
            )
            anchor_score = 0.0
            for closed in search.run():
                if time.process_time() >= deadline:
                    break
                construction = (
                    original.to_construction(anchor, closed)
                    if self.assignment is None
                    else self.assign_loop(
                        anchor.start,
                        anchor.heading_deg,
                        [(f.heading, f.option.length) for f in closed.faces],
                    )
                )
                score = self.accept(construction)
                if score is not None:
                    anchor_score = max(anchor_score, score)
            ranked.append((anchor_score, anchor))
        return [anchor for _, anchor in sorted(ranked, key=lambda row: -row[0])]

    def grow(self, deadline):
        if not self.seeds:
            return
        self.assignment.max_nodes = 3000
        grower = Grower(
            region=self.region,
            room_corners=np.array(self.room.get_boundary_points()),
            buildable=set(self.assignment.lengths),
            rights=self.connectors[1],
            diagonals=self.connectors[2],
            total_length=self.assignment.total_length,
            score=lambda area, perimeter: (
                BASELINE_SCORE
                + self.weights.A * area
                + self.weights.C * perimeter
                + (self.weights.G if len(self.assignment.targets) == 2 else 0)
            ),
            accept=lambda start, heading, faces: self.accept(
                self.assign_loop(start, heading, faces)
            ),
            can_build=lambda lengths: self.assignment.assign(lengths) is not None,
            max_faces=min(900, self.connectors[1] + self.connectors[2]),
        )
        seeds = []
        for seed in self.seeds:
            if all(
                math.dist(seed.start, other.start) > 10 or seed.faces != other.faces
                for other in seeds
            ):
                seeds.append(seed)
            if len(seeds) == 4:
                break
        grower.grow(seeds, deadline, first_charge=max(self.weights.A, 0.1))

    def checkpoint(self, name, started):
        self.stages.append(
            {
                "stage": name,
                "cpu": time.process_time() - started,
                "score": self.best_score,
            }
        )

    def build_enclosure(self):
        # GUI builds reuse the player instance; each build gets fresh state.
        self.best, self.best_score = None, 0.0
        self.seeds, self.stages = [], []
        self.assignment = None
        self.note = None
        started = time.process_time()
        end = started + self.budget
        if not self.inventory.has_gate():
            return None
        self.catalog = FaceCatalog(
            self.inventory, max_pieces_per_face=max_pieces_per_face(self.inventory)
        )
        if not self.catalog.lengths:
            return None
        self.connectors = tuple(
            self.inventory.connectors.get(kind, 0)
            for kind in (
                ConnectorType.STRAIGHT,
                ConnectorType.RIGHT,
                ConnectorType.DIAGONAL,
            )
        )
        self.region = self.room.polygon.buffer(TOL)
        shapely.prepare(self.region)
        if self.variant in {"assigned", "grown"}:
            self.assignment = Assigner(
                self.inventory, want_second_gate=self.weights.G > 0
            )
        anchors = spread_anchors(original.room_anchors(self.room.polygon))
        # The grown variant reserves half of the total budget for local repair.
        search_end = started + self.budget * (0.5 if self.variant == "grown" else 1.0)
        duration = max(0, search_end - time.process_time())
        stage_start = time.process_time()
        if should_use_macro(self.inventory):
            self.macro(anchors, stage_start + duration * 0.25)
        self.checkpoint("macro", started)
        ranked = self.beam(anchors, 40, stage_start + duration * 0.50)
        self.checkpoint("narrow", started)
        top = ranked[:8] or anchors[:8]
        self.beam(top, 500, stage_start + duration * 0.85)
        self.checkpoint("wide", started)
        restarts = (
            spread_anchors(anchors_from_construction(self.best), 8)
            if self.best
            else top
        )
        self.beam(restarts, 500, search_end)
        self.checkpoint("restart", started)
        if self.variant == "grown":
            self.grow(end)
            self.checkpoint("grow", started)
        self.note = f"{self.variant}: score {self.best_score:.1f}"
        return self.best
