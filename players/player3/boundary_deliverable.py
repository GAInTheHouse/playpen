"""Stock-aware boundary tracing from the room’s most open faces."""

import math
import time

from shapely.geometry import Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import MIN_WALL_LENGTH
from src.enclosure import score_construction, validate_construction

from .boundary import boundary_search
from .faces import FaceInventory


class Player3(Player):
    # Compatible with the submission checker's 60 CPU-second default, even
    # though the simulator's separately configurable default is 300 seconds.
    SEARCH_SECONDS = 55.0
    BRANCHES = 24

    def _boundary_parameters(self):
        """Allocate branching using scale-independent room compactness.

        Compactness approaches one for open, compact rooms and zero for
        thin or winding rooms. Keep alternative inward detours even at
        the narrow end of this boundary-guided beam. Breadth stays high in
        early layers and tapers with depth, especially in winding rooms.
        """
        polygon = self.room.polygon
        compactness = max(0.0, min(1.0, 4 * math.pi * polygon.area / polygon.length**2))
        branches = max(1, round(self.BRANCHES * (1.0 + 0.5 * compactness)))
        return compactness, branches

    def build_enclosure(self):
        begun = time.process_time()
        deadline = begun + self.SEARCH_SECONDS
        self.search_stats = {"candidates": 0, "valid_candidates": 0, "improvements": 0}
        self.progress = []
        if not self.inventory.has_gate() or self.inventory.count_connectors() < 3:
            self.note = "No usable gate or insufficient connectors"
            return None
        if any(
            length < MIN_WALL_LENGTH
            for pool in (self.inventory.walls, self.inventory.gates)
            for length, count in pool.items()
            if count > 0
        ):
            self.note = "Inventory contains a piece shorter than the minimum"
            return None
        catalog = FaceInventory(self.inventory, deadline)
        prepared_room = prep(self.room.polygon.buffer(1e-7))
        best, best_score = None, 0.0
        seen = {}
        stage = "boundary"
        stage_deadline = deadline

        def consider(vertices):
            nonlocal best, best_score
            if time.process_time() >= stage_deadline:
                return None
            key = tuple((round(x, 6), round(y, 6)) for x, y in vertices)
            if key in seen:
                return seen[key]
            self.search_stats["candidates"] += 1
            polygon = Polygon(vertices)
            if not polygon.is_valid or polygon.area <= 1e-7:
                seen[key] = None
                return None
            if not prepared_room.covers(polygon):
                seen[key] = None
                return None
            construction = catalog.build(
                vertices, prefer_bonus=self.weights.G > 0, deadline=stage_deadline
            )
            if construction is None:
                if time.process_time() < stage_deadline:
                    seen[key] = None
                return None
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                seen[key] = None
                return None
            score = score_construction(result, self.weights)
            seen[key] = score
            self.search_stats["valid_candidates"] += 1
            if score > best_score + 1e-8:
                best, best_score = construction, score
                self.search_stats["improvements"] += 1
                self.progress.append(
                    {"cpu": time.process_time() - begun, "score": score, "stage": stage}
                )
            return score

        compactness, branches = self._boundary_parameters()
        max_faces = max(14, round(14 * (2.0 - compactness)))
        self.search_stats.update(
            room_compactness=compactness, branches=branches, max_faces=max_faces
        )
        self.search_stats.update(
            boundary_search(
                self.room,
                self.inventory,
                catalog,
                self.weights,
                deadline,
                consider,
                branches=branches,
                compactness=compactness,
                max_faces=max_faces,
            )
        )
        self.search_stats["cpu_seconds"] = time.process_time() - begun
        self.search_stats["best_score"] = best_score
        self.note = (
            f"Boundary beam: {self.search_stats.get('states', 0)} states, "
            f"{self.search_stats.get('starts', 0)} starts, "
            f"{self.search_stats['valid_candidates']} valid candidates; best {best_score:.1f}"
        )
        return best
