"""Anytime, room-guided face beam search with a validated rectangle incumbent."""

import time

from shapely.geometry import Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import MIN_WALL_LENGTH
from src.enclosure import score_construction, validate_construction

from .beam import face_beam
from .faces import FaceInventory
from .placement import rectangle_seeds


class Player3(Player):
    # Compatible with the submission checker's 60 CPU-second default, even
    # though the simulator's separately configurable default is 300 seconds.
    SEARCH_SECONDS = 55.0
    RECTANGLE_SECONDS = 5.0
    BEAM_WIDTH = 24
    MAX_FACES = 14
    USE_BEAM = True

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
        stage = "rectangle"
        stage_deadline = min(deadline, time.process_time() + self.RECTANGLE_SECONDS)

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

        self.search_stats.update(
            rectangle_seeds(
                self.room,
                catalog,
                self.weights,
                stage_deadline,
                consider,
            )
        )
        if self.USE_BEAM:
            stage = "beam"
            reserve = min(5.0, self.SEARCH_SECONDS * 0.15) if best is None else 0.25
            beam_deadline = deadline - reserve
            stage_deadline = beam_deadline
            self.search_stats.update(
                face_beam(
                    self.room,
                    self.inventory,
                    catalog,
                    self.weights,
                    beam_deadline,
                    consider,
                    beam_width=self.BEAM_WIDTH,
                    max_faces=self.MAX_FACES,
                )
            )
        if best is None and deadline - time.process_time() > 0.1:
            from .legacy import Player3 as LegacyPlayer

            fallback = LegacyPlayer(self.room, self.inventory, self.weights)
            fallback.SEARCH_SECONDS = max(0.01, deadline - time.process_time() - 0.1)
            construction = fallback.build_enclosure()
            if construction is not None:
                result = validate_construction(construction, self.room, self.inventory)
                score = score_construction(result, self.weights) if result.valid else 0
                if score > best_score:
                    best, best_score = construction, score
                    self.progress.append(
                        {
                            "cpu": time.process_time() - begun,
                            "score": score,
                            "stage": "fallback",
                        }
                    )
        self.search_stats["cpu_seconds"] = time.process_time() - begun
        self.search_stats["best_score"] = best_score
        self.note = (
            f"Face beam: {self.search_stats.get('states', 0)} states, "
            f"{self.search_stats.get('starts', 0)} starts, "
            f"{self.search_stats['valid_candidates']} valid candidates; best {best_score:.1f}"
        )
        return best
