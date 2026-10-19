"""Previous deliverable seed search followed by map-guided growth beams."""

import time

from src.enclosure import score_construction, validate_construction

from .faces import FaceInventory
from .growth import grow_enclosure
from .prev_deliverable import Player3 as PreviousPlayer3


class Player3(PreviousPlayer3):
    def build_enclosure(self):
        begun = time.process_time()
        deadline = begun + self.SEARCH_SECONDS
        seed_player = PreviousPlayer3(self.room, self.inventory, self.weights)
        seed_player.SEARCH_DEADLINE = deadline
        seed_player.BEAM_WIDTH = self.BEAM_WIDTH
        seed_player.MAX_FACES = self.MAX_FACES
        seed_player.USE_BEAM = self.USE_BEAM
        construction = seed_player.build_enclosure()
        self.progress = list(seed_player.progress)
        self.search_stats = {"seed": dict(seed_player.search_stats)}
        if construction is None:
            self.note = seed_player.note
            self.search_stats["cpu_seconds"] = time.process_time() - begun
            return None
        if time.process_time() < deadline:
            catalog = FaceInventory(self.inventory, deadline)
            construction, stats = grow_enclosure(
                self.room,
                self.inventory,
                self.weights,
                catalog,
                construction,
                deadline,
            )
            self.search_stats.update(stats)
        result = validate_construction(construction, self.room, self.inventory)
        score = score_construction(result, self.weights)
        self.search_stats.update(
            cpu_seconds=time.process_time() - begun,
            best_score=score,
            final_area=result.area,
        )
        self.progress.append(
            {
                "cpu": self.search_stats["cpu_seconds"],
                "score": score,
                "stage": "mapped_face_growth",
            }
        )
        self.note = (
            f"Mapped growth: {self.search_stats.get('growth_steps', 0)} extensions, "
            f"{self.search_stats.get('merges', 0)} merges, "
            f"{self.search_stats.get('active_fronts', 0)} fronts; area {result.area:.1f}"
        )
        return construction
