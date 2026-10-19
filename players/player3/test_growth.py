"""Regression checks: python -m unittest players.player3.test_growth."""

import time
import unittest
from pathlib import Path

from shapely.geometry import Polygon

from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.limits import CpuLimit
from src.pieces import ConnectorType
from src.room import Room
from src.scenario import read_scenario
from src.weights import Weights

from .faces import FaceInventory
from .growth import grow_enclosure
from .player import Player3


class GrowthScoreTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2]
        self.scenario = read_scenario(path / "scenarios/players/player3/trapezoid.json")

    def grow(self, room, inventory, weights, vertices, *, expired=False):
        original_inventory = inventory.to_dict()
        deadline = time.process_time() + 2
        catalog = FaceInventory(inventory, deadline)
        seed = catalog.build(vertices, prefer_bonus=weights.G > 0)
        self.assertIsNotNone(seed)
        initial = validate_construction(seed, room, inventory)
        self.assertTrue(initial.valid, initial.reason)
        with CpuLimit(3):
            construction, stats = grow_enclosure(
                room,
                inventory,
                weights,
                catalog,
                seed,
                time.process_time() - 1 if expired else deadline,
            )
        result = validate_construction(construction, room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(stats["final_area"], result.area)
        self.assertEqual(inventory.to_dict(), original_inventory)
        return score_construction(initial, weights), score_construction(result, weights)

    def test_player_keeps_seed_when_growth_lowers_score(self):
        for penalty, expected in [(-10, 750), (-30, 30)]:
            with self.subTest(C=penalty):
                scenario = self.scenario.copy()
                weights = Weights(A=1, C=penalty, G=30)
                player = Player3(scenario.room, scenario.inventory, weights)
                with CpuLimit(60):
                    construction = player.build_enclosure()
                self.assertIsNotNone(construction)
                result = validate_construction(
                    construction, scenario.room, scenario.inventory
                )
                self.assertTrue(result.valid, result.reason)
                self.assertAlmostEqual(score_construction(result, weights), expected)

    def test_keeps_better_intermediate_before_costly_extension(self):
        room = Room(Polygon([(0, 0), (30, 0), (30, 5), (20, 5), (20, 20), (0, 20)]))
        inventory = Inventory(
            walls={5: 8, 10: 8, 20: 4, 30: 2},
            gates={5: 2},
            connectors={ConnectorType.RIGHT: 8, ConnectorType.STRAIGHT: 4},
        )
        initial, final = self.grow(
            room,
            inventory,
            Weights(A=1, C=-3, G=0),
            [(5, 5), (15, 5), (15, 15), (5, 15)],
        )
        self.assertAlmostEqual(initial, 980)
        self.assertAlmostEqual(final, 1160)

    def test_keeps_gate_bonus_and_best_validated_union(self):
        room = Room(Polygon([(0, 0), (30, 0), (30, 20), (0, 20)]))
        inventory = Inventory(
            walls={5: 2, 10: 2, 20: 3, 30: 2},
            gates={5: 2},
            connectors={ConnectorType.RIGHT: 4, ConnectorType.STRAIGHT: 2},
        )
        initial, final = self.grow(
            room,
            inventory,
            Weights(A=1, C=-1, G=1000),
            [(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        self.assertAlmostEqual(initial, 2060)
        self.assertAlmostEqual(final, 2220)

    def test_accepts_growth_that_improves_score(self):
        scenario = self.scenario
        initial, final = self.grow(
            scenario.room,
            scenario.inventory,
            Weights(A=1, C=-1, G=30),
            [(10, 7), (20, 7), (20, 15), (10, 15)],
        )
        self.assertGreater(final, initial)

    def test_expired_growth_deadline_preserves_seed(self):
        scenario = self.scenario
        initial, final = self.grow(
            scenario.room,
            scenario.inventory,
            scenario.weights,
            [(10, 7), (20, 7), (20, 15), (10, 15)],
            expired=True,
        )
        self.assertAlmostEqual(initial, final)


if __name__ == "__main__":
    unittest.main()
