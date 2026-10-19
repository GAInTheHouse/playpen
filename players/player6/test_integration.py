"""Regression coverage for the inscribed-merged integration."""

import time
import unittest
from unittest.mock import patch

from shapely.geometry import Polygon, box

from players.player6 import inscribed_diagonal as idg
from players.player6.benchmark import scenarios
from players.player6.diagonal_search import DiagonalSearch
from players.player6.inscribed_polygon import SearchStopped
from players.player6.inscribed_polygon_stock import assign_inventory
from players.player6.local_search import LocalSearchStats
from players.player6.player import Player6
from players.player6.test_inscribed import make_player
from src.enclosure import score_construction, validate_construction
from src.pieces import ConnectorType as CT

STAIRS = [
    (10, 0),
    (40, 0),
    (40, 15),
    (35, 15),
    (35, 20),
    (30, 20),
    (30, 25),
    (0, 25),
    (0, 10),
    (5, 10),
    (5, 5),
    (10, 5),
]


class StaircaseTests(unittest.TestCase):
    def seeded_player(self):
        player = make_player(
            walls={5: 30, 7: 4},
            gates={5: 2},
            connectors={CT.RIGHT: 20, CT.STRAIGHT: 30, CT.DIAGONAL: 4},
            room=box(-20, -20, 60, 60),
        )
        player.possible_faces = {}
        player.enumerate_faces()
        lengths, _, _ = idg.to_faces(STAIRS)
        plan = assign_inventory(
            lengths,
            player.inventory.walls,
            player.inventory.gates,
            player.inventory.connectors,
        )
        self.assertIsNotNone(plan)
        self.assertIsNotNone(player._consider_inscribed(Polygon(STAIRS), plan))
        player.local_stats = LocalSearchStats()
        player.inscribed_diagonal_stats = LocalSearchStats()
        player._local_seconds = 0.2
        return player

    def test_paired_staircase_replacements_close_exactly(self):
        lengths, turns, _ = idg.to_faces(STAIRS)
        valid = 0
        for first, second in idg.pairs(lengths, turns):
            for length in idg.diag_lengths(first["a"]):
                candidate = idg.apply_pair(lengths, turns, first, second, length)
                if candidate is not None:
                    new_lengths, new_turns, _ = candidate
                    _, closure = idg.walk(new_lengths, new_turns)
                    self.assertEqual(closure, (0, 0, 0, 0))
                    self.assertTrue(
                        all(isinstance(n, int) and 5 <= n <= 30 for n in new_lengths)
                    )
                    valid += 1
        self.assertGreater(valid, 0)

    def test_geometry_enumeration_checks_deadline(self):
        with self.assertRaises(SearchStopped):
            idg.to_faces(STAIRS, stop=lambda: True)
        lengths, turns, _ = idg.to_faces(STAIRS)
        with self.assertRaises(SearchStopped):
            idg.pairs(lengths, turns, stop=lambda: True)

    def test_experiment_is_disabled_by_default(self):
        player = self.seeded_player()
        with patch.object(player, "diagonalize_inscribed") as search:
            player._improve_staircases(time.process_time() + 1)
        search.assert_not_called()

    def test_enabled_experiment_keeps_valid_score_improvements_and_stock(self):
        player = self.seeded_player()
        player.INSCRIBED_DIAGONALS = True
        original_inventory = player.inventory.copy()
        previous_score = player.best_score
        player._improve_staircases(time.process_time() + 0.3)
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(player.best_score, previous_score)
        self.assertAlmostEqual(
            player.best_score, score_construction(result, player.weights)
        )
        self.assertGreater(player.inscribed_diagonal_stats.accepted, 0)
        self.assertEqual(player.inventory, original_inventory)
        self.assertLess(player._local_cpu_used(), 0.3)

    def test_expired_experiment_retains_incumbent(self):
        player = self.seeded_player()
        original = player.best_construction
        player.diagonalize_inscribed(
            STAIRS, player.best_score, deadline=time.process_time() - 1
        )
        self.assertIs(player.best_construction, original)
        self.assertEqual(player.inscribed_diagonal_stats.attempted, 0)

    def test_staircase_and_existing_local_search_share_one_quota(self):
        player = self.seeded_player()
        player.inscribed_diagonal_stats.cpu_seconds = player._local_seconds
        with patch("players.player6.player.Local45Search") as search:
            player._improve_local(time.process_time() + 1)
        search.assert_not_called()


class PhaseTests(unittest.TestCase):
    def test_beam_history_uses_full_simulator_scores(self):
        scenario = scenarios()["diagonal_room"]
        search = DiagonalSearch(
            scenario.room, scenario.inventory, scenario.weights, time_budget=0.2
        )
        construction = search.build_enclosure()
        result = validate_construction(construction, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(search.history)
        self.assertAlmostEqual(
            search.history[-1][1], score_construction(result, scenario.weights)
        )
        self.assertTrue(
            all(
                after[1] > before[1]
                for before, after in zip(search.history, search.history[1:])
            )
        )

    def test_phase_notes_preserve_budget_and_refresh_between_runs(self):
        scenario = scenarios()["tiny_room"]
        player = Player6(
            scenario.room, scenario.inventory, scenario.weights, cpu_limit=0.2
        )
        for _ in range(2):
            construction = player.build_enclosure()
            self.assertTrue(
                validate_construction(construction, player.room, player.inventory).valid
            )
            self.assertIn("CPU budget 0.2s", player.note)
            self.assertIn("grid", player.note)
            self.assertIn("local 45", player.phase_seconds)
            self.assertEqual(player.beam_history, [])
            self.assertEqual(player.inscribed_diagonal_stats.cpu_seconds, 0)
            self.assertTrue(
                all(seconds >= 0 for seconds in player.phase_seconds.values())
            )

    def test_player_exports_full_beam_history_in_run_note(self):
        player = make_player(
            walls={5: 3},
            gates={5: 2},
            room=box(0, 0, 5, 5),
            connectors={CT.RIGHT: 4, CT.DIAGONAL: 4},
        )
        player.cpu_limit = 0.2
        with (
            patch.object(player, "full_checks"),
            patch.object(player, "generate_quads"),
            patch.object(player, "generate_quad_shapes"),
            patch.object(player, "_improve_local"),
            patch("players.player6.player.DiagonalSearch") as beam,
        ):
            beam.MAX_FACES = 12
            beam.return_value.history = [(0.01, 1090.0), (0.02, 1100.0)]
            beam.return_value.build_enclosure.return_value = None
            player.build_enclosure()
        self.assertEqual([score for _, score in player.beam_history], [1090, 1100])
        self.assertIn("beam improvements", player.note)
        self.assertIn("1100.00", player.note)


if __name__ == "__main__":
    unittest.main()
