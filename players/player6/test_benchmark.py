"""Small-room/scarce regressions and reproducible benchmark output checks."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from players.player6.benchmark import run_case, scenarios, write_report


class BenchmarkTests(unittest.TestCase):
    budgets: ClassVar[dict[str, float]] = {
        "search_seconds": 0.3,
        "grid_seconds": 0.05,
        "beam_seconds": 0.1,
        "local_seconds": 0.03,
    }

    def test_small_room_and_short_stock_end_to_end(self):
        cases = scenarios()
        for variant in ("baseline", "local45"):
            tiny = run_case("tiny_room", cases["tiny_room"], variant, **self.budgets)
            self.assertEqual(tiny["status"], "valid", tiny)
            self.assertAlmostEqual(tiny["score"], 990)
            self.assertTrue(tiny["inventory_unchanged"])
            short = run_case(
                "insufficient_stock",
                cases["insufficient_stock"],
                variant,
                **self.budgets,
            )
            self.assertEqual(short["status"], "none", short)
            self.assertEqual(short["score"], 0)
            self.assertTrue(short["inventory_unchanged"])

    def test_fixed_scarce_seeds_return_valid_or_none_within_budget(self):
        cases = scenarios()
        for name in ("scarce_seed1", "scarce_seed2"):
            row = run_case(name, cases[name], "local45", **self.budgets)
            self.assertIn(row["status"], ("valid", "none"), row)
            self.assertTrue(row["inventory_unchanged"])
            self.assertLess(row["cpu_seconds"], 0.8)

    def test_shared_lengths_use_only_actual_gate_stock(self):
        row = run_case(
            "shared_lengths", scenarios()["shared_lengths"], "local45", **self.budgets
        )
        self.assertEqual(row["status"], "valid", row)
        self.assertEqual(row["gate_count"], 1)
        self.assertFalse(row["second_gate_bonus"])

    def test_reports_record_both_variants_scores_validity_and_time(self):
        case = scenarios()["tiny_room"]
        rows = [
            run_case("tiny_room", case, variant, **self.budgets)
            for variant in ("baseline", "local45")
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            write_report(output, rows, self.budgets)
            report = json.loads(output.read_text())
            self.assertEqual(report["config"], self.budgets)
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(output.with_suffix(".csv").exists())
            self.assertIn("Score delta", output.with_suffix(".md").read_text())
        for row in rows:
            self.assertIn("cpu_seconds", row)
            self.assertIn("wall_seconds", row)
            self.assertIn("local_accepted", row)


if __name__ == "__main__":
    unittest.main()
