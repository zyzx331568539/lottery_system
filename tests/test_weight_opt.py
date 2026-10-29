# -*- coding: utf-8 -*-
"""weight_opt 模块的轻量测试（用合成数据加速）。"""
import unittest

from tests.test_analytics import synth_records
from weight_opt import (DEFAULT_WEIGHTS, SPREAD_FIXED, best_static_weights,
                        optimize_weights_walkforward, weight_grid)


class TestWeightOpt(unittest.TestCase):

    def test_grid_contains_default_and_sums_to_one(self):
        grid = weight_grid()
        self.assertIn(DEFAULT_WEIGHTS, grid)
        for w in grid:
            self.assertAlmostEqual(sum(w), 1.0, places=6)
            self.assertEqual(w[3], SPREAD_FIXED)

    def test_walkforward_output_structure(self):
        recs = synth_records(40)
        r = optimize_weights_walkforward(recs, min_outer=20, inner_window=10)
        for key in ("optimized", "default", "baseline_mean_hits",
                    "weight_usage", "details"):
            self.assertIn(key, r)
        self.assertEqual(r["optimized"]["trials"], r["default"]["trials"])
        self.assertEqual(len(r["details"]), r["optimized"]["trials"])
        # 每期 detail 应有完整字段
        for d in r["details"]:
            self.assertIn("weights", d)
            self.assertIn("hits_optimized", d)
            self.assertIn("hits_default", d)
            self.assertEqual(len(d["weights"]), 4)

    def test_default_weights_in_grid(self):
        # 默认权重必须在网格中，保证"不优化"是合法候选
        self.assertIn(DEFAULT_WEIGHTS, weight_grid())

    def test_best_static_weights_structure(self):
        recs = synth_records(40)
        r = best_static_weights(recs, inner_window=10)
        self.assertIn("best_weights", r)
        self.assertEqual(len(r["best_weights"]), 4)
        self.assertAlmostEqual(sum(r["best_weights"]), 1.0, places=6)
        self.assertIn("best_mean_hits", r)
        self.assertIn("default_mean_hits", r)
        # best_mean_hits 应 >= default_mean_hits（事后最优至少不差）
        self.assertGreaterEqual(r["best_mean_hits"], r["default_mean_hits"] - 1e-9)

    def test_too_few_records_raises(self):
        recs = synth_records(5)
        with self.assertRaises(Exception):
            optimize_weights_walkforward(recs, min_outer=20, inner_window=10)


if __name__ == "__main__":
    unittest.main()
