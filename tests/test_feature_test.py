# -*- coding: utf-8 -*-
"""tests.test_feature_test —— 特征预测力检验(IC) 单元测试"""
from __future__ import annotations

import math
import unittest

import numpy as np

import feature_test as ft
from tests.test_analytics import synth_records


class TestRankCorrelation(unittest.TestCase):
    def test_rankdata_ties(self):
        # 并列值取平均秩
        r = ft._rankdata(np.array([1.0, 2.0, 2.0, 3.0]))
        self.assertEqual(list(r), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        r = ft._spearman(x, y)
        # y 有并列，秩相关不为 1.0，但应强正相关
        self.assertGreater(r, 0.8)

    def test_spearman_anti(self):
        x = np.array([4.0, 3.0, 2.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        r = ft._spearman(x, y)
        self.assertLess(r, -0.8)


class TestFeatureIC(unittest.TestCase):
    def test_runs_on_synth(self):
        rep = ft.feature_predictive_test(synth_records(30), min_train=8)
        self.assertIn("features", rep)
        self.assertIn("freq_z", rep["features"])
        self.assertIn("recent_z", rep["features"])
        self.assertIn("gap_score", rep["features"])
        self.assertIn("segment_z", rep["features"])
        self.assertIn("z_num_composite", rep["features"])
        for f in rep["features"].values():
            for k in ("mean_ic", "t_stat", "p_value", "verdict"):
                self.assertIn(k, f)
            self.assertTrue(-1.0 <= f["mean_ic"] <= 1.0)
            self.assertTrue(0.0 <= f["p_value"] <= 1.0)

    def test_needs_min_train(self):
        # 数据不足应抛错
        small = synth_records(6)
        with self.assertRaises(Exception):
            ft.feature_predictive_test(small, min_train=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
