# -*- coding: utf-8 -*-
"""
tests.test_msize —— 多规模扩展（1/2/3码）单元测试 + 4码回归等价性

覆盖：
  * 理论概率（精确超几何值、与 predictor/picksize 口径一致、分布归一）
  * FeaturesM 三码共现维度（计数守恒、与基类特征一致）
  * 广义组合评分（m=1 恒等、m=4 委派等价、确定性与排序）
  * 概率分析（输入验证、规模推断、命中期、m=4 委派等价）
  * 科学筛选管道（结构、确定性、m=4 委派等价）
  * 推荐（结构、置信度、m=4 委派等价）
  * 回测（基准归一、均值口径、切片正确性、m=4 委派等价）

运行（在 lottery_system 目录）：
    python -m unittest tests.test_msize -v
"""
from __future__ import annotations

import os
import sys
import unittest
from math import comb, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from backtest import BASELINE_SD, run_backtest
from data_store import ValidationError
from msize import (SIZES, FeaturesM, _check_size, analyze_combo_m,
                   combo_score_m, ranked_combos_m, recommend_m,
                   run_analysis_m, run_backtest_m, theory_for_size,
                   theory_table)
from predictor import Features, analyze_combo, exact_probability, recommend
from tests.test_analytics import real_records, synth_records


# ============================================================ 理论概率
class TestTheoryM(unittest.TestCase):
    def test_known_exact_values(self):
        """全中概率精确值：25% / 6.012658% / 1.387537% / 0.306339%"""
        self.assertAlmostEqual(theory_for_size(1)["all_hit"], 0.25, places=12)
        self.assertAlmostEqual(theory_for_size(2)["all_hit_pct"],
                               6.012658, places=5)
        self.assertAlmostEqual(theory_for_size(3)["all_hit_pct"],
                               1.387537, places=5)
        self.assertAlmostEqual(theory_for_size(4)["all_hit_pct"],
                               0.306339, places=5)

    def test_consistency_with_predictor(self):
        """与 predictor 既有常量同口径：P4、P2 必须一致"""
        th4, th2 = theory_for_size(4), theory_for_size(2)
        ep = exact_probability()
        self.assertAlmostEqual(th4["all_hit"], ep["p4"], places=15)
        self.assertAlmostEqual(th2["all_hit"], ep["p2"], places=15)

    def test_distribution_normalized_and_moments(self):
        for m in SIZES:
            th = theory_for_size(m)
            dist = th["distribution_pct"]
            self.assertEqual(sorted(int(k) for k in dist), list(range(m + 1)))
            self.assertAlmostEqual(sum(dist.values()), 100.0, places=3)
            self.assertAlmostEqual(th["mean_hits"], m / 4, places=9)
            var = m * 0.25 * 0.75 * (80 - m) / 79
            self.assertAlmostEqual(th["sd_hits"], sqrt(var), places=3)
            # 至少中1 = 1 - P(0)
            self.assertAlmostEqual(th["at_least_1_pct"],
                                   100 - dist["0"], places=3)

    def test_sd_matches_backtest_baseline_at_m4(self):
        """m=4 的 SD 必须与 backtest 基准常量完全一致（口径统一）"""
        self.assertAlmostEqual(theory_for_size(4)["sd_hits"],
                               BASELINE_SD, places=3)

    def test_invalid_size_rejected(self):
        for bad in (0, 5, -1, 2.5, True, "3"):
            with self.assertRaises((ValueError, ValidationError)):
                theory_for_size(bad)
            with self.assertRaises(ValidationError):
                _check_size(bad)

    def test_theory_table_structure(self):
        tb = theory_table()
        self.assertEqual(tb["sizes"], [1, 2, 3, 4])
        self.assertEqual(len(tb["rows"]), 4)
        self.assertIn("conditional_factors", tb)
        # 相邻倍率：P_{m-1}/P_m = (80-m+1)/(20-m+1)
        self.assertAlmostEqual(tb["rows"][1]["all_hit_ratio_vs_m_minus_1"],
                               79 / 19, places=4)


# ============================================================ FeaturesM 三码维度
class TestFeaturesM(unittest.TestCase):
    def test_triple_count_conservation(self):
        """三码共现总数必须守恒：n 期 × C(20,3)"""
        recs = synth_records(12)
        fx = FeaturesM(recs)
        self.assertEqual(sum(fx.triple.values()), 12 * comb(20, 3))

    def test_inherited_features_identical(self):
        """继承维度（频次/两两共现）与基类完全一致"""
        recs = synth_records(10)
        f0, f1 = Features(recs), FeaturesM(recs)
        self.assertEqual(f0.freq, f1.freq)
        self.assertEqual(f0.pair, f1.pair)
        self.assertEqual(f0.four, f1.four)
        for n in (1, 40, 80):
            self.assertAlmostEqual(f0.z_num(n), f1.z_num(n), places=12)

    def test_z_triple_never_seen(self):
        """从未共现的三码组合：z = -期望/SD（负值）"""
        recs = synth_records(10, excluded=80)
        fx = FeaturesM(recs)
        z = fx.z_triple((1, 2, 80))
        self.assertLess(z, 0)
        self.assertAlmostEqual(z, -fx.m_3 / fx.sd_3, places=9)


# ============================================================ 广义组合评分
class TestComboScoreM(unittest.TestCase):
    def setUp(self):
        self.recs = synth_records(15)
        self.fx1 = Features(self.recs)      # m=1/2 用基类
        self.fx3 = FeaturesM(self.recs)     # m=3 需要三码维度

    def test_m1_score_is_z_num(self):
        for n in (5, 42, 80):
            s, parts = combo_score_m(self.fx1, (n,))
            self.assertAlmostEqual(s, self.fx1.z_num(n), places=12)
            self.assertEqual(set(parts), {"number_z"})

    def test_m2_weights(self):
        c = (10, 20)
        s, parts = combo_score_m(self.fx1, c)
        vn = (self.fx1.z_num(10) + self.fx1.z_num(20)) / 2
        vp = self.fx1.z_pair(10, 20)
        self.assertAlmostEqual(s, 0.6 * vn + 0.4 * vp, places=9)
        self.assertEqual(set(parts), {"number_z", "pair_z"})

    def test_m3_parts_and_determinism(self):
        c = (3, 17, 42)
        s1, p1 = combo_score_m(self.fx3, c)
        s2, p2 = combo_score_m(self.fx3, c)
        self.assertEqual((s1, p1), (s2, p2))       # 确定性
        self.assertEqual(set(p1), {"number_z", "pair_z", "triple_z", "spread"})
        # 同区三码 spread=0，三区各一 spread=1
        _, pa = combo_score_m(self.fx3, (1, 2, 3))
        _, pb = combo_score_m(self.fx3, (5, 15, 75))
        self.assertEqual(pa["spread"], 0.0)
        self.assertEqual(pb["spread"], 1.0)

    def test_m3_requires_features_m(self):
        with self.assertRaises(TypeError):
            combo_score_m(self.fx1, (1, 2, 3))     # 基类无三码维度 → 明确报错

    def test_m4_delegates_exactly(self):
        c = (8, 29, 42, 65)
        s_new, p_new = combo_score_m(self.fx1, c)
        s_old, p_old = self.fx1.combo_score(c)
        self.assertEqual((s_new, p_new), (s_old, p_old))

    def test_ranked_combos_sorted_and_complete(self):
        pool = [3, 8, 15, 29, 42, 65]
        rows = ranked_combos_m(self.fx1, pool, 2)
        self.assertEqual(len(rows), comb(6, 2))
        scores = [r["score"] for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ============================================================ 概率分析
class TestAnalyzeM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = real_records()

    def test_input_validation(self):
        for bad in ([], [1, 2, 3, 4, 5], [7, 7], [0], [81], ["x"], None):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                analyze_combo_m(bad, self.recs)

    def test_empty_pool_rejected(self):
        with self.assertRaises(ValidationError):
            analyze_combo_m([5], [])

    def test_size_inferred(self):
        for m, combo in ((1, [42]), (2, [8, 42]), (3, [8, 29, 42])):
            r = analyze_combo_m(combo, self.recs)
            self.assertEqual(r["size"], m)
            self.assertEqual(r["combo"], sorted(combo))
            self.assertEqual(r["theory_m"]["size"], m)
            self.assertEqual(len(r["numbers"]), m)
            self.assertEqual(len(r["pairs"]), comb(m, 2))

    def test_co_kind_and_hit_periods(self):
        recs = synth_records(10, forced=77)
        r1 = analyze_combo_m([77], recs)
        self.assertEqual(r1["empirical"]["co_kind"], "number")
        self.assertEqual(r1["empirical"]["co_occurrence"], 10)
        self.assertEqual(len(r1["empirical"]["hit_periods"]), 10)

        # 两码：命中期 = 两码同时在场的期（用真实数据自洽校验）
        c = self.recs[-1]["numbers"][:2]
        r2 = analyze_combo_m(c, self.recs)
        self.assertEqual(r2["empirical"]["co_kind"], "pair")
        self.assertIn(self.recs[-1]["period"], r2["empirical"]["hit_periods"])
        expect = sum(1 for x in self.recs if set(c) <= set(x["numbers"]))
        self.assertEqual(len(r2["empirical"]["hit_periods"]), expect)

        c3 = self.recs[-1]["numbers"][:3]
        r3 = analyze_combo_m(c3, self.recs)
        self.assertEqual(r3["empirical"]["co_kind"], "triple")
        self.assertIn(self.recs[-1]["period"], r3["empirical"]["hit_periods"])

    def test_m4_delegation_equivalent(self):
        """m=4 必须与原 analyze_combo 逐字段一致（另加 size/theory_m/co_kind）"""
        combo = [8, 29, 42, 65]
        rn = analyze_combo_m(combo, self.recs)
        ro = analyze_combo(combo, self.recs)
        self.assertEqual(rn["size"], 4)
        self.assertEqual(rn["combo"], ro["combo"])
        self.assertEqual(rn["empirical"]["score"], ro["empirical"]["score"])
        self.assertEqual(rn["empirical"]["co_occurrence"],
                         ro["empirical"]["co_occurrence"])
        self.assertEqual(rn["empirical"]["co_kind"], "four")
        self.assertEqual(rn["empirical"]["hit_periods"],
                         ro["empirical"]["hit_periods"])
        self.assertEqual(rn["numbers"], ro["numbers"])
        self.assertEqual(rn["pairs"], ro["pairs"])
        self.assertEqual(rn["theory"], ro["theory"])


# ============================================================ 科学筛选管道
class TestRunAnalysisM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = real_records()

    def test_structure_all_sizes(self):
        for m in SIZES:
            rep = run_analysis_m(self.recs, m, persist=False)
            self.assertEqual(rep["size"], m)
            self.assertEqual(len(rep["numbers"]), m)
            self.assertEqual(len(set(rep["numbers"])), m)
            self.assertTrue(all(1 <= n <= 80 for n in rep["numbers"]))
            self.assertEqual(len(rep["ranking_basis"]), m)
            self.assertEqual(len(rep["hot_cold"]), 80)
            self.assertEqual(len(rep["score_table"]), 80)
            self.assertEqual(rep["theory_m"]["size"], m)
            self.assertGreater(rep["perf_ms"], 0)
            # 最优组合必须来自 shortlist
            self.assertTrue(set(rep["numbers"]) <= set(rep["shortlist"]))

    def test_deterministic(self):
        for m in (1, 2, 3):
            r1 = run_analysis_m(self.recs, m, persist=False)
            r2 = run_analysis_m(self.recs, m, persist=False)
            self.assertEqual(r1["numbers"], r2["numbers"])
            self.assertEqual(r1["combo_score"], r2["combo_score"])

    def test_m4_identical_to_original(self):
        """4 码回归：多规模入口必须得到与原管道完全相同的号码与评分"""
        rn = run_analysis_m(self.recs, 4, persist=False)
        from analytics import run_analysis
        ro = run_analysis(self.recs, persist=False)
        self.assertEqual(rn["numbers"], ro["numbers"])
        self.assertEqual(rn["combo_score"], ro["combo_score"])
        self.assertEqual(rn["ranking_basis"], ro["ranking_basis"])

    def test_bad_inputs(self):
        with self.assertRaises(ValidationError):
            run_analysis_m(self.recs, 5, persist=False)
        with self.assertRaises(ValueError):
            run_analysis_m([], 2, persist=False)
        with self.assertRaises(ValueError):
            run_analysis_m(synth_records(3), 2, persist=False)


# ============================================================ 推荐
class TestRecommendM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = real_records()

    def test_structure(self):
        for m in (1, 2, 3):
            r = recommend_m(self.recs, m, top_k=3)
            self.assertEqual(r["size"], m)
            self.assertEqual(len(r["recommendations"]), 3)
            for rec in r["recommendations"]:
                self.assertEqual(len(rec["combo"]), m)
                self.assertIn(rec["confidence_label"], ("High", "Moderate", "Low"))
                self.assertTrue(0.0 <= rec["confidence"] <= 1.0)
            self.assertEqual(r["theory_m"]["size"], m)
            self.assertIn("method", r)

    def test_deterministic_fixed_seed(self):
        r1 = recommend_m(self.recs, 2, top_k=3)
        r2 = recommend_m(self.recs, 2, top_k=3)
        self.assertEqual(r1["recommendations"], r2["recommendations"])

    def test_m4_delegation_equivalent(self):
        rn = recommend_m(self.recs, 4, top_k=3)
        ro = recommend(self.recs, None, 3)
        self.assertEqual(rn["recommendations"], ro["recommendations"])
        self.assertEqual(rn["candidate_pool"], ro["candidate_pool"])

    def test_bad_size_and_window(self):
        with self.assertRaises(ValidationError):
            recommend_m(self.recs, 0)
        with self.assertRaises(ValidationError):
            recommend_m(synth_records(3), 2)


# ============================================================ 回测
class TestBacktestM(unittest.TestCase):
    def test_baseline_per_size(self):
        recs = synth_records(20)
        for m in (1, 2, 3):
            r = run_backtest_m(recs, m, train_window=15)
            base = r["baseline"]
            self.assertAlmostEqual(
                sum(base["distribution_pct"].values()), 100.0, places=2)
            self.assertAlmostEqual(base["mean_hits"], m / 4, places=9)
            self.assertEqual(len(base["distribution_pct"]), m + 1)

    def test_structure_and_slice(self):
        recs = synth_records(20)
        r = run_backtest_m(recs, 2, train_window=15)
        self.assertEqual(r["size"], 2)
        self.assertEqual(r["trials"], 5)
        self.assertEqual(sum(r["hit_distribution"].values()), 5)
        for d in r["details"]:
            self.assertEqual(len(d["picked"]), 2)
            self.assertTrue(0 <= d["hits"] <= 2)
            self.assertEqual(d["hits"], len(d["hit_numbers"]))
            # 时间切片：目标期必须晚于全部训练期（picked 不含未来信息的结构性保证）
            self.assertTrue(2026001 <= d["target_period"] <= 2026020)
        self.assertEqual(len(r["mean_hits_bootstrap95"]), 2)
        self.assertIn("verdict", r)

    def test_m4_delegation_equivalent(self):
        recs = synth_records(20)
        rn = run_backtest_m(recs, 4, train_window=15)
        ro = run_backtest(recs, train_window=15)
        self.assertEqual(rn["mean_hits"], ro["mean_hits"])
        self.assertEqual(rn["hit_distribution"], ro["hit_distribution"])
        self.assertEqual(rn["z_vs_baseline"], ro["z_vs_baseline"])
        self.assertEqual(rn["details"], ro["details"])

    def test_bad_inputs(self):
        recs = synth_records(10)
        with self.assertRaises(ValidationError):
            run_backtest_m(recs, 7, train_window=5)
        with self.assertRaises(ValueError):
            run_backtest_m(recs, 2, train_window=3)
        with self.assertRaises(ValueError):
            run_backtest_m(synth_records(4), 1, train_window=3)

    def test_real_pool_all_sizes(self):
        recs = real_records()
        for m in (1, 2, 3):
            r = run_backtest_m(recs, m, train_window=20)
            self.assertEqual(r["trials"], len(recs) - 20)
            self.assertGreaterEqual(r["trials"], 1)
            self.assertGreaterEqual(r["mean_hits"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
