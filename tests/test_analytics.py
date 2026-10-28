# -*- coding: utf-8 -*-
"""
tests.test_analytics —— 重构后核心系统的单元测试

覆盖：
  * 统计工具（norm_ppf / Wilson CI / softmax）
  * 冷热识别（近期热号阈值、遗漏冷号阈值、gap 计算）
  * 号段八区分析
  * 统一筛选管道（确定性纪律、结构完整性、过程归档、性能字段）
  * 滑动回测（结构、基准概率、与线上同路径）

运行（在 lottery_system 目录）：
    python -m unittest tests.test_analytics -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analytics import (AnalysisParams, build_score_table, compute_gaps,
                       hot_cold_table, norm_ppf, run_analysis,
                       segment_analysis, softmax, wilson_ci)
from backtest import (BASELINE_MEAN, HYPER_BASELINE, run_backtest)
from predictor import Features


# ------------------------------------------------------------ 测试数据（确定性合成）
def synth_record(i: int, forced: int | None = None,
                 excluded: int | None = None) -> dict:
    """
    第 i 期20码：{ (7i+3k) mod 80 + 1 : k=0..19 }（步长3与80互素 → 20个互不重复）
    forced：额外保证某号码在场（替换一个占位号）
    excluded：保证某号码不在场（若被选中则与forced交换处理）
    """
    nums = {(7 * i + 3 * k) % 80 + 1 for k in range(20)}
    if excluded is not None and excluded in nums:
        nums.discard(excluded)
    if forced is not None and forced not in nums:
        nums.add(forced)
    # 真实开奖恒为20码：强制加入导致超出时，移除一个普通占位号
    while len(nums) > 20:
        drop = next(x for x in nums if x != forced)
        nums.discard(drop)
    # 始终补足到20个互不重复号码
    while len(nums) < 20:
        for fill in range(1, 81):
            if fill not in nums and fill != excluded:
                nums.add(fill)
                break
    period = 2026001 + i
    return {"period": period,
            "date": "2026-01-%02d" % ((i % 28) + 1),
            "numbers": sorted(nums),
            "source": "test"}


def synth_records(n: int, **kw) -> list:
    return [synth_record(i, **kw) for i in range(n)]


REAL_POOL = os.path.join(ROOT, "data", "pool.json")


def real_records() -> list:
    with open(REAL_POOL, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


# ============================================================ 统计工具
class TestStatTools(unittest.TestCase):
    def test_norm_ppf_known_quantiles(self):
        self.assertAlmostEqual(norm_ppf(0.5), 0.0, places=6)
        self.assertTrue(1.959 < norm_ppf(0.975) < 1.961)
        self.assertTrue(-1.961 < norm_ppf(0.025) < -1.959)
        self.assertTrue(1.64 < norm_ppf(0.95) < 1.65)
        self.assertTrue(2.32 < norm_ppf(0.99) < 2.33)

    def test_norm_ppf_domain(self):
        with self.assertRaises(ValueError):
            norm_ppf(0)
        with self.assertRaises(ValueError):
            norm_ppf(1)

    def test_wilson_ci_known(self):
        lo, hi = wilson_ci(50, 100)
        self.assertTrue(0.40 < lo < 0.41)
        self.assertTrue(0.59 < hi < 0.60)

    def test_wilson_ci_extremes(self):
        lo, hi = wilson_ci(0, 30)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.15)          # 小样本零成功，上界不为0但很小
        lo2, hi2 = wilson_ci(30, 30)
        self.assertGreater(lo2, 0.85)
        self.assertLessEqual(hi2, 1.0)

    def test_wilson_ci_empty(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_softmax_properties(self):
        probs = softmax([1.0, 2.0, 3.0, -1.0])
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertLess(probs[0], probs[1])      # 单调
        self.assertLess(probs[1], probs[2])

    def test_softmax_temperature(self):
        p_hot = softmax([0.0, 2.0], tau=0.5)     # 更极端
        p_cold = softmax([0.0, 2.0], tau=5.0)    # 更平均
        self.assertGreater(p_hot[1], p_cold[1])


# ============================================================ 遗漏值
class TestGaps(unittest.TestCase):
    def test_gap_basic(self):
        # 3期：号码5只在第1期出现 → gap=2；号码9从不出现 → gap=3；
        # 最新一期在场的号 → gap=0
        recs = synth_records(3)
        # 手工构造确定性场景：三期号码区间 1-20 / 21-40 / 41-60
        recs[0]["numbers"] = list(range(1, 21))    # 号码5在场
        recs[1]["numbers"] = list(range(21, 41))
        recs[2]["numbers"] = list(range(41, 61))   # 号码50在场
        gaps = compute_gaps(recs)
        self.assertEqual(gaps[5], 2)               # 第1期后连续2期未出
        self.assertEqual(gaps[77], 3)              # 全程未出现
        self.assertEqual(gaps[50], 0)              # 最新一期在场


# ============================================================ 冷热识别
class TestHotCold(unittest.TestCase):
    def setUp(self):
        self.params = AnalysisParams()

    def test_hot_number_detected(self):
        # 10期，号码77强制每期在场：recent z=(10-2.5)/sqrt(1.875)=5.48 → hot
        recs = synth_records(10, forced=77)
        fx = Features(recs)
        table = hot_cold_table(recs, self.params, fx)
        row77 = next(r for r in table if r["number"] == 77)
        self.assertEqual(row77["recent_label"], "hot")
        self.assertEqual(row77["recent_count"], 10)

    def test_cold_gap_detected(self):
        # 10期，号码80从不出现 → gap=10 → cold_gap
        recs = synth_records(10, excluded=80)
        fx = Features(recs)
        table = hot_cold_table(recs, self.params, fx)
        row80 = next(r for r in table if r["number"] == 80)
        self.assertEqual(row80["gap_label"], "cold_gap")
        self.assertEqual(row80["gap"], 10)
        self.assertLessEqual(row80["gap_tail_pct"], 6)

    def test_table_row_completeness(self):
        recs = synth_records(12)
        table = hot_cold_table(recs, self.params, Features(recs))
        self.assertEqual(len(table), 80)
        labels = {"hot", "warm", "cold_recent"}
        self.assertTrue({r["recent_label"] for r in table} <= labels)
        for r in table:
            self.assertIn(r["gap_label"], {"cold_gap", "active", "gap_normal"})


# ============================================================ 号段分析
class TestSegments(unittest.TestCase):
    def test_segment_structure(self):
        recs = synth_records(30)
        seg = segment_analysis(recs, Features(recs))
        self.assertEqual(len(seg["zones"]), 8)
        self.assertEqual(seg["df"], 7)
        self.assertEqual(seg["zones"][0]["range"], [1, 10])
        self.assertEqual(seg["zones"][7]["range"], [71, 80])
        # 8区总出现数 = 每期20 × 30 = 600
        total = sum(z["observed"] for z in seg["zones"])
        self.assertEqual(total, 600)

    def test_segment_chi2_nonneg(self):
        recs = synth_records(20)
        seg = segment_analysis(recs, Features(recs))
        self.assertGreaterEqual(seg["chi_square"], 0)


# ============================================================ 统一管道
class TestRunAnalysis(unittest.TestCase):
    def test_deterministic_same_output(self):
        """纪律检验：相同输入必须得到完全相同的4码与评分"""
        recs = real_records()
        r1 = run_analysis(recs, persist=False)
        r2 = run_analysis(recs, persist=False)
        self.assertEqual(r1["numbers"], r2["numbers"])
        self.assertEqual(r1["combo_score"], r2["combo_score"])
        self.assertEqual(
            [(x["number"], x["total_score"]) for x in r1["ranking_basis"]],
            [(x["number"], x["total_score"]) for x in r2["ranking_basis"]])

    def test_report_structure(self):
        recs = real_records()
        rep = run_analysis(recs, persist=False)
        self.assertEqual(len(rep["numbers"]), 4)
        self.assertEqual(len(set(rep["numbers"])), 4)
        for n in rep["numbers"]:
            self.assertTrue(1 <= n <= 80)
        self.assertEqual(len(rep["ranking_basis"]), 4)
        self.assertEqual(len(rep["hot_cold"]), 80)
        self.assertEqual(len(rep["score_table"]), 80)
        self.assertEqual(len(rep["segments"]["zones"]), 8)
        self.assertGreater(rep["perf_ms"], 0)

    def test_ranking_basis_has_probability_and_ci(self):
        rep = run_analysis(real_records(), persist=False)
        for row in rep["ranking_basis"]:
            self.assertIn("total_score", row)
            self.assertIn("model_index_pct", row)
            self.assertIn("wilson95_ci_pct", row)
            self.assertEqual(len(row["wilson95_ci_pct"]), 2)
            self.assertIn("factors", row)
            for f in ("full_freq_z", "recent_z", "gap_score", "segment_score"):
                self.assertIn(f, row["factors"])

    def test_model_index_sums_about_100(self):
        rep = run_analysis(real_records(), persist=False)
        total = sum(t["model_index_pct"] for t in rep["score_table"])
        self.assertAlmostEqual(total, 100.0, places=2)

    def test_empty_and_short_inputs(self):
        with self.assertRaises(ValueError):
            run_analysis([], persist=False)
        with self.assertRaises(ValueError):
            run_analysis(synth_records(3), persist=False)

    def test_persist_writes_archive(self):
        from audit_log import ANALYSIS_DIR
        before = set(os.listdir(ANALYSIS_DIR)) if os.path.isdir(ANALYSIS_DIR) else set()
        rep = run_analysis(real_records(), persist=True)
        after = set(os.listdir(ANALYSIS_DIR))
        new_files = after - before
        self.assertEqual(len(new_files), 1)
        path = os.path.join(ANALYSIS_DIR, new_files.pop())
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        self.assertIsNotNone(doc["data_snapshot"])
        self.assertIsNotNone(doc["intermediate"])
        self.assertEqual(doc["final"]["numbers"], rep["numbers"])
        self.assertEqual(len(doc["integrity_sha256"]), 64)

    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisParams(w_full=0.9, w_recent=0.9).validate()
        with self.assertRaises(ValueError):
            AnalysisParams(shortlist=3).validate()


# ============================================================ 回测
class TestBacktest(unittest.TestCase):
    def test_baseline_sums_to_one(self):
        self.assertAlmostEqual(sum(HYPER_BASELINE.values()), 1.0, places=9)
        self.assertAlmostEqual(HYPER_BASELINE[4],
                               0.003063, places=5)
        self.assertAlmostEqual(BASELINE_MEAN, 1.0, places=9)

    def test_backtest_structure(self):
        recs = synth_records(20)
        res = run_backtest(recs, train_window=15)
        self.assertEqual(res["trials"], 5)
        self.assertEqual(sum(res["hit_distribution"].values()), 5)
        self.assertEqual(len(res["details"]), 5)
        for d in res["details"]:
            self.assertEqual(len(d["picked"]), 4)
            self.assertTrue(0 <= d["hits"] <= 4)
            self.assertEqual(d["hits"], len(d["hit_numbers"]))

    def test_backtest_bad_window(self):
        recs = synth_records(10)
        with self.assertRaises(ValueError):
            run_backtest(recs, train_window=3)
        with self.assertRaises(ValueError):
            run_backtest(recs, train_window=10)
        with self.assertRaises(ValueError):
            run_backtest(synth_records(4), train_window=3)

    def test_backtest_real_pool_runs(self):
        recs = real_records()
        res = run_backtest(recs, train_window=20)
        self.assertEqual(res["trials"], len(recs) - 20)
        self.assertGreaterEqual(res["trials"], 1)
        self.assertEqual(len(res["mean_hits_bootstrap95"]), 2)
        self.assertIn("verdict", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
