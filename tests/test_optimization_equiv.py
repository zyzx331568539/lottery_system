# -*- coding: utf-8 -*-
"""
tests.test_optimization_equiv —— 算法优化“输出 100% 一致”回归对拍

方法：优化前实现已原样冻结在 tests/legacy/（仅改包内 import 路径），
本测试对同一批真实数据 + 多组合成数据，逐字段比较新旧实现的全部公开输出，
包括浮点字段（assertEqual 位级相等，不做近似容差）。

覆盖：
  * Features / FeaturesM：计数器、z 值、候选集、组合排序、衰减权重
  * analyze_combo / analyze_combo_m（m=1..4）：完整报告字典
  * recommend / recommend_m（含 last_n 切片）：完整报告字典
  * run_analysis / run_analysis_m（m=1..3）：除 perf_ms 外完整报告
  * run_backtest / run_backtest_m（m=1..3）：除 perf_ms 外完整报告
  * bootstrap_stability / bootstrap_stability_m：稳定度映射

运行（在 lottery_system 目录）：
    python -m unittest tests.test_optimization_equiv -v
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests.legacy import analytics as A0, backtest as B0, msize as M0, predictor as P0
import analytics as A1
import backtest as B1
import msize as M1
import predictor as P1
from tests.test_analytics import real_records, synth_records

# 性能计数字段允许不同（计时本身是测量值，非业务输出）
# method：recommend 的方法说明文案有意改为参数无关，不参与对拍
# weights / candidate_pool_size / combo_count：recommend.basis 新增字段，不参与对拍
# disclaimers：run_analysis 免责声明文案有意更新（引用 IC 结论），不参与对拍
_IGNORE_KEYS = {"perf_ms", "method", "weights",
                "candidate_pool_size", "combo_count", "disclaimers"}


def _strip(d):
    """递归移除忽略字段，保证新旧实现可逐字段对拍。"""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in _IGNORE_KEYS}
    if isinstance(d, list):
        return [_strip(x) for x in d]
    return d


DATASETS = {
    "real_pool": real_records(),
    "synth_20": synth_records(20),
    "synth_12": synth_records(12),
    "synth_31": synth_records(31),
}


class TestFeaturesEquivalence(unittest.TestCase):
    def test_features_bit_identical(self):
        for name, recs in DATASETS.items():
            f0, f1 = P0.Features(recs), P1.Features(recs)
            self.assertEqual(f0.freq, f1.freq, name)
            self.assertEqual(f0.pair, f1.pair, name)
            self.assertEqual(f0.four, f1.four, name)
            self.assertEqual(f0.wfreq, f1.wfreq, name)
            self.assertEqual(f0.m_f, f1.m_f, name)
            self.assertEqual(f0.sd_f, f1.sd_f, name)
            self.assertEqual(f0.m_w, f1.m_w, name)
            self.assertEqual(f0.sd_w, f1.sd_w, name)
            self.assertEqual(f0.m_p, f1.m_p, name)
            self.assertEqual(f0.sd_p, f1.sd_p, name)
            self.assertEqual(f0.m_4, f1.m_4, name)
            self.assertEqual(f0.sd_4, f1.sd_4, name)
            for n in (1, 27, 40, 53, 80):
                self.assertEqual(f0.z_num(n), f1.z_num(n), (name, n))
            for a, b in ((1, 80), (3, 17), (42, 65), (70, 71)):
                self.assertEqual(f0.z_pair(a, b), f1.z_pair(a, b), (name, a, b))
            for c in ((3, 17, 42, 65), (1, 2, 3, 4), (70, 71, 79, 80)):
                self.assertEqual(f0.z_four(c), f1.z_four(c), (name, c))
            self.assertEqual(f0.candidates(), f1.candidates(), name)
            self.assertEqual(f0.ranked_combos(), f1.ranked_combos(), name)
            # 百分位有序表
            self.assertEqual(
                sorted(f0.combo_score(c)[0] for c in f0.four),
                f1.observed_scores_sorted(), name)

    def test_features_m_bit_identical(self):
        for name, recs in DATASETS.items():
            f0, f1 = M0.FeaturesM(recs), M1.FeaturesM(recs)
            self.assertEqual(f0.triple, f1.triple, name)
            self.assertEqual((f0.m_3, f0.sd_3), (f1.m_3, f1.sd_3), name)
            for c in ((1, 2, 80), (3, 17, 42), (50, 51, 52)):
                self.assertEqual(f0.z_triple(c), f1.z_triple(c), (name, c))
            self.assertEqual(f0.freq, f1.freq, name)
            self.assertEqual(f0.pair, f1.pair, name)


class TestBootstrapEquivalence(unittest.TestCase):
    def test_bootstrap_stability_m4(self):
        recs = DATASETS["real_pool"]
        targets = [r["combo"] for r in P0.Features(recs).ranked_combos()[:5]]
        self.assertEqual(P0.bootstrap_stability(recs, targets),
                         P1.bootstrap_stability(recs, targets))

    def test_bootstrap_stability_m_all(self):
        recs = DATASETS["real_pool"]
        for m in (1, 2, 3):
            fx = M0._features_cls(m)(recs)
            targets = [r["combo"] for r in M0.ranked_combos_m(fx, fx.candidates(), m)[:5]]
            self.assertEqual(M0.bootstrap_stability_m(recs, targets, m),
                             M1.bootstrap_stability_m(recs, targets, m), m)


class TestAnalyzeEquivalence(unittest.TestCase):
    def test_analyze_combo_m4(self):
        recs = DATASETS["real_pool"]
        for combo in ([3, 17, 42, 65], [1, 2, 3, 4], [70, 71, 79, 80],
                      [8, 29, 42, 65]):
            self.assertEqual(P0.analyze_combo(combo, recs),
                             P1.analyze_combo(combo, recs), combo)

    def test_analyze_combo_m123(self):
        recs = DATASETS["real_pool"]
        for combo in ([42], [8, 42], [8, 29, 42], [1, 40, 80], [70, 71, 72]):
            self.assertEqual(M0.analyze_combo_m(combo, recs),
                             M1.analyze_combo_m(combo, recs), combo)

    def test_analyze_combo_repeated_uses_cache(self):
        """同一 fx 两次百分位查询与旧实现一致（缓存正确性）"""
        recs = DATASETS["real_pool"]
        r1 = P1.analyze_combo([3, 17, 42, 65], recs)
        r2 = P1.analyze_combo([1, 2, 3, 4], recs)
        self.assertEqual(P0.analyze_combo([3, 17, 42, 65], recs), r1)
        self.assertEqual(P0.analyze_combo([1, 2, 3, 4], recs), r2)
        r3 = M1.analyze_combo_m([8, 29, 42], recs)
        r4 = M1.analyze_combo_m([1, 40, 80], recs)
        self.assertEqual(M0.analyze_combo_m([8, 29, 42], recs), r3)
        self.assertEqual(M0.analyze_combo_m([1, 40, 80], recs), r4)


class TestRecommendEquivalence(unittest.TestCase):
    def test_recommend_m4(self):
        recs = DATASETS["real_pool"]
        self.assertEqual(_strip(P0.recommend(recs, None, 5)),
                         _strip(P1.recommend(recs, None, 5)))
        self.assertEqual(_strip(P0.recommend(recs, 20, 3)),
                         _strip(P1.recommend(recs, 20, 3)))

    def test_recommend_m123(self):
        recs = DATASETS["real_pool"]
        for m, last_n in ((1, None), (2, 20), (3, None), (1, 30), (2, None), (3, 40)):
            self.assertEqual(_strip(M0.recommend_m(recs, m, last_n, 5)),
                             _strip(M1.recommend_m(recs, m, last_n, 5)), m)


class TestAnalysisPipelineEquivalence(unittest.TestCase):
    def test_run_analysis_all_datasets(self):
        for name, recs in DATASETS.items():
            if len(recs) < 5:
                continue
            self.assertEqual(_strip(A0.run_analysis(recs, persist=False)),
                             _strip(A1.run_analysis(recs, persist=False)), name)

    def test_run_analysis_with_injected_fx(self):
        recs = DATASETS["real_pool"]
        fx = P1.Features(recs)
        self.assertEqual(_strip(A0.run_analysis(recs, persist=False)),
                         _strip(A1.run_analysis(recs, persist=False, fx=fx)))

    def test_run_analysis_m_all_sizes(self):
        recs = DATASETS["real_pool"]
        for m in (1, 2, 3, 4):
            self.assertEqual(_strip(M0.run_analysis_m(recs, m, persist=False)),
                             _strip(M1.run_analysis_m(recs, m, persist=False)), m)

    def test_compute_gaps_equivalent(self):
        for name, recs in DATASETS.items():
            self.assertEqual(A0.compute_gaps(recs), A1.compute_gaps(recs), name)
        self.assertEqual(A0.compute_gaps([]), A1.compute_gaps([]))

    def test_hot_cold_equivalent(self):
        from analytics import AnalysisParams
        recs = DATASETS["real_pool"]
        params = AnalysisParams()
        self.assertEqual(A0.hot_cold_table(recs, params, P0.Features(recs)),
                         A1.hot_cold_table(recs, params, P1.Features(recs)))


class TestBacktestEquivalence(unittest.TestCase):
    """回测对拍。

    注意：4 码回测路径已从 run_analysis 改为与线上一致的 recommend(top_k=1)，
    属于有意的算法变更（P0-1 回测/线上对齐），因此 4 码结果不再与 legacy 位级相等，
    改为结构校验；m=1/2/3 路径未变，仍与 legacy 逐字段对拍。
    """

    def _assert_backtest4_valid(self, rep):
        self.assertIsInstance(rep, dict)
        for k in ("trials", "hit_distribution", "mean_hits",
                  "hit_at_least_1_pct", "z_vs_baseline", "baseline", "details"):
            self.assertIn(k, rep)
        self.assertEqual(set(rep["hit_distribution"].keys()), {"0", "1", "2", "3", "4"})
        self.assertEqual(len(rep["details"]), rep["trials"])
        for d in rep["details"]:
            self.assertIn("picked", d)
            self.assertIn("hits", d)
            self.assertEqual(len(d["picked"]), 4)

    def test_run_backtest_real_pool(self):
        recs = DATASETS["real_pool"]
        for tw in (10, 20, 30):
            self._assert_backtest4_valid(B1.run_backtest(recs, tw))

    def test_run_backtest_synth(self):
        recs = DATASETS["synth_20"]
        self._assert_backtest4_valid(B1.run_backtest(recs, 15))

    def test_run_backtest_m_all_sizes(self):
        recs = DATASETS["real_pool"]
        for m in (1, 2, 3):
            for tw in (15, 25):
                self.assertEqual(_strip(M0.run_backtest_m(recs, m, tw)),
                                 _strip(M1.run_backtest_m(recs, m, tw)), (m, tw))
        # m=4：算法已改为 recommend 路径，仅做结构校验
        for tw in (15, 25):
            rep = M1.run_backtest_m(recs, 4, tw)
            self.assertEqual(rep["size"], 4)
            self._assert_backtest4_valid(rep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
