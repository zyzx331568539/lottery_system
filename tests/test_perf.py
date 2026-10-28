# -*- coding: utf-8 -*-
"""
tests.test_perf —— 性能回归测试（保守阈值，防止优化被回退；不追求精确计时）

阈值取值依据 tests/perf_benchmark.py 在 55 期真实池上的实测中位加速比，
仅取实测值的 1/3 ~ 1/2 作为门槛，留出 CI/负载抖动余量：
  recommend(m2) 实测 37x  → 门槛 5x
  recommend(m4) 实测 5.6x → 门槛 2x
  backtest      实测 3.1x → 门槛 1.8x
  analyze 缓存命中 62ms   → 门槛 300ms
"""
from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests.legacy import backtest as B0, msize as M0, predictor as P0
import backtest as B1
import msize as M1
import predictor as P1
from tests.test_analytics import synth_records


def _ms(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


class TestPerformanceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = synth_records(31)

    def test_recommend_m2_speedup(self):
        old = _ms(lambda: M0.recommend_m(self.recs, 2, None, 5))
        new = _ms(lambda: M1.recommend_m(self.recs, 2, None, 5))
        self.assertLess(new * 5, old, f"m2 recommend 加速不足: old={old:.0f}ms new={new:.0f}ms")

    def test_recommend_m3_speedup(self):
        old = _ms(lambda: M0.recommend_m(self.recs, 3, None, 5))
        new = _ms(lambda: M1.recommend_m(self.recs, 3, None, 5))
        self.assertLess(new * 3, old, f"m3 recommend 加速不足: old={old:.0f}ms new={new:.0f}ms")

    def test_recommend_m4_speedup(self):
        # NumPy 向量化后实测约 20×+（OPT-7），门槛 8× 留抖动余量
        old = _ms(lambda: P0.recommend(self.recs, None, 5))
        new = _ms(lambda: P1.recommend(self.recs, None, 5))
        self.assertLess(new * 8, old, f"m4 recommend 加速不足: old={old:.0f}ms new={new:.0f}ms")

    def test_backtest_speedup(self):
        old = _ms(lambda: B0.run_backtest(self.recs, 20))
        new = _ms(lambda: B1.run_backtest(self.recs, 20))
        self.assertLess(new * 1.8, old, f"backtest 加速不足: old={old:.0f}ms new={new:.0f}ms")

    def test_analyze_cache_hit_latency(self):
        recs = self.recs
        P1.clear_caches()
        P1.analyze_combo([3, 17, 42, 65], recs)          # 冷启动建表
        hot = _ms(lambda: P1.analyze_combo([1, 2, 3, 4], recs))
        self.assertLess(hot, 300, f"缓存命中延迟过高: {hot:.0f}ms")
        M1.analyze_combo_m([8, 29, 42], recs)            # 预热 m3
        hot3 = _ms(lambda: M1.analyze_combo_m([1, 40, 80], recs))
        self.assertLess(hot3, 300, f"m3 缓存命中延迟过高: {hot3:.0f}ms")

    def test_cache_invalidation_on_pool_change(self):
        """数据池变化后必须重建（缓存键失效），且重建结果与新数据一致"""
        recs = self.recs
        P1.clear_caches()
        r1 = P1.analyze_combo([3, 17, 42, 65], recs)
        changed = [dict(r) for r in recs]
        changed.append({"period": changed[-1]["period"] + 1,
                        "date": None, "source": "test",
                        "numbers": list(range(21, 41))})
        r2 = P1.analyze_combo([3, 17, 42, 65], changed)
        r2_legacy = P0.analyze_combo([3, 17, 42, 65], changed)
        self.assertEqual(r2, r2_legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
