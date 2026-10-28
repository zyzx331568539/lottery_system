# -*- coding: utf-8 -*-
"""
tests.test_robustness —— 优化新增的边界条件/异常输入校验 + 内部等价性单元测试

两部分：
1. 鲁棒性：非法类型、空数据、越界参数、布尔值伪装整数等必须抛出明确异常；
   合法路径行为不变。
2. 内部结构等价：位掩码重采样特征、滑窗增量特征、权重模板与缓存，
   与“朴素全量构建”在浮点/整数层面逐项相等。

运行（在 lottery_system 目录）：
    python -m unittest tests.test_robustness -v
"""
from __future__ import annotations

import os
import random
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import analytics
import backtest
import msize
import predictor
from data_store import ValidationError
from predictor import (Features, _RecordIndex, _SampledFeatures,
                       _decay_template, slide_window_features,
                       validate_records)
from tests.test_analytics import real_records, synth_records


# ============================================================ 输入校验
class TestInputValidation(unittest.TestCase):
    def test_validate_user_combo_bad(self):
        for bad in ([], [0], [81], [1, 1], ["7"], [1.5], [True],
                    list(range(1, 6)), None):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                predictor.validate_user_combo(bad)
        for bad in ([], [0], [81], [1, 2, 2]):
            with self.assertRaises(ValidationError):
                predictor.validate_user_combo(bad, 1, 10)

    def test_validate_user_combo_good(self):
        self.assertEqual(predictor.validate_user_combo([80, 3, 42, 7],
                                                       min_size=1, max_size=10),
                         [3, 7, 42, 80])
        self.assertEqual(predictor.validate_user_combo([80, 3, 42, 7]),
                         [3, 7, 42, 80])

    def test_validate_records(self):
        with self.assertRaises(ValidationError):
            validate_records("not-list")
        with self.assertRaises(ValidationError):
            validate_records([])
        with self.assertRaises(ValidationError):
            validate_records([{"numbers": list(range(1, 20))}])  # 只有19码
        with self.assertRaises(ValidationError):
            validate_records([{"numbers": list(range(1, 19)) + [19, 19]}])  # 重复
        with self.assertRaises(ValidationError):
            validate_records([{"numbers": list(range(1, 19)) + [20, 81]}])  # 越界
        self.assertEqual(len(validate_records(synth_records(2))), 2)

    def test_recommend_bad_args(self):
        recs = synth_records(10)
        with self.assertRaises(ValidationError):
            predictor.recommend([])
        with self.assertRaises(ValidationError):
            predictor.recommend(synth_records(3))
        for bad in (0, -1, 1.5, True, "10"):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                predictor.recommend(recs, last_n=bad)
        for bad in (0, -3, 1.0, False, "5"):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                predictor.recommend(recs, top_k=bad)

    def test_bootstrap_rounds_validation(self):
        recs = synth_records(10)
        targets = [(1, 2, 3, 4)]
        for bad in (0, -1, 1.5, True, "200"):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                predictor.bootstrap_stability(recs, targets, rounds=bad)
            with self.assertRaises(ValidationError, msg=repr(bad)):
                msize.bootstrap_stability_m(recs, [(1,)], 1, rounds=bad)
        # 空数据安全返回 0 稳定度（不抛异常）
        self.assertEqual(predictor.bootstrap_stability([], targets)[(1, 2, 3, 4)], 0.0)

    def test_analyze_empty_pool(self):
        with self.assertRaises(ValidationError):
            predictor.analyze_combo([1, 2, 3, 4], [])
        with self.assertRaises(ValidationError):
            msize.analyze_combo_m([7], [])

    def test_pipeline_bad_records(self):
        with self.assertRaises(ValueError):
            analytics.run_analysis([])
        with self.assertRaises(ValueError):
            analytics.run_analysis(synth_records(3))
        bad_records = synth_records(5) + [{"numbers": [1] * 20}]  # 末期号码重复
        with self.assertRaises(ValidationError):
            analytics.run_analysis(bad_records)
        with self.assertRaises(ValueError):
            msize.run_analysis_m([], 2)

    def test_backtest_bad_args(self):
        recs = synth_records(10)
        with self.assertRaises(ValueError):
            backtest.run_backtest(synth_records(4), 3)
        for bad in (3.0, True, "10"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                backtest.run_backtest(recs, train_window=bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                msize.run_backtest_m(recs, 2, train_window=bad)
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                backtest.run_backtest(recs, 5, bootstrap_rounds=bad)

    def test_msize_size_validation(self):
        recs = synth_records(10)
        for bad in (0, 5, -1, 2.5, True, "3", None):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                msize.run_analysis_m(recs, bad)
            with self.assertRaises(ValidationError, msg=repr(bad)):
                msize.recommend_m(recs, bad)
            with self.assertRaises(ValidationError, msg=repr(bad)):
                msize.run_backtest_m(recs, bad, 5)

    def test_recommend_m_bad_args(self):
        recs = synth_records(10)
        with self.assertRaises(ValidationError):
            msize.recommend_m(synth_records(3), 2)
        for bad in (0, -2, 1.5, True):
            with self.assertRaises(ValidationError):
                msize.recommend_m(recs, 2, last_n=bad)
            with self.assertRaises(ValidationError):
                msize.recommend_m(recs, 2, top_k=bad)


# ============================================================ 内部结构等价
class TestDecayTemplate(unittest.TestCase):
    def test_template_matches_inline(self):
        for n in (0, 1, 5, 10, 55):
            tpl = _decay_template(n)
            for i, w in enumerate(tpl):
                self.assertEqual(w, 0.5 ** ((n - 1 - i) / 10.0), (n, i))


class TestSampledFeatures(unittest.TestCase):
    def test_sampled_equals_naive_rebuild_m4(self):
        """固定 chosen 序列下，位掩码特征必须与朴素 Features(sub) 逐项相等"""
        recs = real_records()
        rng = random.Random(20260920)
        idx = list(range(len(recs)))
        index = _RecordIndex(recs)
        for trial in range(5):
            chosen = [rng.choice(idx) for _ in idx]
            sub = [recs[k] for k in chosen]
            naive = Features(sub)
            fast = _SampledFeatures(recs, chosen, index)
            self.assertEqual(naive.freq, fast.freq, trial)
            self.assertEqual(naive.wfreq, fast.wfreq, trial)
            self.assertEqual((naive.m_f, naive.sd_f, naive.m_w, naive.sd_w,
                              naive.m_p, naive.sd_p, naive.m_4, naive.sd_4),
                             (fast.m_f, fast.sd_f, fast.m_w, fast.sd_w,
                              fast.m_p, fast.sd_p, fast.m_4, fast.sd_4), trial)
            for a, b in ((1, 80), (3, 17), (42, 65), (70, 71), (5, 77)):
                self.assertEqual(naive.z_pair(a, b), fast.z_pair(a, b),
                                 (trial, a, b))
            for c in ((3, 17, 42, 65), (1, 2, 3, 4), (70, 71, 79, 80)):
                self.assertEqual(naive.z_four(c), fast.z_four(c), (trial, c))
            self.assertEqual(naive.candidates(), fast.candidates(), trial)
            self.assertEqual(naive.ranked_combos(), fast.ranked_combos(), trial)

    def test_vectorized_freq_wfreq_znum_arrays(self):
        """OPT-8：构造期的频次/加权频次/81维z向量与朴素 Features(sub) 逐值相等"""
        recs = real_records()
        rng = random.Random(20260920)
        idx = list(range(len(recs)))
        index = predictor._RecordIndex(recs)
        for trial in range(8):
            chosen = [rng.choice(idx) for _ in idx]
            sub = [recs[k] for k in chosen]
            naive = Features(sub)
            fast = predictor._SampledFeatures(recs, chosen, index)
            # 整数频次数组
            for n in range(1, 81):
                self.assertEqual(int(fast._freq_arr[n]), naive.freq[n], (trial, n))
            # 加权频次数组（cumsum 顺序求和，必须位级相等）
            for n in range(1, 81):
                self.assertEqual(float(fast._wfreq_arr[n]), naive.wfreq[n], (trial, n))
            # 81 维 z 向量 vs 标量 z_num
            for n in range(1, 81):
                self.assertEqual(float(fast._znum_arr[n]), naive.z_num(n), (trial, n))
            # Counter 视图键集合与取值
            self.assertEqual(fast.freq, naive.freq, trial)
            self.assertEqual(fast.wfreq, naive.wfreq, trial)

    def test_candidates_vectorized_equals_naive(self):
        """OPT-8：向量化候选集与 Features.candidates 在多轮重采样上完全一致"""
        for recs in (real_records(), synth_records(31), synth_records(12)):
            rng = random.Random(4242)
            idx = list(range(len(recs)))
            index = predictor._RecordIndex(recs)
            for _ in range(30):
                chosen = [rng.choice(idx) for _ in idx]
                fast = predictor._SampledFeatures(recs, chosen, index)
                naive = Features([recs[k] for k in chosen])
                self.assertEqual(fast.candidates_vectorized(),
                                 naive.candidates())
                self.assertEqual(len(fast.candidates_vectorized()), 12)

    def test_cumsum_is_sequential_order(self):
        """wfreq 向量化依赖 cumsum 逐列顺序累加：显式锁定该语义（防 NumPy 行为变更）"""
        rnd = random.Random(20261020)
        w = np.array([rnd.uniform(-2, 2) for _ in range(55)], dtype=np.float64)
        for shape in ((3, 55), (81, 55)):
            m = np.array([[rnd.choice((0.0, 1.0)) for _ in range(55)]
                          for _ in range(shape[0])], dtype=np.float64)
            got = (m * w).cumsum(axis=1)[:, -1]
            for i in range(shape[0]):
                seq = 0.0
                for j in range(55):
                    seq += m[i, j] * w[j]
                self.assertEqual(float(got[i]), seq, (shape, i))

    def test_sampled_m3_equals_naive_rebuild(self):
        recs = real_records()
        rng = random.Random(777)
        idx = list(range(len(recs)))
        index = _RecordIndex(recs, with_triples=True)
        chosen = [rng.choice(idx) for _ in idx]
        sub = [recs[k] for k in chosen]
        naive = msize.FeaturesM(sub)
        fast = msize._SampledFeaturesM(recs, chosen, index)
        for c in ((1, 2, 80), (3, 17, 42), (50, 51, 52), (8, 29, 42)):
            self.assertEqual(naive.z_triple(c), fast.z_triple(c), c)
        self.assertEqual((naive.m_3, naive.sd_3), (fast.m_3, fast.sd_3))
        self.assertIsInstance(fast, msize.FeaturesM)
        self.assertEqual(msize.ranked_combos_m(naive, naive.candidates(), 3),
                         msize.ranked_combos_m(fast, fast.candidates(), 3))


class TestSlidingFeatures(unittest.TestCase):
    def _assert_features_equal(self, fresh, slid):
        self.assertEqual(fresh.freq, slid.freq)
        self.assertEqual(fresh.pair, slid.pair)
        self.assertEqual(fresh.four, slid.four)
        self.assertEqual(fresh.wfreq, slid.wfreq)
        self.assertEqual((fresh.m_f, fresh.sd_f, fresh.m_w, fresh.sd_w,
                          fresh.m_p, fresh.sd_p, fresh.m_4, fresh.sd_4),
                         (slid.m_f, slid.sd_f, slid.m_w, slid.sd_w,
                          slid.m_p, slid.sd_p, slid.m_4, slid.sd_4))
        self.assertEqual(fresh.candidates(), slid.candidates())
        self.assertEqual(fresh.ranked_combos(), slid.ranked_combos())

    def test_slide_features_m4(self):
        recs = synth_records(30)
        tw = 12
        fx = Features(recs[:tw])
        for step in range(1, 12):
            end = tw + step
            window = recs[end - tw:end]
            slide_window_features(fx, recs[end - tw - 1]["numbers"],
                                  recs[end - 1]["numbers"], window)
            self._assert_features_equal(Features(window), fx)

    def test_slide_features_real_pool(self):
        recs = real_records()
        tw = 20
        fx = Features(recs[:tw])
        # 与回测同构：i=tw 用初始窗 recs[0:tw]，之后每步滑入 recs[i-1]、滑出 recs[i-tw-1]；
        # 每一步都必须与对该窗口朴素全量构建的 Features 完全相等。
        self._assert_features_equal(Features(recs[:tw]), fx)
        for end in range(tw + 1, len(recs)):
            window = recs[end - tw:end]
            slide_window_features(fx, recs[end - tw - 1]["numbers"],
                                  recs[end - 1]["numbers"], window)
            self._assert_features_equal(Features(window), fx)

    def test_slide_features_m3(self):
        recs = synth_records(25)
        tw = 12
        fx = msize.FeaturesM(recs[:tw])
        for end in range(tw + 1, 18):
            window = recs[end - tw:end]
            slide_window_features(fx, recs[end - tw - 1]["numbers"],
                                  recs[end - 1]["numbers"], window)
            fresh = msize.FeaturesM(window)
            self.assertEqual(fresh.triple, fx.triple)
            self.assertEqual((fresh.m_3, fresh.sd_3), (fx.m_3, fx.sd_3))
            self.assertEqual(msize.ranked_combos_m(fresh, fresh.candidates(), 3),
                             msize.ranked_combos_m(fx, fx.candidates(), 3))


class TestNumpyVectorization(unittest.TestCase):
    def test_neumaier_matches_builtin_sum(self):
        """向量化 Neumaier 补偿求和必须与 CPython 内置 sum()（纯 Python float）位级一致"""
        import random as _random
        from predictor import _neumaier_sum
        rnd = _random.Random(20261019)
        # 混合普通随机值与大小量级悬殊值（制造真实舍入补偿）
        scales = (-6, 6), (-300, 300), (0, 1)
        for k in (1, 2, 3, 4, 6, 10):
            rows = []
            for _ in range(20000):
                lo, hi = rnd.choice(scales)
                rows.append([rnd.uniform(lo, hi) for _ in range(k)])
            arr = np.array(rows, dtype=np.float64)
            got = _neumaier_sum([arr[:, j] for j in range(k)])
            for i, row in enumerate(rows):
                self.assertEqual(float(got[i]), sum(row), (k, i, row))

    def test_vectorized_ranked_equals_python_all_rounds(self):
        """自助重采样每轮：NumPy 评分表（分数/分量/排序）与 Python ranked_combos 完全相等"""
        for recs in (real_records(), synth_records(31), synth_records(12)):
            rng = random.Random(20260920)
            idx = list(range(len(recs)))
            index = predictor._RecordIndex(recs)
            for _ in range(8):
                chosen = [rng.choice(idx) for _ in idx]
                f = predictor._SampledFeatures(recs, chosen, index)
                pool = f.candidates()
                self.assertEqual(f.ranked_combos_vectorized(pool),
                                 f.ranked_combos(pool))
                # top5 快捷方法与完整排序的前5一致
                top5_slow = {r["combo"] for r in f.ranked_combos(pool)[:5]}
                self.assertEqual(f.top5_combos_vectorized(pool), top5_slow)

    def test_vectorized_arbitrary_pool_sizes(self):
        """向量化路径对 4..12 任意池大小（_combo_structures 各规格）都与 Python 版一致"""
        recs = synth_records(20)
        index = predictor._RecordIndex(recs)
        f = predictor._SampledFeatures(recs, list(range(len(recs))), index)
        for size in (4, 5, 8, 10, 12):
            pool = list(range(1, size + 1))
            self.assertEqual(f.ranked_combos_vectorized(pool),
                             f.ranked_combos(pool), size)

    def test_occ_matrix_matches_masks(self):
        """出现矩阵 occ 与位掩码 nmask 对任意号码/期一致"""
        recs = real_records()
        index = predictor._RecordIndex(recs)
        n = len(recs)
        for num in (1, 27, 80):
            for k in range(n):
                self.assertEqual(bool(index.occ[num, k]),
                                 bool(index.nmask[num] & (1 << k)))


class TestOpt9BatchChoices(unittest.TestCase):
    """OPT-9：批量预生成抽样索引必须逐元素复刻逐次 rng.choice 序列。"""

    def test_choice_matrix_matches_sequential(self):
        for n in (1, 2, 3, 5, 8, 12, 31, 56, 63, 64, 100):
            for seed in (0, 1, 20260920):
                rng0 = random.Random(seed)
                idx = list(range(n))
                ref = np.array([[rng0.choice(idx) for _ in range(n)]
                                for _ in range(20)], dtype=np.int64)
                rng1 = random.Random(seed)
                got = predictor._bootstrap_choice_matrix(rng1, n, 20)
                self.assertEqual(got.shape, (20, n), (n, seed))
                self.assertTrue(np.array_equal(got, ref), (n, seed))

    def test_choice_matrix_real_shape_200_rounds(self):
        """真实生产形态：真实池期数 ×200 轮，逐行等于旧实现的抽样序列。"""
        recs = real_records()
        n = len(recs)
        rng0 = random.Random(predictor.BOOTSTRAP_SEED)
        idx = list(range(n))
        ref = [[rng0.choice(idx) for _ in idx] for _ in range(200)]
        rng1 = random.Random(predictor.BOOTSTRAP_SEED)
        got = predictor._bootstrap_choice_matrix(rng1, n, 200)
        for r in range(200):
            self.assertEqual(got[r].tolist(), ref[r], r)

    def test_sampled_features_accept_ndarray_chosen(self):
        """ndarray chosen（bincount 计数）与 list chosen（逐槽 +=）构造结果完全相同。"""
        recs = real_records()
        rng = random.Random(99)
        idx = list(range(len(recs)))
        index = predictor._RecordIndex(recs)
        for _ in range(5):
            chosen = [rng.choice(idx) for _ in idx]
            f_list = predictor._SampledFeatures(recs, chosen, index)
            f_arr = predictor._SampledFeatures(
                recs, np.asarray(chosen, dtype=np.int64), index)
            self.assertEqual(f_list._mult, f_arr._mult)
            self.assertEqual(f_list.freq, f_arr.freq)
            self.assertEqual(f_list.wfreq, f_arr.wfreq)
            np.testing.assert_array_equal(f_list._znum_arr, f_arr._znum_arr)
            self.assertEqual(f_list.ranked_combos_vectorized(),
                             f_arr.ranked_combos_vectorized())


class TestOpt9MScoreVectorization(unittest.TestCase):
    """OPT-9：m=1/2/3 向量化评分/排序与标量 ranked_combos_m 逐行位级相等。"""

    def test_vectorized_rows_equal_scalar_all_rounds(self):
        for recs in (real_records(), synth_records(31), synth_records(12)):
            for m, seed in ((1, 7), (2, 4242), (3, 20260920)):
                rng = random.Random(seed)
                idx = list(range(len(recs)))
                index = predictor._RecordIndex(recs, with_triples=(m == 3))
                cls = (msize._SampledFeaturesM if m == 3
                       else predictor._SampledFeatures)
                for _ in range(6):
                    chosen = [rng.choice(idx) for _ in idx]
                    f = cls(recs, chosen, index)
                    pool = f.candidates_vectorized()
                    slow = msize.ranked_combos_m(f, pool, m)
                    fast = msize.ranked_combos_vectorized_m(f, pool, m)
                    self.assertEqual(fast, slow, (len(recs), m))
                    top5_slow = {r["combo"] for r in slow[:5]}
                    self.assertEqual(
                        msize.top5_combos_vectorized_m(f, pool, m), top5_slow)

    def test_vectorized_rows_arbitrary_pool_sizes(self):
        recs = synth_records(20)
        chosen = list(range(len(recs)))
        for m in (1, 2, 3):
            index = predictor._RecordIndex(recs, with_triples=(m == 3))
            cls = (msize._SampledFeaturesM if m == 3
                   else predictor._SampledFeatures)
            f = cls(recs, chosen, index)
            for p in (m, m + 1, 5, 8, 12):
                pool = list(range(1, p + 1))
                self.assertEqual(
                    msize.ranked_combos_vectorized_m(f, pool, m),
                    msize.ranked_combos_m(f, pool, m), (m, p))

    def test_m3_vectorized_works_without_tmask_index(self):
        """bootstrap 生产路径不建 tmask：向量化 m3 评分仍与朴素全量构建一致；
        标量 z_triple 在无 tmask 时给出明确错误而不是隐晦 AttributeError。"""
        recs = synth_records(20)
        index = predictor._RecordIndex(recs)          # with_triples=False
        self.assertIsNone(index.tmask)
        f = msize._SampledFeaturesM(recs, list(range(len(recs))), index)
        rows = msize.ranked_combos_vectorized_m(f, f.candidates_vectorized(), 3)
        naive = msize.FeaturesM(recs)
        self.assertEqual(rows,
                         msize.ranked_combos_m(naive, naive.candidates(), 3))
        with self.assertRaises(RuntimeError):
            f.z_triple((1, 2, 3))


class TestObservedScoreCache(unittest.TestCase):
    def test_cache_consistency(self):
        recs = real_records()
        fx = Features(recs)
        first = fx.observed_scores_sorted()
        second = fx.observed_scores_sorted()
        self.assertIs(first, second)
        self.assertEqual(sorted(fx.combo_score(c)[0] for c in fx.four), first)

    def test_pool_level_cache_hit_and_invalidation(self):
        predictor.clear_caches()
        recs = real_records()
        fx = Features(recs)
        first = predictor.get_observed_scores(recs, 4, fx.observed_scores_sorted)
        second = predictor.get_observed_scores(recs, 4, self._must_not_build)
        self.assertIs(first, second)  # 同数据池 → 命中同一对象
        # m=2/m=3 走 msize 构建器
        fxm = msize.FeaturesM(recs)
        s2 = predictor.get_observed_scores(
            recs, 2, lambda: sorted(msize.combo_score_m(fxm, c)[0]
                                    for c in fxm.pair.keys()))
        self.assertIs(s2, predictor.get_observed_scores(recs, 2, self._must_not_build))
        # 数据池内容变化 → 签名变化 → 必须重建，绝不返回旧表
        changed = [dict(r) for r in recs]
        changed[-1] = {"period": changed[-1]["period"],
                       "date": changed[-1]["date"], "source": "x",
                       "numbers": sorted(set(range(1, 21)))}
        third = predictor.get_observed_scores(changed, 4,
                                              lambda: Features(changed).observed_scores_sorted())
        self.assertIsNot(third, first)
        self.assertEqual(third, Features(changed).observed_scores_sorted())
        predictor.clear_caches()

    @staticmethod
    def _must_not_build():
        raise AssertionError("缓存未命中，触发了不应发生的重建")


if __name__ == "__main__":
    unittest.main(verbosity=2)
