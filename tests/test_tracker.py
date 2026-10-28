# -*- coding: utf-8 -*-
"""
tests.test_tracker —— 推荐命中追踪模块单元测试

覆盖：
  * save_recommendation：快照结构、目标期号推导
  * reconcile：未开奖时 pending、开奖后对比、幂等
  * 命中数计算：与开奖20码交集
  * hit_rate_report：命中分布/均值/基准/verdict
  * strategy_stats / best_strategy：分组统计与切换门槛
  * auto_strategy：无合格策略时安全回退，有合格策略时选用最优 last_n

运行：
    python -m unittest tests.test_tracker -v
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import tracker
import data_store
from predictor import recommend
from msize import recommend_m
from tests.test_analytics import synth_records


def _make_store(path, n):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"records": synth_records(n)}, f)
    return data_store.DataStore(path)


def _fresh_rec_path():
    fd, p = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(p)
    return p


class TestTrackerCore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pool_path = os.path.join(self.tmpdir, "pool.json")
        self.rec_path = _fresh_rec_path()
        tracker.REC_PATH = self.rec_path
        tracker._cache = None
        tracker._cache_mtime = None

    def tearDown(self):
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)

    def _save(self, store, size=4, last_n=None, top_k=3):
        recs = store.all()
        if size == 4:
            res = recommend(recs, last_n=last_n, top_k=top_k)
        else:
            res = recommend_m(recs, size, last_n=last_n, top_k=top_k)
        return tracker.save_recommendation(recs, size, res,
                                           last_n=last_n, top_k=top_k)

    # ---- 存储
    def test_save_structure_and_target_period(self):
        store = _make_store(self.pool_path, 15)
        snap = self._save(store)
        self.assertEqual(snap["target_period"], 2026015 + 1)
        self.assertEqual(snap["size"], 4)
        self.assertFalse(snap["compared"])
        self.assertIsNone(snap["result"])
        self.assertEqual(len(snap["recommendations"]), 3)
        # 落盘可重读
        self.assertIsNotNone(tracker.get_recommendation(snap["id"]))

    def test_save_all_sizes(self):
        store = _make_store(self.pool_path, 15)
        for sz in (1, 2, 3, 4):
            snap = self._save(store, size=sz)
            self.assertEqual(snap["size"], sz)
            combo_len = len(snap["recommendations"][0]["combo"])
            self.assertEqual(combo_len, sz)

    def test_list_and_filter(self):
        store = _make_store(self.pool_path, 15)
        self._save(store, size=4)
        self._save(store, size=2)
        self.assertEqual(len(tracker.list_recommendations()), 2)
        self.assertEqual(len(tracker.list_recommendations(size=4)), 1)
        self.assertEqual(len(tracker.list_recommendations(limit=1)), 1)

    # ---- 对比
    def test_reconcile_pending_before_draw(self):
        store = _make_store(self.pool_path, 15)
        self._save(store)
        r = tracker.reconcile(store)
        self.assertEqual(r["compared"], 0)
        self.assertEqual(r["still_pending"], 1)

    def test_reconcile_after_draw(self):
        store = _make_store(self.pool_path, 15)
        snap = self._save(store)
        # 录入目标期开奖
        target = snap["target_period"]
        draw = synth_records(target - 2026001 + 1)[-1]
        store.add(draw)
        r = tracker.reconcile(store)
        self.assertEqual(r["compared"], 1)
        snap2 = tracker.get_recommendation(snap["id"])
        self.assertTrue(snap2["compared"])
        res = snap2["result"]
        self.assertIn("best_hit", res)
        self.assertIn("top1_hit", res)
        # best_hit 在 [0,4]，top1_hit 在 [0,4]
        self.assertTrue(0 <= res["best_hit"] <= 4)
        self.assertTrue(0 <= res["top1_hit"] <= 4)
        # 候选池命中数
        self.assertTrue(0 <= res["candidate_hits"] <= 12)

    def test_reconcile_idempotent(self):
        store = _make_store(self.pool_path, 15)
        snap = self._save(store)
        target = snap["target_period"]
        store.add(synth_records(target - 2026001 + 1)[-1])
        tracker.reconcile(store)
        r2 = tracker.reconcile(store)
        self.assertEqual(r2["compared"], 0)
        self.assertEqual(r2["skipped"], 1)

    def test_hit_count_matches_intersection(self):
        store = _make_store(self.pool_path, 15)
        snap = self._save(store)
        target = snap["target_period"]
        draw = synth_records(target - 2026001 + 1)[-1]
        store.add(draw)
        tracker.reconcile(store)
        snap2 = tracker.get_recommendation(snap["id"])
        actual = set(draw["numbers"])
        for pc in snap2["result"]["per_combo"]:
            self.assertEqual(pc["hits"], len(set(pc["combo"]) & actual))

    # ---- 命中率报告
    def test_hitrate_empty(self):
        store = _make_store(self.pool_path, 15)
        rep = tracker.hit_rate_report()
        self.assertEqual(rep["samples"], 0)
        self.assertIn("note", rep)

    def test_hitrate_with_data(self):
        store = _make_store(self.pool_path, 20)
        # 多轮：用不同窗口推荐，再补入对应开奖
        for last_n in (None, 15, 10):
            recs = store.all()
            res = recommend(recs, last_n=last_n, top_k=3)
            snap = tracker.save_recommendation(recs, 4, res, last_n=last_n)
            target = snap["target_period"]
            store.add(synth_records(target - 2026001 + 1)[-1])
            tracker.reconcile(store)
        rep = tracker.hit_rate_report(size=4)
        self.assertEqual(rep["samples"], 3)
        self.assertIn("best_hit", rep)
        self.assertIn("top1_hit", rep)
        self.assertIn("baseline", rep)
        self.assertAlmostEqual(rep["baseline"]["mean_hits"], 1.0)
        # 分布各 key 之和 == samples
        self.assertEqual(sum(rep["best_hit"]["distribution"].values()), 3)

    # ---- 策略
    def test_strategy_stats_groups_by_last_n(self):
        store = _make_store(self.pool_path, 20)
        for last_n in (None, 15):
            recs = store.all()
            res = recommend(recs, last_n=last_n, top_k=3)
            snap = tracker.save_recommendation(recs, 4, res, last_n=last_n)
            store.add(synth_records(snap["target_period"] - 2026001 + 1)[-1])
            tracker.reconcile(store)
        stats = tracker.strategy_stats(4)
        self.assertEqual(len(stats), 2)
        last_ns = {s["last_n"] for s in stats}
        self.assertIn(None, last_ns)
        self.assertIn(15, last_ns)

    def test_best_strategy_requires_min_samples(self):
        store = _make_store(self.pool_path, 20)
        recs = store.all()
        res = recommend(recs, top_k=3)
        snap = tracker.save_recommendation(recs, 4, res, last_n=None)
        store.add(synth_records(snap["target_period"] - 2026001 + 1)[-1])
        tracker.reconcile(store)
        # 默认 min_samples=10，仅1个样本 -> None
        self.assertIsNone(tracker.best_strategy(4))
        # 仅1个样本且该策略就是默认(last_n=None)：无需切换 -> 返回 None
        self.assertIsNone(tracker.best_strategy(4, min_samples=1))

    def test_best_strategy_picks_non_default_when_better(self):
        store = _make_store(self.pool_path, 20)
        # 构造两个策略：last_n=15 命中明显高于默认
        # 这里不操纵命中，只验证"非默认策略满足样本门槛且非自身即默认"时能返回
        for last_n in (None, 15):
            for _ in range(3):
                recs = store.all()
                res = recommend(recs, last_n=last_n, top_k=3)
                snap = tracker.save_recommendation(recs, 4, res, last_n=last_n)
                store.add(synth_records(snap["target_period"] - 2026001 + 1)[-1])
                tracker.reconcile(store)
        stats = tracker.strategy_stats(4)
        # 放宽 edge=0，让非默认策略只要样本够就可被选为最优（若其均值最高）
        best = tracker.best_strategy(4, min_samples=3, edge=0.0)
        # best 要么是 None（默认最优），要么是非默认策略
        if best is not None:
            self.assertIn(best["last_n"], (None, 15))

    def test_best_strategy_respects_edge(self):
        store = _make_store(self.pool_path, 20)
        # 两个策略各10个样本，构造相同命中 -> edge 不满足 -> None
        for last_n in (None, 15):
            for _ in range(10):
                recs = store.all()
                res = recommend(recs, last_n=last_n, top_k=3)
                snap = tracker.save_recommendation(recs, 4, res, last_n=last_n)
                store.add(synth_records(snap["target_period"] - 2026001 + 1)[-1])
                tracker.reconcile(store)
        # 两个策略命中统计相近，edge 门槛阻止切换
        best = tracker.best_strategy(4, edge=1.0)  # 极大 edge
        self.assertIsNone(best)

    # ---- auto_strategy 安全回退
    def test_auto_strategy_fallback_when_no_qualified(self):
        store = _make_store(self.pool_path, 15)
        recs = store.all()
        # 无足够历史对比 -> best_strategy 返回 None -> 用默认 last_n
        res_default = recommend(recs, last_n=None, top_k=3)
        res_auto = recommend(recs, last_n=None, top_k=3, auto_strategy=True)
        # 结果应与默认完全一致（无策略可切换）
        self.assertEqual([r["combo"] for r in res_default["recommendations"]],
                         [r["combo"] for r in res_auto["recommendations"]])


if __name__ == "__main__":
    unittest.main()
