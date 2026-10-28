# -*- coding: utf-8 -*-
"""
tests.perf_benchmark —— 优化前后性能对比基准（可独立运行，输出量化报告）

用法（在 lottery_system 目录）：
    python -m tests.perf_benchmark
    python -m tests.perf_benchmark --json      # 输出 JSON 便于归档

冻结的旧实现位于 tests/legacy/，与当前实现输入完全相同；
同一数据池下所有输出的逐字段一致性由 tests.test_optimization_equiv 保证。
本脚本只负责计时对比。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests.legacy import analytics as A0, backtest as B0, msize as M0, predictor as P0
import analytics as A1
import backtest as B1
import msize as M1
import predictor as P1
from tests.test_analytics import real_records, synth_records


def _median_ms(fn, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def build_cases(recs):
    """返回 (场景名, 旧实现调用, 新实现调用, 计时轮数)。"""
    P1.clear_caches()
    return [
        ("analyze_combo(4码) 首次",
         lambda: P0.analyze_combo([3, 17, 42, 65], recs),
         lambda: P1.analyze_combo([3, 17, 42, 65], recs), 1),
        ("analyze_combo(4码) 二次(缓存命中)",
         lambda: P0.analyze_combo([1, 2, 3, 4], recs),
         lambda: P1.analyze_combo([1, 2, 3, 4], recs), 1),
        ("analyze_combo_m(3码) 二次(缓存命中)",
         lambda: M0.analyze_combo_m([8, 29, 42], recs),
         lambda: M1.analyze_combo_m([8, 29, 42], recs), 1),
        ("recommend 自助推荐(4码)",
         lambda: P0.recommend(recs, None, 5),
         lambda: P1.recommend(recs, None, 5), 1),
        ("recommend_m(1码)",
         lambda: M0.recommend_m(recs, 1, None, 5),
         lambda: M1.recommend_m(recs, 1, None, 5), 1),
        ("recommend_m(2码)",
         lambda: M0.recommend_m(recs, 2, None, 5),
         lambda: M1.recommend_m(recs, 2, None, 5), 1),
        ("recommend_m(3码)",
         lambda: M0.recommend_m(recs, 3, None, 5),
         lambda: M1.recommend_m(recs, 3, None, 5), 1),
        ("run_analysis 完整筛选(4码)",
         lambda: A0.run_analysis(recs, persist=False),
         lambda: A1.run_analysis(recs, persist=False), 3),
        ("run_analysis_m(3码)",
         lambda: M0.run_analysis_m(recs, 3, persist=False),
         lambda: M1.run_analysis_m(recs, 3, persist=False), 3),
        ("run_backtest 时间切片(4码,窗20)",
         lambda: B0.run_backtest(recs, 20),
         lambda: B1.run_backtest(recs, 20), 2),
        ("run_backtest_m(3码,窗20)",
         lambda: M0.run_backtest_m(recs, 3, 20),
         lambda: M1.run_backtest_m(recs, 3, 20), 2),
    ]


def run(dataset_name, recs):
    rows = []
    cases = build_cases(recs)
    # 预热 m=3 百分位表（其用例测量缓存命中的稳态延迟）；m4 的两个用例自然覆盖冷/热
    P1.clear_caches()
    M1.analyze_combo_m([8, 29, 42], recs)
    for name, old_fn, new_fn, rounds in cases:
        new_ms = _median_ms(new_fn, rounds)
        old_ms = _median_ms(old_fn, rounds)
        rows.append({
            "dataset": dataset_name,
            "periods": len(recs),
            "case": name,
            "legacy_ms": round(old_ms, 1),
            "optimized_ms": round(new_ms, 1),
            "speedup": round(old_ms / new_ms, 2) if new_ms > 0 else None,
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    datasets = [
        ("real_pool", real_records()),
        ("synth_31", synth_records(31)),
    ]
    all_rows = []
    for ds_name, recs in datasets:
        all_rows.extend(run(ds_name, recs))

    if args.json:
        print(json.dumps(all_rows, ensure_ascii=False, indent=2))
        return 0

    print("=" * 78)
    print("算法模块性能对比（中位数毫秒；数据→输出一致性见 test_optimization_equiv）")
    print("=" * 78)
    cur = None
    for r in all_rows:
        if r["dataset"] != cur:
            cur = r["dataset"]
            print(f"\n[数据集] {cur}  期数={r['periods']}")
            print("%-30s %10s %10s %9s" % ("场景", "旧(ms)", "新(ms)", "加速比"))
            print("-" * 78)
        print("%-30s %10.1f %10.1f %8.2fx"
              % (r["case"], r["legacy_ms"], r["optimized_ms"], r["speedup"]))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
