# -*- coding: utf-8 -*-
"""
backtest.py —— 算法效果历史回测

方法
====
* 严格时间切片：在第 i 期开奖前，只允许使用 records[:i]（或其最近 train_window 期），
  调用与线上完全相同的 run_analysis() 得到4码，再用 records[i] 核对命中 0–4 个。
* 随机基准：随机选4码时命中数服从超几何分布 Hypergeometric(80, 20, 4)
    P(h=k) = C(20,k)·C(60,4-k) / C(80,4)
  任何科学筛选若连基准都跑不赢，就不具备实证有效性（这是最重要的诚实检验）。
* 不确定性：固定种子 bootstrap 200 轮，给出"平均命中数"的95%区间。
"""
from __future__ import annotations

import math
import random
import time
from math import comb

from analytics import AnalysisParams, run_analysis
from audit_log import PerfTimer, get_logger

# ------------------------------------------------------------ 超几何基准（精确值）
_C80_4 = comb(80, 4)
HYPER_BASELINE = {
    k: comb(20, k) * comb(60, 4 - k) / _C80_4 for k in range(5)
}
# E[h]=4·20/80=1；Var=4·(20/80)(60/80)·(80-4)/(80-1)
BASELINE_MEAN = 4 * 20 / 80
BASELINE_VAR = 4 * 0.25 * 0.75 * (80 - 4) / (80 - 1)
BASELINE_SD = math.sqrt(BASELINE_VAR)

BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20260921


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def run_backtest(records, train_window: int | None = None,
                 params: AnalysisParams | None = None,
                 bootstrap_rounds: int = BOOTSTRAP_ROUNDS) -> dict:
    """
    滑动回测。
      train_window：训练窗口（None=用全部历史；否则每期只用最近 train_window 期）
    """
    params = params or AnalysisParams()
    logger = get_logger()
    if len(records) < 6:
        raise ValueError("历史数据至少需要6期才能回测（5期训练+1期验证）")
    tw = train_window or (len(records) - 1)
    if tw < 5 or tw >= len(records):
        raise ValueError("train_window 必须 >=5 且 < 总期数(%d)" % len(records))

    t0 = time.perf_counter()
    with PerfTimer("backtest", logger=logger,
                   extra={"trials": len(records) - tw}):
        details = []
        hits = []
        for i in range(tw, len(records)):
            train = records[i - tw:i]
            result = run_analysis(train, params=params, persist=False)
            picked = result["numbers"]
            actual = records[i]["numbers"]
            h = len(set(picked) & set(actual))
            hits.append(h)
            details.append({
                "target_period": records[i]["period"],
                "target_date": records[i].get("date"),
                "picked": picked,
                "hits": h,
                "hit_numbers": sorted(set(picked) & set(actual)),
                "combo_score": result["combo_score"],
            })

        trials = len(hits)
        dist = {k: hits.count(k) for k in range(5)}
        dist_pct = {k: dist[k] / trials * 100 for k in range(5)}
        mean_h = sum(hits) / trials
        ge1 = sum(1 for h in hits if h >= 1) / trials
        ge2 = sum(1 for h in hits if h >= 2) / trials

        # 与基准的差异（大样本近似 z：(均值-1)/(σ/√n)）
        z_vs_base = ((mean_h - BASELINE_MEAN) /
                     (BASELINE_SD / math.sqrt(trials)))

        # 固定种子 bootstrap：平均命中数 95% CI
        rng = random.Random(BOOTSTRAP_SEED)
        boot_means = []
        for _ in range(bootstrap_rounds):
            sample = [hits[rng.randrange(trials)] for _ in range(trials)]
            boot_means.append(sum(sample) / trials)
        boot_means.sort()
        ci_mean = [round(_percentile(boot_means, 0.025), 3),
                   round(_percentile(boot_means, 0.975), 3)]

        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "train_window": tw,
            "trials": trials,
            "hit_distribution": {str(k): dist[k] for k in range(5)},
            "hit_distribution_pct": {str(k): round(dist_pct[k], 2) for k in range(5)},
            "mean_hits": round(mean_h, 4),
            "mean_hits_bootstrap95": ci_mean,
            "hit_at_least_1_pct": round(ge1 * 100, 2),
            "hit_at_least_2_pct": round(ge2 * 100, 2),
            "z_vs_baseline": round(z_vs_base, 3),
            "baseline": {
                "distribution_pct": {str(k): round(HYPER_BASELINE[k] * 100, 3)
                                     for k in range(5)},
                "mean_hits": round(BASELINE_MEAN, 3),
                "sd_hits": round(BASELINE_SD, 3),
                "at_least_1_pct": round(
                    (1 - HYPER_BASELINE[0]) * 100, 3),
                "at_least_2_pct": round(
                    (HYPER_BASELINE[2] + HYPER_BASELINE[3]
                     + HYPER_BASELINE[4]) * 100, 3),
            },
            "verdict": _verdict(mean_h, ci_mean, z_vs_base, trials),
            "details": details,
            "perf_ms": round(elapsed, 2),
            "note": ("回测样本仅 %d 期，结论不确定性较大；|z|>1.96 才可视为显著优于/弱于随机基准。"
                     % trials),
        }


def _verdict(mean_h: float, ci: list, z: float, trials: int) -> str:
    if abs(z) <= 1.96:
        return ("平均命中 %.2f，与随机基准(1.00)无统计学显著差异（z=%.2f，n=%d）；"
                "现有样本不足以证明模型具备超越随机的预测力。" % (mean_h, z, trials))
    if z > 1.96:
        return "平均命中 %.2f，显著高于随机基准（z=%.2f），模型在该窗口表现出实证有效性。" % (mean_h, z)
    return "平均命中 %.2f，显著低于随机基准（z=%.2f），模型在该窗口有害，应复查因子口径。" % (mean_h, z)
