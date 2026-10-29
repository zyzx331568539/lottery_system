# -*- coding: utf-8 -*-
"""
weight_opt.py —— 评分权重的 walk-forward 滚动优化

目标
====
predictor 默认权重 (0.40, 0.30, 0.25, 0.05) 是拍脑袋的，从未被数据验证。
本模块用嵌套 walk-forward 找"在历史上表现最优的权重"，并对比默认权重。

方法（严格无未来信息泄漏）
==========================
外层：对每期 i（从 min_outer 到 n-1）
  内层：在 records[:i] 上，对每个候选权重组合做 inner_window 滑窗回测，
        取 top1 平均命中最高者作为当期最优权重。
  预测：用该最优权重在 records[:i] 上生成推荐，核对 records[i] 的命中数。

这样每一期的权重只依赖该期之前的数据，杜绝未来信息。
"""
from __future__ import annotations

from math import comb, sqrt

from predictor import W_FOUR, W_NUM, W_PAIR, W_SPREAD, recommend, validate_records

DEFAULT_WEIGHTS = (W_NUM, W_PAIR, W_FOUR, W_SPREAD)
SPREAD_FIXED = W_SPREAD  # 分区离散权重固定，只优化前三维


def weight_grid() -> list[tuple[float, float, float, float]]:
    """候选权重网格：W_NUM × W_PAIR，W_FOUR 取剩余，W_SPREAD 固定。

    网格刻意包含默认权重，保证"不优化"也是一个候选。
    """
    grid = []
    for wn in (0.20, 0.35, 0.50, 0.65):
        for wp in (0.20, 0.30, 0.40, 0.50):
            wf = 1.0 - SPREAD_FIXED - wn - wp
            if 0.0 <= wf <= 0.60:
                grid.append((round(wn, 2), round(wp, 2),
                             round(wf, 2), SPREAD_FIXED))
    if DEFAULT_WEIGHTS not in grid:
        grid.append(DEFAULT_WEIGHTS)
    return grid


def _inner_mean_hits(records, weights, inner_window: int) -> float:
    """在 records 上用 inner_window 滑窗回测，返回 top1 平均命中数。"""
    hits = []
    for i in range(inner_window, len(records)):
        train = records[i - inner_window:i]
        r = recommend(train, top_k=1, with_confidence=False, weights=weights)
        picked = set(r["recommendations"][0]["combo"])
        actual = set(records[i]["numbers"])
        hits.append(len(picked & actual))
    return sum(hits) / len(hits) if hits else 0.0


def optimize_weights_walkforward(records, min_outer: int = 25,
                                 inner_window: int = 15) -> dict:
    """嵌套 walk-forward 权重优化 + 对比默认权重。

    返回：
      {
        "trials": 外层验证期数,
        "optimized": {mean_hits, at_least_1_pct, hit_distribution},
        "default":   {同结构，用默认权重},
        "weight_usage": 各权重组合被选中的次数,
        "details": [{period, weights, hits}, ...]
      }
    """
    validate_records(records, min_count=min_outer + 1)
    grid = weight_grid()
    n = len(records)

    opt_hits, def_hits = [], []
    weight_usage: dict[tuple, int] = {}
    details = []

    for i in range(min_outer, n):
        train = records[:i]
        # 内层：在 train 上找最优权重
        best_w, best_score = DEFAULT_WEIGHTS, -1.0
        for w in grid:
            s = _inner_mean_hits(train, w, inner_window)
            if s > best_score:
                best_score, best_w = s, w
        weight_usage[best_w] = weight_usage.get(best_w, 0) + 1

        # 外层验证：最优权重预测 period i
        r_opt = recommend(records[:i], top_k=1,
                          with_confidence=False, weights=best_w)
        picked_opt = set(r_opt["recommendations"][0]["combo"])

        # 默认权重预测同期（对照）
        r_def = recommend(records[:i], top_k=1,
                          with_confidence=False, weights=DEFAULT_WEIGHTS)
        picked_def = set(r_def["recommendations"][0]["combo"])

        actual = set(records[i]["numbers"])
        h_opt = len(picked_opt & actual)
        h_def = len(picked_def & actual)
        opt_hits.append(h_opt)
        def_hits.append(h_def)
        details.append({
            "period": records[i]["period"],
            "weights": list(best_w),
            "hits_optimized": h_opt,
            "hits_default": h_def,
        })

    def _summary(hits):
        trials = len(hits)
        dist = {k: hits.count(k) for k in range(5)}
        mean = sum(hits) / trials if trials else 0.0
        ge1 = sum(1 for h in hits if h >= 1) / trials * 100 if trials else 0.0
        return {"trials": trials, "mean_hits": round(mean, 4),
                "at_least_1_pct": round(ge1, 2),
                "hit_distribution": {str(k): dist[k] for k in range(5)}}

    baseline_mean = 4 * 20 / 80
    return {
        "optimized": _summary(opt_hits),
        "default": _summary(def_hits),
        "baseline_mean_hits": baseline_mean,
        "weight_usage": {str(list(k)): v for k, v in weight_usage.items()},
        "details": details,
        "note": ("嵌套 walk-forward：每期用历史数据选最优权重再预测当期。"
                 "若 optimized.mean_hits 不显著高于 default，说明权重优化无收益。"),
    }


def best_static_weights(records, inner_window: int = 15) -> dict:
    """在全集上做单次 walk-forward，找出"单一最优固定权重"（用于对比参考）。

    注意：这不是真正的 walk-forward（权重用了全数据），仅供"事后最优"参考，
    不能用于实盘预测。真实预测请用 optimize_weights_walkforward。
    """
    validate_records(records, min_count=inner_window + 1)
    grid = weight_grid()
    scores = []
    for w in grid:
        s = _inner_mean_hits(records, w, inner_window)
        scores.append((s, w))
    scores.sort(reverse=True)
    return {
        "best_weights": list(scores[0][1]),
        "best_mean_hits": round(scores[0][0], 4),
        "default_mean_hits": round(
            _inner_mean_hits(records, DEFAULT_WEIGHTS, inner_window), 4),
        "all": [{"weights": list(w), "mean_hits": round(s, 4)} for s, w in scores],
        "note": "事后最优（用了全数据），仅供对比参考，不可直接用于预测。",
    }
