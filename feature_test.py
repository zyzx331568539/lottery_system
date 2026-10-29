# -*- coding: utf-8 -*-
"""
feature_test.py —— 特征预测力检验（Information Coefficient, IC）

核心问题
========
predictor / analytics 用了一堆"描述性统计量"（频次 z、近期 z、遗漏 gap、号段 z、
共现 z）当作选号依据，但从未检验过：**这些特征对"下一期是否开出"到底有没有预测力？**

方法
====
逐期滚动（严格无未来信息泄漏）：
  对每个 t 从 min_train 到 n-1：
    1. 只用 records[:t] 构建特征（Features / hot_cold / segment）；
    2. 取 80 个号码的特征值向量 X（连续）；
    3. 取这 80 个号码在 records[t] 是否开出的 0/1 向量 Y；
    4. 计算 Spearman 秩相关 IC_t = corr(rank(X), Y)。
  汇总：mean_ic / sd_ic / t_stat = mean/(sd/√T) / 近似 p 值。

判读
====
* |t_stat| > 1.96 且样本量充足 → 该特征在统计上有（微弱）预测力；
* |t_stat| ≤ 1.96 → 无法拒绝"无预测力"原假设，该特征本质是噪声，应降权或剔除。
* 注意：即便显著，IC 绝对值通常也很小（彩票近乎独立），本检验的价值在于
  **区分"有微弱信号"与"纯噪声"**，而非追求高 IC。
"""
from __future__ import annotations

import math
from math import erf, sqrt

import numpy as np

from analytics import AnalysisParams, hot_cold_table, segment_analysis
from predictor import Features, validate_records

N_NUMBERS = 80


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩次（处理并列），与 scipy.stats.rankdata method='average' 一致。"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # 并列修正
    sorted_a = a[order]
    # 找出值相同的连续段
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    denom = sqrt((dx * dx).sum() * (dy * dy).sum())
    if denom == 0:
        return 0.0
    return float((dx * dy).sum() / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关：对 x 取秩（y 为 0/1 二值，秩次等价于原值排序）。"""
    return _pearson(_rankdata(x), y)


def _two_sided_p_from_t(t: float, df: int) -> float:
    """t 分布双侧 p 值的正态近似（df 较大时足够；df<30 时偏保守）。"""
    if df <= 0:
        return 1.0
    # 用正态近似；|t| 大时 erf 足够精确
    return float(math.erfc(abs(t) / sqrt(2.0)))


def feature_predictive_test(records, min_train: int = 10) -> dict:
    """逐期滚动计算各号码级特征对下一期开出结果的 IC。

    返回：
      {
        "n_periods": 用于计算的期数,
        "features": {
            特征名: {
                "mean_ic", "sd_ic", "t_stat", "p_value",
                "positive_ratio": IC>0 的期数占比
            }, ...
        },
        "note": 判读说明
      }
    """
    validate_records(records, min_count=min_train + 1)
    params = AnalysisParams()
    n = len(records)
    # 收集每个特征逐期 IC
    ics: dict[str, list[float]] = {
        "freq_z": [],
        "recent_z": [],
        "gap_score": [],
        "segment_z": [],
        "z_num_composite": [],
    }

    for t in range(min_train, n):
        train = records[:t]
        fx = Features(train)
        hc = hot_cold_table(train, params, fx)
        seg = segment_analysis(train, fx)
        zone_dev = {z["zone"]: z["z"] for z in seg["zones"]}

        # 80 个号码的特征向量
        nums = np.arange(1, N_NUMBERS + 1)
        freq_z = np.array([(fx.freq[num] - fx.m_f) / (fx.sd_f or 1e-9)
                           for num in nums], dtype=np.float64)
        recent_z = np.array([row["recent_z"] for row in hc], dtype=np.float64)
        gap_score = np.array([
            max(-3.0, min(3.0, _norm_ppf(min(1 - 1e-6, max(1e-6, 1 - 0.75 ** row["gap"])))))
            for row in hc
        ], dtype=np.float64)
        segment_z = np.array([-zone_dev[(num - 1) // 10 + 1] for num in nums],
                             dtype=np.float64)
        z_num = np.array([fx.z_num(num) for num in nums], dtype=np.float64)

        actual_set = set(records[t]["numbers"])
        y = np.array([1.0 if num in actual_set else 0.0 for num in nums])

        ics["freq_z"].append(_spearman(freq_z, y))
        ics["recent_z"].append(_spearman(recent_z, y))
        ics["gap_score"].append(_spearman(gap_score, y))
        ics["segment_z"].append(_spearman(segment_z, y))
        ics["z_num_composite"].append(_spearman(z_num, y))

    periods = len(ics["freq_z"])
    result = {"n_periods": periods, "features": {}}
    for name, vals in ics.items():
        arr = np.array(vals, dtype=np.float64)
        mean_ic = float(arr.mean())
        sd_ic = float(arr.std(ddof=1)) if periods > 1 else 0.0
        se = sd_ic / sqrt(periods) if periods > 0 else 1e-9
        t_stat = mean_ic / se if se > 0 else 0.0
        p_val = _two_sided_p_from_t(t_stat, periods - 1)
        pos_ratio = float((arr > 0).mean())
        result["features"][name] = {
            "mean_ic": round(mean_ic, 5),
            "sd_ic": round(sd_ic, 5),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "positive_ratio": round(pos_ratio, 3),
            "verdict": _ic_verdict(t_stat, p_val, periods),
        }
    result["note"] = (
        "IC = 特征值与下一期是否开出的 Spearman 秩相关（逐期滚动，无未来信息）。"
        "|t|>1.96 且 p<0.05 视为有统计显著的预测力；否则无法拒绝'纯噪声'原假设。"
    )
    return result


def _ic_verdict(t_stat: float, p_value: float, periods: int) -> str:
    if periods < 20:
        return f"样本仅 {periods} 期，结论不确定性大"
    if abs(t_stat) <= 1.96:
        return f"无显著预测力（t={t_stat:.2f}, p={p_value:.3f}），可视为噪声"
    if t_stat > 1.96:
        return f"显著正向预测力（t={t_stat:.2f}, p={p_value:.3f}），高值→更可能开出"
    return f"显著负向预测力（t={t_stat:.2f}, p={p_value:.3f}），高值→更不可能开出"


def _norm_ppf(p: float) -> float:
    """标准正态分位数（与 analytics.norm_ppf 同源，此处独立实现避免循环）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 超出 (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
