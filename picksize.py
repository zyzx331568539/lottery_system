# -*- coding: utf-8 -*-
"""
picksize.py —— 选号数量 m = 1, 2, 3, 4 的命中概率对比分析

定位
====
* **不改动、不依赖现有 4 码筛选算法**：本模块是独立的理论概率分析，
  用于回答"保持开奖机制（80 开 20）不变，只把所选号码从 4 个减到 3/2/1 个，
  命中概率如何变化"。
* 模型：命中数 X ~ 超几何分布 Hypergeometric(N=80, K=20, n=m)
      P(X=k) = C(20,k)·C(60, m-k) / C(80, m),  k = 0..m
* 全部为整数组合数精确值（纯标准库），相同输入永远得到相同输出，无随机、无近似。

三种"命中口径"（结论方向不同，必须区分）
========================================
1. 全中（所选 m 码全部开出）：  P_m = C(20,m)/C(80,m)        —— m 越小越大
2. 至少中 1 个：               P(X>=1) = 1 - P(X=0)          —— m 越小越小
3. 期望命中数：                E[X] = m·20/80 = m/4          —— 随 m 线性
"""
from __future__ import annotations

import json
from math import comb

TOTAL = 80          # 候选号码总数
WIN = 20            # 每期开出号码数
MISS = TOTAL - WIN  # 每期未开出号码数 = 60


# ============================================================ 单 m 精确分布
def hit_distribution(m: int) -> dict[int, float]:
    """
    选 m 个号码时，命中数 k=0..m 的精确概率（超几何分布）。
    """
    if not 1 <= m <= WIN:
        raise ValueError("m 取值 1..20")
    denom = comb(TOTAL, m)
    return {k: comb(WIN, k) * comb(MISS, m - k) / denom for k in range(m + 1)}


def all_hit_probability(m: int) -> float:
    """所选 m 个号码全部开出的精确概率 = C(20,m)/C(80,m)。"""
    return comb(WIN, m) / comb(TOTAL, m)


def conditional_factors(m_max: int = 4) -> list[dict]:
    """
    全中概率的条件概率连乘分解（解释变化原因的核心）：
      P_m = Π_{j=1..m} (20-j+1)/(80-j+1)
    第 j 个因子 = "前 j-1 个已命中"条件下，第 j 个号码也命中的条件概率。
    """
    out = []
    cum = 1.0
    for j in range(1, m_max + 1):
        factor = (WIN - j + 1) / (TOTAL - j + 1)
        cum *= factor
        out.append({"step": j, "conditional_prob": round(factor, 6),
                    "cumulative_all_hit": round(cum, 8)})
    return out


# ============================================================ 多 m 对比
def compare(sizes: tuple[int, ...] = (1, 2, 3, 4)) -> dict:
    """生成 m=1..4 的完整对比报告（百分比、赔率、分布、相邻倍率、期望、方差）。"""
    rows = []
    for m in sorted(sizes):
        dist = hit_distribution(m)
        p_all = all_hit_probability(m)
        # 超几何均值/方差：E=mK/N；Var=m(K/N)(1-K/N)·(N-m)/(N-1)
        mean = m * WIN / TOTAL
        var = m * (WIN / TOTAL) * (MISS / TOTAL) * (TOTAL - m) / (TOTAL - 1)
        rows.append({
            "m": m,
            "all_hit_pct": round(p_all * 100, 6),
            "odds_1_in": round(1 / p_all, 2),
            "at_least_1_pct": round((1 - dist[0]) * 100, 4),
            "mean_hits": round(mean, 4),
            "sd_hits": round(var ** 0.5, 4),
            "distribution_pct": {str(k): round(dist[k] * 100, 4)
                                 for k in sorted(dist)},
        })

    # 相邻 m 的全中概率倍率：P_{m-1}/P_m = (80-m+1)/(20-m+1)
    for i in range(1, len(rows)):
        m = rows[i]["m"]
        exact_ratio = (TOTAL - m + 1) / (WIN - m + 1)
        rows[i]["all_hit_ratio_vs_m_minus_1"] = round(exact_ratio, 4)

    return {
        "model": "Hypergeometric(N=80, K=20, n=m): "
                 "P(X=k)=C(20,k)C(60,m-k)/C(80,m)",
        "conditional_factors": conditional_factors(max(sizes)),
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
