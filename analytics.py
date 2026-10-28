# -*- coding: utf-8 -*-
"""
analytics.py —— 科学数字筛选核心（80 → 8 → 4）

严格纪律
========
* **无任何随机选择**：全部为确定性统计计算，相同输入永远得到相同输出。
* **口径分离**：真实概率（超几何/二项理论值，对所有号码相同且固定）与
  模型相对指数（多因子评分的 softmax）严格区分，字段名与文档一致。
* **时间截面唯一**：所有特征只基于传入 records，回测传切片即天然无未来信息泄漏。

管道
====
阶段1  特征统计（80码逐个）
  - 全局出现频次 z：          H0: p=0.25，f~Binomial(N,0.25)
  - 近 W 期频次 z：           热号识别（W=10，单侧 z≥1.28）
  - 遗漏值 gap 与尾概率：     冷号识别（gap≥8，0.75^8≈0.10）
  - 八区（10码/区）分布 z：   号段分析（期望 2.5 个/期，卡方 df=7）
阶段2  多因子单码评分
  S = .30·全局z + .35·近期z + .15·遗漏回补分 + .20·号段回补分
  → softmax(S/τ) 得模型相对指数；经验频率给 Wilson 95% 置信区间
阶段3  取评分前 8 名，枚举 C(8,4)=70 组合
  组合分 = .40·号码z + .30·两两共现z + .25·四码共现z + .05·分区离散（与 predictor 同口径）
  → 最优 4 码
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, asdict

from audit_log import AnalysisLogger, PerfTimer, get_logger
from predictor import Features, exact_probability, validate_records

N_NUMBERS = 80
ZONES = 8
ZONE_SIZE = 10


# ============================================================ 参数（可配/可复现）
@dataclass(frozen=True)
class AnalysisParams:
    recent_window: int = 10            # 热号观察窗口（期）
    hot_z: float = 1.28                # 热号单侧阈值（α≈0.10）
    cold_z: float = -1.28              # 近期维度冷号阈值
    gap_cold: int = 8                  # 遗漏冷号阈值（0.75^8=0.1001）
    gap_active: int = 1                # 遗漏≤1 视为活跃
    w_full: float = 0.30               # 单码评分：全局频次z
    w_recent: float = 0.35             # 单码评分：近期频次z
    w_gap: float = 0.15                # 单码评分：遗漏回补分
    w_segment: float = 0.20            # 单码评分：号段回补分
    shortlist: int = 8                 # 进入组合枚举的人数
    softmax_tau: float = 1.0           # softmax 温度

    def validate(self) -> None:
        if self.recent_window < 3:
            raise ValueError("recent_window 至少为3")
        if abs((self.w_full + self.w_recent + self.w_gap + self.w_segment) - 1.0) > 1e-9:
            raise ValueError("单码评分权重之和必须为1")
        if self.shortlist < 4 or self.shortlist > 20:
            raise ValueError("shortlist 取值4-20")
        if self.softmax_tau <= 0:
            raise ValueError("softmax_tau 必须为正")


# ============================================================ 统计工具函数
def norm_ppf(p: float) -> float:
    """
    标准正态分布分位数（Peter Acklam 有理近似，相对误差 <1.15e-9）。
    """
    if not 0.0 < p < 1.0:
        if p == 0.0 or p == 1.0:
            raise ValueError("norm_ppf 不接受 0/1")
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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例 Wilson score 95% 置信区间（比正态近似在小样本/极端比例下更可靠）"""
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def softmax(scores: list[float], tau: float = 1.0) -> list[float]:
    """数值稳定 softmax"""
    exps = [math.exp((s - max(scores)) / tau) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# ============================================================ 冷热号码识别
def compute_gaps(records) -> list[int]:
    """
    每号码 gap = 截至最后一期，连续未出现的完整期数。
      0 = 最新一期出现；P(G≥g)=0.75^g（H0）。

    优化：旧实现 80 码 × n 期 × 20 码列表扫描 O(80·20·n²) 次比较；
    改为从末期反向单趟扫描，每个号码只记录最近一次出现位置，O(20·n)。结果完全一致。
    """
    total = len(records)
    gaps = [0] + [total] * N_NUMBERS   # 槽位0不使用（保持0，兼容旧实现）；从未出现 → gap=总期数
    for i, r in enumerate(reversed(records)):
        for num in r["numbers"]:
            if gaps[num] == total:
                gaps[num] = i
    return gaps


def hot_cold_table(records, params: AnalysisParams, fx: Features) -> list[dict]:
    """
    80码冷热明细（两维口径独立输出）：
      recent_label：近W期维度 hot/warm/cold（二项 z 检验）
      gap_label：   遗漏维度 cold_gap/active/gap_normal
    """
    n = len(records)
    w = min(params.recent_window, n)
    recent_records = records[-w:]
    gaps = compute_gaps(records)
    sd_r = math.sqrt(0.1875 * w)

    # 近 W 期计数单趟聚合（旧实现对80码分别扫描 w×20 次列表）
    recent_counter = Counter()
    for r in recent_records:
        recent_counter.update(r["numbers"])

    rows = []
    for num in range(1, N_NUMBERS + 1):
        rc = recent_counter[num]
        r_z = (rc - 0.25 * w) / (sd_r or 1e-9)
        gap = gaps[num]
        gap_tail = 0.75 ** gap                       # H0 下遗漏≥g 的尾概率
        if r_z >= params.hot_z:
            recent_label = "hot"
        elif r_z <= params.cold_z:
            recent_label = "cold_recent"
        else:
            recent_label = "warm"
        if gap >= params.gap_cold:
            gap_label = "cold_gap"
        elif gap <= params.gap_active:
            gap_label = "active"
        else:
            gap_label = "gap_normal"
        rows.append({
            "number": num,
            "freq": fx.freq[num],
            "freq_pct": round(fx.freq[num] / n * 100, 2),
            "recent_count": rc,
            "recent_window": w,
            "recent_z": round(r_z, 3),
            "recent_label": recent_label,
            "gap": gap,
            "gap_tail_pct": round(gap_tail * 100, 2),
            "gap_label": gap_label,
        })
    return rows


# ============================================================ 号段（八区）分布
def segment_analysis(records, fx: Features) -> dict:
    """
    八区分布：每区10码，每期期望 20·10/80 = 2.5 个。
    输出每区：总出现、期均、占比、偏离 z；整体卡方拟合优度（df=7）。
    """
    n = len(records)
    expected_total = 2.5 * n
    sd_zone = math.sqrt(n * 20 * (1 / 8) * (7 / 8))
    zones = []
    chi2 = 0.0
    for z in range(ZONES):
        lo, hi = z * ZONE_SIZE + 1, (z + 1) * ZONE_SIZE
        observed = sum(fx.freq[k] for k in range(lo, hi + 1))
        zdev = (observed - expected_total) / (sd_zone or 1e-9)
        chi2 += (observed - expected_total) ** 2 / (expected_total or 1e-9)
        zones.append({
            "zone": z + 1,
            "range": [lo, hi],
            "observed": observed,
            "per_draw": round(observed / n, 3),
            "expected_per_draw": 2.5,
            "share_pct": round(observed / (20 * n) * 100, 2),
            "expected_share_pct": 12.5,
            "z": round(zdev, 3),
        })
    return {
        "zones": zones,
        "chi_square": round(chi2, 4),
        "df": ZONES - 1,
        "note": "卡方检验8区占比是否一致；|z|>1.28 表示该区相对偏热/偏冷。",
    }


# ============================================================ 单码多因子评分
def _gap_score(gap: int) -> float:
    tail = 0.75 ** gap
    p = min(1 - 1e-6, max(1e-6, 1 - tail))   # Φ⁻¹ 的输入
    return max(-3.0, min(3.0, norm_ppf(p)))


def build_score_table(records, fx: Features, hot_cold: list[dict],
                      segments: dict, params: AnalysisParams) -> list[dict]:
    """生成80码完整评分表（含模型指数、Wilson区间）"""
    n = len(records)
    zone_dev = {z["zone"]: z["z"] for z in segments["zones"]}
    table = []
    for row in hot_cold:
        num = row["number"]
        full_z = (fx.freq[num] - 0.25 * n) / (fx.sd_f or 1e-9)
        gap_s = _gap_score(row["gap"])
        seg_s = -zone_dev[(num - 1) // ZONE_SIZE + 1]
        total = (params.w_full * full_z + params.w_recent * row["recent_z"]
                 + params.w_gap * gap_s + params.w_segment * seg_s)
        ci_lo, ci_hi = wilson_ci(fx.freq[num], n)
        table.append({
            "number": num,
            "score": round(total, 5),
            "factors": {
                "full_freq_z": round(full_z, 3),
                "recent_z": row["recent_z"],
                "gap_score": round(gap_s, 3),
                "segment_score": round(seg_s, 3),
            },
            "weights": {"full": params.w_full, "recent": params.w_recent,
                        "gap": params.w_gap, "segment": params.w_segment},
            "empirical_freq_pct": row["freq_pct"],
            "wilson95_ci": [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
            "recent_label": row["recent_label"],
            "gap_label": row["gap_label"],
            "gap": row["gap"],
        })
    probs = softmax([t["score"] for t in table], params.softmax_tau)
    for t, p in zip(table, probs):
        t["model_index_pct"] = round(p * 100, 4)
    table.sort(key=lambda r: (-r["score"], r["number"]))
    for rank, t in enumerate(table, 1):
        t["rank"] = rank
    return table


# ============================================================ 统一筛选管道
def run_analysis(records, params: AnalysisParams | None = None,
                 persist: bool = True, fx: Features | None = None) -> dict:
    """
    完整分析管道。回测也调用本函数（persist=False），保证线上/回测同一路径。
    返回完整报告字典；persist=True 时归档全过程并写系统日志。

    fx：可选的预构建 Features（必须由同一份 records 构建），供调用方在多阶段
    管道中复用，避免重复计数；不传则与旧版本一致地现场构建。
    """
    params = params or AnalysisParams()
    params.validate()
    logger = get_logger()
    if not records:
        raise ValueError("数据池为空，无法分析；请先录入或抓取历史开奖数据")
    if len(records) < 5:
        raise ValueError("有效历史仅%d期，至少需要5期才能形成统计" % len(records))
    validate_records(records, min_count=5)

    archiver = AnalysisLogger("daily_pick", logger) if persist else None
    t0 = time.perf_counter()
    with PerfTimer("analytics", logger=logger,
                   extra={"periods": len(records)}):
        if fx is None:
            fx = Features(records)
        hot_cold = hot_cold_table(records, params, fx)
        segments = segment_analysis(records, fx)
        score_table = build_score_table(records, fx, hot_cold, segments, params)

        # 阶段3：前 M 名内枚举组合（组合分口径与 predictor 完全一致）
        shortlist_nums = [t["number"] for t in score_table[:params.shortlist]]
        ranked = fx.ranked_combos(shortlist_nums)
        best = ranked[0]
        final_numbers = sorted(best["combo"])

        ranking_basis = []
        score_by_num = {t["number"]: t for t in score_table}
        for num in final_numbers:
            t = score_by_num[num]
            ranking_basis.append({
                "number": num,
                "rank_of_80": t["rank"],
                "total_score": t["score"],
                "factors": t["factors"],
                "model_index_pct": t["model_index_pct"],
                "empirical_freq_pct": t["empirical_freq_pct"],
                "wilson95_ci_pct": t["wilson95_ci"],
                "recent_label": t["recent_label"],
                "gap_label": t["gap_label"],
            })
        ranking_basis.sort(key=lambda x: x["rank_of_80"])

        report = {
            "numbers": final_numbers,
            "ranking_basis": ranking_basis,
            "combo_score": round(best["score"], 5),
            "combo_score_parts": best["parts"],
            "shortlist": shortlist_nums,
            "theory": exact_probability(),
            "params": asdict(params),
            "hot_cold": hot_cold,
            "segments": segments,
            "score_table": score_table,
            "combo_ranked": [{"combo": sorted(r["combo"]),
                              "score": round(r["score"], 4),
                              "parts": r["parts"]} for r in ranked[:10]],
            "disclaimers": [
                "任意4码组合的真实下期全中概率恒为 %.4f%%（约1/%d），历史数据不能改变它。"
                % (exact_probability()["percent"], exact_probability()["odds_1_in"]),
                "model_index_pct 是80码多因子评分 softmax 后的相对指数（合计100%%），"
                "表示模型在'若只选1码'时的相对倾向，不是该号码下期真实出现概率（真实=25%%）。",
                "遗漏/号段回补项基于'均值回归'经验假设；在开奖独立的原假设下它们无预测力，"
                "其实际贡献应由回测检验。",
            ],
        }

    elapsed = (time.perf_counter() - t0) * 1000.0
    if archiver is not None:
        archiver.set_snapshot(AnalysisLogger.snapshot_from_records(records))
        archiver.set_params(asdict(params))
        archiver.set_intermediate({
            "hot_cold_summary": {
                "hot": [r["number"] for r in hot_cold if r["recent_label"] == "hot"],
                "cold_recent": [r["number"] for r in hot_cold
                                if r["recent_label"] == "cold_recent"],
                "cold_gap": [r["number"] for r in hot_cold
                             if r["gap_label"] == "cold_gap"],
            },
            "segments_chi_square": segments["chi_square"],
            "shortlist": shortlist_nums,
            "combo_ranked_top10": report["combo_ranked"],
        })
        archiver.finish({"numbers": final_numbers,
                         "combo_score": report["combo_score"]}, elapsed)
    report["perf_ms"] = round(elapsed, 2)
    return report
