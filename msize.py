# -*- coding: utf-8 -*-
"""
msize.py —— 多规模扩展（1 / 2 / 3 码），与 4 码同口径、同纪律

定位
====
* **不改动现有 4 码任何路径**：size=4 时全部委派 predictor / analytics /
  backtest 的原函数，仅在外层附加 ``size`` / ``theory_m`` 字段，返回值原样保留。
* 1/2/3 码复用同一套基础设施：
    - 单码/两两特征（z_num / z_pair / 12码候选集）来自 predictor.Features
    - 三码共现维度由 FeaturesM 子类扩展（triple counter + z_triple，泊松近似同四码口径）
    - 科学筛选管道复用 analytics 的阶段1/2（冷热/号段/多因子评分/前8 shortlist）
    - 理论概率复用 picksize 的超几何精确模型（唯一口径来源）
* 全部确定性计算：固定权重、固定随机种子，相同输入永远得到相同输出。

组合评分口径（固定权重，结论可复现）
==================================
  m=1：S = 号码z
  m=2：S = .60·平均号码z + .40·两两共现z
  m=3：S = .50·平均号码z + .30·两两共现z + .15·三码共现z + .05·分区离散
  m=4：沿用 predictor（.40/.30/.25/.05），本模块不重复实现
"""
from __future__ import annotations

import random
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from math import comb, sqrt

import numpy as np

from analytics import (AnalysisParams, build_score_table, hot_cold_table,
                       run_analysis, segment_analysis)
from audit_log import AnalysisLogger, PerfTimer, get_logger
from backtest import BOOTSTRAP_SEED as BT_SEED
from backtest import run_backtest
from data_store import ValidationError
from picksize import all_hit_probability, compare as _theory_compare
from picksize import hit_distribution
from predictor import (BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED, Features,
                       _RecordIndex, _SampledFeatures,
                       _bootstrap_choice_matrix, _mask_weighted_count,
                       analyze_combo, get_observed_scores, recommend,
                       slide_window_features, validate_records,
                       validate_user_combo)

SIZES = (1, 2, 3, 4)

# 各规模组合评分固定权重（m=4 口径在 predictor 中，此处不重复）
W_M = {
    1: {"num": 1.00},
    2: {"num": 0.60, "pair": 0.40},
    3: {"num": 0.50, "pair": 0.30, "triple": 0.15, "spread": 0.05},
}

_SCORE_DESC = {
    1: "号码z",
    2: ".60号码z + .40两两共现z",
    3: ".50号码z + .30两两共现z + .15三码共现z + .05分区离散",
    4: ".40号码z + .30两两共现z + .25四码共现z + .05分区离散",
}


def _check_size(size) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size not in SIZES:
        raise ValidationError("size 必须是 1/2/3/4 的整数，当前：%r" % (size,))
    return size


# ============================================================ 理论概率（精确）
def theory_for_size(m: int) -> dict:
    """
    选 m 个号码的理论指标（超几何精确值，与选号策略无关、对所有组合相同）：
      全中 P = C(20,m)/C(80,m)；至少中1；期望命中 m/4；完整命中分布。
    """
    if isinstance(m, bool) or not isinstance(m, int) or m not in SIZES:
        raise ValueError("m 取值 1..4")
    dist = hit_distribution(m)
    p_all = all_hit_probability(m)
    mean = m * 20 / 80
    var = m * 0.25 * 0.75 * (80 - m) / (80 - 1)   # 超几何方差（与 backtest 基准同式）
    return {
        "size": m,
        "all_hit": p_all,
        "all_hit_pct": round(p_all * 100, 6),
        "odds_1_in": round(1 / p_all, 2),
        "at_least_1_pct": round((1 - dist[0]) * 100, 4),
        "mean_hits": round(mean, 4),
        "sd_hits": round(sqrt(var), 4),
        "distribution_pct": {str(k): round(dist[k] * 100, 4)
                             for k in sorted(dist)},
    }


def theory_table() -> dict:
    """m=1..4 理论对比表（直接复用 picksize.compare，口径唯一）。"""
    out = _theory_compare()
    out["sizes"] = list(SIZES)
    return out


# ============================================================ 特征扩展（仅加三码维度）
class FeaturesM(Features):
    """
    在 predictor.Features 之上仅增加三码共现维度（m=3 评分用）。
    四码及以下的全部特征/打分逻辑原样继承，不做任何修改。
    """
    P3 = all_hit_probability(3)          # 指定3码全中 = C(20,3)/C(80,3)

    def __init__(self, records):
        super().__init__(records)
        self.triple = Counter()
        for r in records:
            self.triple.update(combinations(r["numbers"], 3))
        self._init_triple_stats()

    def _init_triple_stats(self):
        """三码共现的期望/SD（独立成方法，供滑窗增量后以同一公式重算）。"""
        # 与四码共现相同的泊松近似口径
        self.m_3, self.sd_3 = self.P3 * self.n, sqrt(self.P3 * self.n)

    def z_triple(self, c):
        if self.n == 0:
            return 0.0
        return (self.triple[tuple(sorted(c))] - self.m_3) / (self.sd_3 or 1e-9)


class _SampledFeaturesM(FeaturesM, _SampledFeatures):
    """m=3 自助重采样特征：位掩码三码查询，计数与 FeaturesM(sub) 严格一致。

    自助循环内三码共现走向量化 occ 矩阵，索引可不建 tmask；
    此时标量 z_triple 不可用（调用会给出明确错误）。
    """

    def __init__(self, records, chosen, index):
        _SampledFeatures.__init__(self, records, chosen, index)
        self._init_triple_stats()

    def z_triple(self, c):
        if self.n == 0:
            return 0.0
        if self._index.tmask is None:
            raise RuntimeError(
                "该重采样索引未构建三码位掩码（with_triples=False）；"
                "三码评分请走向量化接口 vectorized_score_parts_m")
        c = tuple(sorted(c))
        cnt = _mask_weighted_count(self._index.tmask.get(c, 0), self._mult)
        return (cnt - self.m_3) / (self.sd_3 or 1e-9)


def _features_cls(size: int):
    """m=3 需要三码共现维度；m=1/2 用基类即可（避免无谓的三码计数开销）。"""
    return FeaturesM if size == 3 else Features


# ============================================================ 组合评分（按规模）
def combo_score_m(fx, c):
    """
    广义组合评分。返回 (score, parts)。
    m=4 直接委派 fx.combo_score（与 predictor 完全一致）。
    """
    c = tuple(sorted(c))
    m = len(c)
    if m == 4:
        return fx.combo_score(c)
    if m not in W_M:
        raise ValueError("组合评分仅支持 1-4 码，当前 %d" % m)
    v_num = sum(fx.z_num(n) for n in c) / m
    parts = {"number_z": round(v_num, 3)}
    if m == 1:
        return v_num, parts
    v_pair = sum(fx.z_pair(a, b) for a, b in combinations(c, 2)) / comb(m, 2)
    parts["pair_z"] = round(v_pair, 3)
    if m == 2:
        w = W_M[2]
        return w["num"] * v_num + w["pair"] * v_pair, parts
    # m == 3
    if not isinstance(fx, FeaturesM):
        raise TypeError("三码评分需要 FeaturesM（含三码共现维度），请用 _features_cls(3)")
    v_tri = fx.z_triple(c)
    v_spread = (len({(n - 1) // 10 for n in c}) - 1) / (m - 1)
    parts["triple_z"] = round(v_tri, 3)
    parts["spread"] = round(v_spread, 3)
    w = W_M[3]
    return (w["num"] * v_num + w["pair"] * v_pair
            + w["triple"] * v_tri + w["spread"] * v_spread), parts


def ranked_combos_m(fx, pool, size: int) -> list:
    """在候选集内枚举 C(|pool|, size) 组并按评分降序（平分按组合字典序，确定性）。"""
    rows = []
    for c in combinations(sorted(pool), size):
        score, parts = combo_score_m(fx, c)
        rows.append({"combo": c, "score": score, "parts": parts})
    rows.sort(key=lambda r: (-r["score"], r["combo"]))
    return rows


# ============================================================ 向量化评分（自助循环专用）
def _vectorized_score_column(fx, pool, size: int):
    """组装 m=1/2/3 向量化总分：权重与左结合顺序与 combo_score_m 完全一致。"""
    _pool, combo_num, v_num, v_pair, v_tri, v_spread = \
        fx.vectorized_score_parts_m(pool, size)
    if size == 1:
        score = v_num
    elif size == 2:
        w = W_M[2]
        score = w["num"] * v_num + w["pair"] * v_pair
    else:
        w = W_M[3]
        score = (w["num"] * v_num + w["pair"] * v_pair
                 + w["triple"] * v_tri + w["spread"] * v_spread)
    return combo_num, score, v_num, v_pair, v_tri, v_spread


def _vectorized_order(combo_num, score, size: int) -> np.ndarray:
    """复刻 sorted(key=(-score, combo))：-score 主键，组合各码字典序次序键。"""
    keys = [combo_num[:, j] for j in range(size - 1, -1, -1)]
    keys.append(-score)
    return np.lexsort(tuple(keys))


def ranked_combos_vectorized_m(fx, pool, size: int) -> list:
    """ranked_combos_m 的向量化版本（仅用于 _SampledFeatures* 重采样特征）：
    行（combo/score/parts，含排序）与标量版逐字段、浮点位级相等。"""
    combo_num, score, v_num, v_pair, v_tri, v_spread = \
        _vectorized_score_column(fx, pool, size)
    order = _vectorized_order(combo_num, score, size)
    rows = []
    for i in order:
        t = tuple(int(x) for x in combo_num[i])
        parts = {"number_z": round(float(v_num[i]), 3)}
        if size >= 2:
            parts["pair_z"] = round(float(v_pair[i]), 3)
        if size == 3:
            parts["triple_z"] = round(float(v_tri[i]), 3)
            parts["spread"] = round(float(v_spread[i]), 3)
        rows.append({"combo": t, "score": float(score[i]), "parts": parts})
    return rows


def top5_combos_vectorized_m(fx, pool, size: int) -> set:
    """自助循环专用：只取前5组合集合，省去构造全部 dict（m=3 时为 220 组）。"""
    combo_num, score, *_ = _vectorized_score_column(fx, pool, size)
    order = _vectorized_order(combo_num, score, size)
    return {tuple(int(x) for x in combo_num[i]) for i in order[:5]}


# ============================================================ 概率分析（1-4码）
def analyze_combo_m(numbers, records) -> dict:
    """
    对 1-4 个用户号码做概率分析（规模由输入个数推断）。
    m=4 委派 predictor.analyze_combo，并补齐 co_kind/co_z 使字段口径统一。
    """
    combo = validate_user_combo(numbers, min_size=1, max_size=4)
    validate_records(records)
    m = len(combo)
    th = theory_for_size(m)

    if m == 4:
        out = analyze_combo(combo, records)
        out["size"] = 4
        out["theory_m"] = th
        out["empirical"]["co_kind"] = "four"
        out["empirical"]["co_z"] = out["empirical"]["four_z"]
        return out

    fx = _features_cls(m)(records)
    score, parts = combo_score_m(fx, combo)

    # 经验共现（按规模取对应维度）与百分位；评分有序表按数据池内容签名跨请求缓存
    def _build_seen():
        if m == 1:
            return sorted(combo_score_m(fx, (n,))[0] for n in range(1, 81))
        if m == 2:
            return sorted(combo_score_m(fx, c)[0] for c in fx.pair.keys())
        return sorted(combo_score_m(fx, c)[0] for c in fx.triple.keys())

    if m == 1:
        co_kind = "number"
        co_count = fx.freq[combo[0]]
        co_expected = 0.25 * fx.n
        co_z = fx.z_num(combo[0])
    elif m == 2:
        co_kind = "pair"
        co_count = fx.pair[tuple(combo)]
        co_expected = fx.m_p
        co_z = fx.z_pair(combo[0], combo[1])
    else:  # m == 3
        co_kind = "triple"
        co_count = fx.triple[tuple(combo)]
        co_expected = fx.m_3
        co_z = fx.z_triple(combo)
    seen = get_observed_scores(records, m, _build_seen)

    percentile = (bisect_right(seen, score) / len(seen) * 100) if seen else None
    hit_periods = [r["period"] for r in records if set(combo) <= set(r["numbers"])]
    number_detail = [{"number": n, "count": fx.freq[n],
                      "expected": round(fx.m_f, 2), "z": round(fx.z_num(n), 2)}
                     for n in combo]
    pair_detail = [{"pair": [a, b],
                    "count": fx.pair[(min(a, b), max(a, b))],
                    "expected": round(fx.m_p, 2), "z": round(fx.z_pair(a, b), 2)}
                   for a, b in combinations(combo, 2)] if m >= 2 else []
    return {
        "size": m,
        "combo": combo,
        "theory_m": th,
        "empirical": {
            "co_kind": co_kind,
            "co_occurrence": co_count,
            "co_expected": round(co_expected, 2),
            "co_z": round(co_z, 2),
            "score": round(score, 4),
            "score_parts": parts,
            "percentile_among_observed": (round(percentile, 1)
                                          if percentile is not None else None),
            "observed_basis": len(seen),
            "hit_periods": hit_periods,
        },
        "numbers": number_detail,
        "pairs": pair_detail,
        "sample_periods": fx.n,
        "disclaimer": ("理论全中概率对任意%d码组合完全相同且固定（%.4f%%，约1/%.0f）；"
                       "经验评分仅反映历史相对强弱，不构成对未来开奖的预测或收益保证。"
                       % (m, th["all_hit_pct"], th["odds_1_in"])),
    }


# ============================================================ 科学筛选管道（1-4码）
def run_analysis_m(records, size: int, params: AnalysisParams | None = None,
                   persist: bool = True, fx=None) -> dict:
    """
    与 analytics.run_analysis 同一管道的多规模版本：80码特征 → 前8 shortlist
    → 枚举 C(8, size) 组合取最优。
    size=4 委派原函数（逐字段不变，仅附加 size/theory_m）。

    fx：可选的预构建特征（必须与 records 同源同规模），供滑动回测复用。
    """
    _check_size(size)
    if size == 4:
        rep = run_analysis(records, params=params, persist=persist, fx=fx)
        rep["size"] = 4
        rep["theory_m"] = theory_for_size(4)
        return rep

    params = params or AnalysisParams()
    params.validate()
    logger = get_logger()
    if not records:
        raise ValueError("数据池为空，无法分析；请先录入或抓取历史开奖数据")
    if len(records) < 5:
        raise ValueError("有效历史仅%d期，至少需要5期才能形成统计" % len(records))
    validate_records(records, min_count=5)

    archiver = AnalysisLogger("daily_pick_m%d" % size, logger) if persist else None
    t0 = time.perf_counter()
    with PerfTimer("analytics", logger=logger,
                   extra={"periods": len(records), "size": size}):
        if fx is None:
            fx = _features_cls(size)(records)
        hot_cold = hot_cold_table(records, params, fx)
        segments = segment_analysis(records, fx)
        score_table = build_score_table(records, fx, hot_cold, segments, params)

        shortlist_nums = [t["number"] for t in score_table[:params.shortlist]]
        ranked = ranked_combos_m(fx, shortlist_nums, size)
        best = ranked[0]
        final_numbers = sorted(best["combo"])
        th = theory_for_size(size)

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
            "size": size,
            "numbers": final_numbers,
            "ranking_basis": ranking_basis,
            "combo_score": round(best["score"], 5),
            "combo_score_parts": best["parts"],
            "shortlist": shortlist_nums,
            "theory_m": th,
            "params": asdict(params),
            "hot_cold": hot_cold,
            "segments": segments,
            "score_table": score_table,
            "combo_ranked": [{"combo": sorted(r["combo"]),
                              "score": round(r["score"], 4),
                              "parts": r["parts"]} for r in ranked[:10]],
            "disclaimers": [
                "任意%d码组合的真实下期全中概率恒为 %.4f%%（约1/%.0f），历史数据不能改变它。"
                % (size, th["all_hit_pct"], th["odds_1_in"]),
                "model_index_pct 是80码多因子评分 softmax 后的相对指数（合计100%%），"
                "表示模型在'若只选1码'时的相对倾向，不是该号码下期真实出现概率（真实=25%%）。",
                "遗漏/号段回补项基于'均值回归'经验假设；在开奖独立的原假设下它们无预测力，"
                "其实际贡献应由回测检验。",
            ],
        }

    elapsed = (time.perf_counter() - t0) * 1000.0
    if archiver is not None:
        archiver.set_snapshot(AnalysisLogger.snapshot_from_records(records))
        archiver.set_params({**asdict(params), "size": size})
        archiver.set_intermediate({
            "size": size,
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
        archiver.finish({"size": size, "numbers": final_numbers,
                         "combo_score": report["combo_score"]}, elapsed)
    report["perf_ms"] = round(elapsed, 2)
    return report


# ============================================================ N+1 推荐（1-4码）
def bootstrap_stability_m(records, targets, size: int,
                          rounds: int = BOOTSTRAP_ROUNDS,
                          seed: int = BOOTSTRAP_SEED) -> dict:
    """固定种子期级自助重采样：目标组合留在候选集且排名前5的比例（与4码同法）。

    优化：抽样索引一次性批量预生成（与逐次 rng.choice 位级同序列，结果可复现）；
    重采样特征用位掩码/occ 矩阵按需查询；候选集与 m=1/2/3 组合评分排序全部
    NumPy 向量化（m=3 的 220 组标量循环已消除），省去每轮全量
    pair/triple/four 计数器重建。三码共现向量化走 occ，索引无需构建 tmask。
    """
    _check_size(size)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValidationError(f"自助轮次非法：{rounds!r}")
    if not records:
        return {t: 0.0 for t in targets}
    rng = random.Random(seed)
    n = len(records)
    choices = _bootstrap_choice_matrix(rng, n, rounds)
    index = _RecordIndex(records)
    sampled_cls = _SampledFeaturesM if size == 3 else _SampledFeatures
    hits = {t: 0 for t in targets}
    target_sets = {t: set(t) for t in targets}
    for r in range(rounds):
        f = sampled_cls(records, choices[r], index)
        # 候选集 80 码 z 排序与 C(12,m) 组评分/排序全部走向量化路径
        pool = f.candidates_vectorized()
        top5 = top5_combos_vectorized_m(f, pool, size)
        pool_set = set(pool)
        for t in targets:
            if target_sets[t] <= pool_set and t in top5:
                hits[t] += 1
    return {t: hits[t] / rounds for t in targets}


def recommend_m(records, size: int, last_n=None, top_k: int = 5,
                auto_strategy: bool = False) -> dict:
    """
    第 N+1 日推荐（多规模）。size=4 委派 predictor.recommend（逐字段不变）。
    auto_strategy: True 时用 tracker 历史命中率最优的 last_n（安全回退到 last_n）。
    结构与4码一致：recommendations / candidate_pool / basis / method / disclaimer。
    """
    _check_size(size)
    if size == 4:
        out = recommend(records, last_n, top_k, auto_strategy=auto_strategy)
        out["size"] = 4
        out["theory_m"] = theory_for_size(4)
        return out
    if not records:
        raise ValidationError("没有任何历史数据，无法生成推荐；请先录入或抓取数据")
    if last_n is not None and (isinstance(last_n, bool)
                               or not isinstance(last_n, int) or last_n <= 0):
        raise ValidationError(f"近N期参数非法：{last_n!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError(f"推荐条数非法：{top_k!r}")
    if auto_strategy:
        from tracker import best_strategy
        picked = best_strategy(size)
        if picked is not None:
            last_n = picked["last_n"]
    use = records if last_n is None else records[-last_n:]
    if len(use) < 5:
        raise ValidationError(f"有效历史仅{len(use)}期，至少需要5期才能形成统计")
    validate_records(use, min_count=5)

    fx = _features_cls(size)(use)
    pool = fx.candidates()
    ranked = ranked_combos_m(fx, pool, size)
    scores = [r["score"] for r in ranked]
    mu = sum(scores) / len(scores)
    sd = sqrt(sum((s - mu) ** 2 for s in scores) / len(scores)) or 1e-9

    targets = [r["combo"] for r in ranked[:top_k]]
    stability = bootstrap_stability_m(use, targets, size)

    sample_factor = min(1.0, len(use) / 50)
    out = []
    for r in ranked[:top_k]:
        t = r["combo"]
        margin = max(0.0, min(1.0, (r["score"] - mu) / (2 * sd)))
        conf = 0.45 * stability[t] + 0.30 * margin + 0.25 * sample_factor
        conf = round(conf, 3)
        label = "High" if conf >= 0.65 else "Moderate" if conf >= 0.45 else "Low"
        out.append({"combo": list(t), "score": round(r["score"], 4),
                    "score_parts": r["parts"], "confidence": conf,
                    "confidence_label": label,
                    "stability": round(stability[t], 2), "margin": round(margin, 2)})
    th = theory_for_size(size)
    return {
        "size": size,
        "recommendations": out,
        "candidate_pool": pool,
        "basis": {"periods_used": len(use),
                  "first_period": use[0]["period"], "last_period": use[-1]["period"],
                  "sample_factor": round(sample_factor, 2)},
        "theory_m": th,
        "method": ("12码候选集(八区保底)内枚举 C(12,%d) 组，按 %s 排序；"
                   "置信度=固定种子自助200轮稳定度/区分裕度/样本充分度。"
                   % (size, _SCORE_DESC[size])),
        "disclaimer": ("置信度是历史数据内部的相对指标，不是中奖概率；任意%d码真实全中概率恒为"
                       "%.4f%%(约1/%.0f)。" % (size, th["all_hit_pct"], th["odds_1_in"])),
    }


# ============================================================ 历史回测（1-4码）
def run_backtest_m(records, size: int, train_window: int | None = None,
                   params: AnalysisParams | None = None,
                   bootstrap_rounds: int = 200) -> dict:
    """
    与 backtest.run_backtest 同一方法的多规模版本：严格时间切片 + 超几何基准
    + 固定种子 bootstrap 95% 区间。size=4 委派原函数（逐字段不变）。
    """
    _check_size(size)
    if size == 4:
        out = run_backtest(records, train_window=train_window, params=params,
                           bootstrap_rounds=bootstrap_rounds)
        out["size"] = 4
        out["theory_m"] = theory_for_size(4)
        return out

    params = params or AnalysisParams()
    logger = get_logger()
    if len(records) < 6:
        raise ValueError("历史数据至少需要6期才能回测（5期训练+1期验证）")
    validate_records(records, min_count=6)
    if (train_window is not None
            and (isinstance(train_window, bool) or not isinstance(train_window, int))):
        raise ValueError("train_window 必须为正整数")
    if (isinstance(bootstrap_rounds, bool) or not isinstance(bootstrap_rounds, int)
            or bootstrap_rounds <= 0):
        raise ValueError("bootstrap_rounds 必须为正整数")
    tw = train_window or (len(records) - 1)
    if tw < 5 or tw >= len(records):
        raise ValueError("train_window 必须 >=5 且 < 总期数(%d)" % len(records))

    th = theory_for_size(size)
    base_dist = hit_distribution(size)                 # {k: 精确概率}
    base_mean = size * 20 / 80
    base_sd = th["sd_hits"]

    t0 = time.perf_counter()
    with PerfTimer("backtest", logger=logger,
                   extra={"trials": len(records) - tw, "size": size}):
        details = []
        hits = []
        # 滑动窗口特征（三码计数器在 FeaturesM 上同样按一期增量更新）
        fx = _features_cls(size)(records[:tw])
        for i in range(tw, len(records)):
            train = records[i - tw:i]
            if i > tw:
                slide_window_features(fx, records[i - tw - 1]["numbers"],
                                      records[i - 1]["numbers"], train)
            result = run_analysis_m(train, size, params=params, persist=False, fx=fx)
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
        dist = {k: hits.count(k) for k in range(size + 1)}
        dist_pct = {k: dist[k] / trials * 100 for k in range(size + 1)}
        mean_h = sum(hits) / trials
        ge1 = sum(1 for h in hits if h >= 1) / trials
        ge2 = sum(1 for h in hits if h >= 2) / trials

        z_vs_base = (mean_h - base_mean) / (base_sd / sqrt(trials))

        rng = random.Random(BT_SEED)
        boot_means = []
        for _ in range(bootstrap_rounds):
            sample = [hits[rng.randrange(trials)] for _ in range(trials)]
            boot_means.append(sum(sample) / trials)
        boot_means.sort()
        ci_mean = [round(_percentile(boot_means, 0.025), 3),
                   round(_percentile(boot_means, 0.975), 3)]

        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "size": size,
            "train_window": tw,
            "trials": trials,
            "hit_distribution": {str(k): dist[k] for k in range(size + 1)},
            "hit_distribution_pct": {str(k): round(dist_pct[k], 2)
                                     for k in range(size + 1)},
            "mean_hits": round(mean_h, 4),
            "mean_hits_bootstrap95": ci_mean,
            "hit_at_least_1_pct": round(ge1 * 100, 2),
            "hit_at_least_2_pct": round(ge2 * 100, 2),
            "z_vs_baseline": round(z_vs_base, 3),
            "baseline": {
                "distribution_pct": {str(k): round(base_dist[k] * 100, 3)
                                     for k in range(size + 1)},
                "mean_hits": round(base_mean, 3),
                "sd_hits": round(base_sd, 3),
                "at_least_1_pct": round((1 - base_dist[0]) * 100, 3),
                "at_least_2_pct": round(
                    sum(base_dist[k] for k in range(2, size + 1)) * 100, 3),
            },
            "theory_m": th,
            "verdict": _verdict_m(mean_h, z_vs_base, trials, size, base_mean),
            "details": details,
            "perf_ms": round(elapsed, 2),
            "note": ("回测样本仅 %d 期，结论不确定性较大；|z|>1.96 才可视为显著优于/弱于随机基准。"
                     % trials),
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(len(sorted_vals) - 1, lo + 1)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _verdict_m(mean_h: float, z: float, trials: int, size: int,
               base_mean: float) -> str:
    if abs(z) <= 1.96:
        return ("平均命中 %.2f，与随机基准(%.2f)无统计学显著差异（z=%.2f，n=%d）；"
                "现有样本不足以证明%d码策略具备超越随机的预测力。"
                % (mean_h, base_mean, z, trials, size))
    if z > 1.96:
        return ("平均命中 %.2f，显著高于随机基准 %.2f（z=%.2f），"
                "%d码策略在该窗口表现出实证有效性。" % (mean_h, base_mean, z, size))
    return ("平均命中 %.2f，显著低于随机基准 %.2f（z=%.2f），"
            "%d码策略在该窗口有害，应复查因子口径。" % (mean_h, base_mean, z, size))
