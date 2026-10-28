# -*- coding: utf-8 -*-
"""
predictor.py —— 概率分析与 N+1 日推荐核心

方法论（全部可复核，纯标准库、确定性输出）：
1. 精确理论概率（超几何分布）
     每天从80码开20码。任意指定4码全中的概率：
       P4 = C(76,16) / C(80,20) ≈ 0.003063（约1/326）
     任意指定2码：P2 = C(78,18) / C(80,20) ≈ 0.06013
   该概率对所有4码组合完全相同，历史数据不能改变它（开奖相互独立）。

2. 经验特征（仅描述历史相对强弱，不是真实概率）
     - 号码频次 z 值：H0 下每期出现概率0.25 → f~Binomial(N,0.25)
     - 近期加权频次 z 值：指数半衰期10期
     - 两两共现 z 值：期望 N·P2
     - 四码共现 z 值（泊松近似）：期望 N·P4
     - 分区离散度：4码落在不同区段加分

3. 组合评分（权重固定，结论可复现）
     S = .40·平均号码z + .30·平均共现z + .25·四码共现z + .05·分区离散
   在12码候选集（频次/近期评分前16 + 八区保底）内枚举 C(12,4)=495 组排序。

4. 置信度（经验置信度，非中奖概率）
     .45·自助稳定度 + .30·区分裕度 + .25·样本充分度
   自助法：固定随机种子重采样200轮，统计该组合留在候选集且排名前5的比例。
"""
import random
from bisect import bisect_right
from collections import Counter
from itertools import combinations
from math import comb, sqrt

from data_store import ValidationError, validate_numbers

# ---------------------------------------------------------------- 常量
P4 = comb(76, 16) / comb(80, 20)     # 指定4码全中
P2 = comb(78, 18) / comb(80, 20)     # 指定2码全中
HALFLIFE = 10                        # 近期权重半衰期(期)
CANDIDATE_SIZE = 12
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20260920
W_NUM, W_PAIR, W_FOUR, W_SPREAD = 0.40, 0.30, 0.25, 0.05


def exact_probability():
    """理论精确概率（对所有组合相同）"""
    return {"p4": P4, "odds_1_in": 1 / P4, "percent": P4 * 100,
            "p2": P2, "p2_percent": P2 * 100}


# ---------------------------------------------------------------- 特征工程
class Features:
    def __init__(self, records):
        self.records = records
        self.n = len(records)
        self.freq = Counter()
        self.wfreq = Counter()
        self.pair = Counter()
        self.four = Counter()
        self._build()
        # H0 下的均值/标准差
        self.m_f, self.sd_f = 0.25 * self.n, sqrt(0.1875 * self.n)
        weights = [0.5 ** ((self.n - 1 - i) / HALFLIFE)
                   for i in range(self.n)]
        w_sum, w_sq = sum(weights), sum(w * w for w in weights)
        self.m_w, self.sd_w = 0.25 * w_sum, sqrt(0.1875 * w_sq)
        self.m_p, self.sd_p = P2 * self.n, sqrt(P2 * (1 - P2) * self.n)
        self.m_4, self.sd_4 = P4 * self.n, sqrt(P4 * self.n)

    def _build(self):
        for i, r in enumerate(self.records):
            nums = r["numbers"]
            w = 0.5 ** ((self.n - 1 - i) / HALFLIFE)
            self.freq.update(nums)
            for n in nums:
                self.wfreq[n] += w
            self.pair.update(combinations(nums, 2))
            self.four.update(combinations(nums, 4))

    # ---- 标准化打分
    def z_num(self, n):
        if self.n == 0:
            return 0.0
        return 0.65 * (self.freq[n] - self.m_f) / self.sd_f \
             + 0.35 * (self.wfreq[n] - self.m_w) / (self.sd_w or 1e-9)

    def z_pair(self, a, b):
        if self.n == 0:
            return 0.0
        return (self.pair[(min(a, b), max(a, b))] - self.m_p) / (self.sd_p or 1e-9)

    def z_four(self, c):
        if self.n == 0:
            return 0.0
        return (self.four[tuple(sorted(c))] - self.m_4) / (self.sd_4 or 1e-9)

    def candidates(self):
        """12码候选集：评分前16 + 八区保底1个"""
        order = sorted(range(1, 81), key=lambda n: (-self.z_num(n), n))
        base = order[:16]
        pool = set()
        for z in range(8):
            cand = [n for n in base if (n - 1) // 10 == z]
            if cand:
                pool.add(cand[0])
        for n in base:
            if len(pool) >= CANDIDATE_SIZE:
                break
            pool.add(n)
        return sorted(pool)

    def combo_score(self, c):
        """组合总分 + 各分量（可解释）"""
        c = tuple(sorted(c))
        v_num = sum(self.z_num(n) for n in c) / 4
        v_pair = sum(self.z_pair(a, b) for a, b in combinations(c, 2)) / 6
        v_four = self.z_four(c)
        v_spread = (len({(n - 1) // 10 for n in c}) - 1) / 3
        score = W_NUM * v_num + W_PAIR * v_pair + W_FOUR * v_four + W_SPREAD * v_spread
        return score, {"number_z": round(v_num, 3), "pair_z": round(v_pair, 3),
                       "four_z": round(v_four, 3), "spread": round(v_spread, 3)}

    def ranked_combos(self, pool=None):
        pool = pool or self.candidates()
        rows = []
        for c in combinations(pool, 4):
            score, parts = self.combo_score(c)
            rows.append({"combo": c, "score": score, "parts": parts})
        rows.sort(key=lambda r: (-r["score"], r["combo"]))
        return rows


# ---------------------------------------------------------------- 输入校验
def validate_user_combo(numbers, min_size=4, max_size=4):
    if not isinstance(numbers, (list, tuple)) or not numbers:
        raise ValidationError("请提供号码列表")
    if not (min_size <= len(numbers) <= max_size):
        raise ValidationError(f"需要{min_size}-{max_size}个号码，当前{len(numbers)}个")
    if len(set(numbers)) != len(numbers):
        raise ValidationError("号码不得重复")
    for n in numbers:
        if not isinstance(n, int) or not 1 <= n <= 80:
            raise ValidationError(f"号码非法：{n!r}")
    return sorted(numbers)


# ---------------------------------------------------------------- 概率分析
def analyze_combo(numbers, records):
    """
    对用户输入的4码做概率分析
    返回：精确理论概率 + 经验指标（共现、期望、百分位、逐码、历史出现期）
    """
    combo = validate_user_combo(numbers)
    if not records:
        raise ValidationError("数据池为空，请先录入或抓取历史开奖数据")
    fx = Features(records)
    score, parts = fx.combo_score(combo)

    # 经验百分位：相对历史中真实出现过的全部4码组合
    seen_scores = sorted(fx.combo_score(c)[0] for c in fx.four.keys())
    percentile = (bisect_right(seen_scores, score) / len(seen_scores) * 100
                  if seen_scores else None)

    co_count = fx.four[tuple(combo)]
    hit_periods = [r["period"] for r in records if set(combo) <= set(r["numbers"])]
    pair_detail = [{"pair": [a, b],
                    "count": fx.pair[(min(a, b), max(a, b))],
                    "expected": round(fx.m_p, 2), "z": round(fx.z_pair(a, b), 2)}
                   for a, b in combinations(combo, 2)]
    number_detail = [{"number": n, "count": fx.freq[n],
                      "expected": round(fx.m_f, 2), "z": round(fx.z_num(n), 2)}
                     for n in combo]
    return {
        "combo": combo,
        "theory": exact_probability(),
        "empirical": {
            "co_occurrence": co_count,
            "co_expected": round(fx.m_4, 2),
            "four_z": parts["four_z"],
            "score": round(score, 4),
            "score_parts": parts,
            "percentile_among_observed": round(percentile, 1) if percentile is not None else None,
            "observed_basis": len(seen_scores),
            "hit_periods": hit_periods,
        },
        "numbers": number_detail,
        "pairs": pair_detail,
        "sample_periods": fx.n,
        "disclaimer": ("理论中奖概率对任意4码完全相同且固定；经验评分仅反映历史相对强弱，"
                       "不构成对未来开奖的预测或收益保证。"),
    }


# ---------------------------------------------------------------- 置信度
def bootstrap_stability(records, targets, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """固定种子的期级自助重采样：目标组合留在候选集且排名前5的比例"""
    if not records:
        return {t: 0.0 for t in targets}
    rng = random.Random(seed)
    idx = list(range(len(records)))
    hits = {t: 0 for t in targets}
    target_sets = {t: set(t) for t in targets}
    for _ in range(rounds):
        sub = [records[rng.choice(idx)] for _ in idx]
        f = Features(sub)
        pool = set(f.candidates())
        top5 = {r["combo"] for r in f.ranked_combos()[:5]}
        for t in targets:
            if target_sets[t] <= pool and t in top5:
                hits[t] += 1
    return {t: hits[t] / rounds for t in targets}


def recommend(records, last_n=None, top_k=5):
    """
    N+1 日推荐。
    last_n: 只使用最近N期（None=全部；会与数据池整体合并，池本身即累积数据）
    返回推荐组合 + 置信度 + 方法说明
    """
    if not records:
        raise ValidationError("没有任何历史数据，无法生成推荐；请先录入或抓取数据")
    use = records if last_n is None else records[-last_n:]
    if len(use) < 5:
        raise ValidationError(f"有效历史仅{len(use)}期，至少需要5期才能形成统计")

    fx = Features(use)
    pool = fx.candidates()
    ranked = fx.ranked_combos(pool)
    scores = [r["score"] for r in ranked]
    mu = sum(scores) / len(scores)
    sd = sqrt(sum((s - mu) ** 2 for s in scores) / len(scores)) or 1e-9

    targets = [r["combo"] for r in ranked[:top_k]]
    stability = bootstrap_stability(use, targets)

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
    return {
        "recommendations": out,
        "candidate_pool": pool,
        "basis": {"periods_used": len(use),
                  "first_period": use[0]["period"], "last_period": use[-1]["period"],
                  "sample_factor": round(sample_factor, 2)},
        "theory": exact_probability(),
        "method": ("12码候选集(八区保底)内枚举495组，按 .40号码z+.30共现z+.25四码共现z+"
                   ".05分区离散 排序；置信度=固定种子自助200轮稳定度/区分裕度/样本充分度。"),
        "disclaimer": ("置信度是历史数据内部的相对指标，不是中奖概率；任意4码真实中奖概率恒为"
                       f"{P4*100:.3f}%(约1/{1/P4:.0f})。"),
    }


# ---------------------------------------------------------------- 组合补充优化
def optimize_combo(numbers, records):
    """
    用数据池自动补充/优化用户输入的1-10码：
      * 依据“与用户号码的两两共现z + 自身评分”挑选伴侣号，补足到10码
      * 在10码轮内，给出含≥2个用户号码的最优4码，以及全局最优4码
    """
    combo = validate_user_combo(numbers, min_size=1, max_size=10)
    if not records:
        raise ValidationError("数据池为空，无法优化")
    fx = Features(records)
    need = max(0, 10 - len(combo))
    user = set(combo)
    companions = []
    for n in range(1, 81):
        if n in user:
            continue
        affinity = (sum(fx.z_pair(n, u) for u in combo) / len(combo)
                    if combo else 0)
        rank = affinity + 0.5 * fx.z_num(n)
        companions.append((rank, n))
    companions.sort(reverse=True)
    added = [n for _, n in companions[:need]]
    wheel = sorted(user | set(added))

    rows = fx.ranked_combos(wheel)
    with_user = [r for r in rows if len(user & set(r["combo"])) >= min(2, len(user))]
    return {
        "input": combo,
        "companions": added,
        "wheel": wheel,
        "best_with_input": [{"combo": list(r["combo"]), "score": round(r["score"], 4)}
                            for r in with_user[:5]],
        "best_global": [{"combo": list(r["combo"]), "score": round(r["score"], 4)}
                        for r in rows[:5]],
        "theory": exact_probability(),
        "disclaimer": "优化仅提升历史维度上的搭配强度，不改变理论中奖概率。",
    }
