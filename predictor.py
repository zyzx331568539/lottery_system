# -*- coding: utf-8 -*-
"""
predictor.py —— 概率分析与 N+1 日推荐核心

方法论（全部可复核、确定性输出；自助评分循环用 NumPy 向量化，结果与纯 Python 路径位级一致）：
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
from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import combinations
from math import comb, sqrt

import numpy as np

from data_store import ValidationError, validate_numbers

# ---------------------------------------------------------------- 常量
P4 = comb(76, 16) / comb(80, 20)     # 指定4码全中
P2 = comb(78, 18) / comb(80, 20)     # 指定2码全中
HALFLIFE = 10                        # 近期权重半衰期(期)
CANDIDATE_SIZE = 12
BOOTSTRAP_ROUNDS = 200
BOOTSTRAP_SEED = 20260920
W_NUM, W_PAIR, W_FOUR, W_SPREAD = 0.40, 0.30, 0.25, 0.05


# ---------------------------------------------------------------- 内部工具
@lru_cache(maxsize=64)
def _decay_template(n):
    """指数衰减权重模板（升序：旧→新），与逐次 0.5**((n-1-i)/HALFLIFE) 位级一致。"""
    return tuple(0.5 ** ((n - 1 - i) / HALFLIFE) for i in range(n))


def _mask_weighted_count(bits, mult):
    """位掩码 bits 中被抽中的记录期，按重采样出现次数 mult[k] 求和（整数结果与 Counter 累加一致）。"""
    total = 0
    while bits:
        lb = bits & -bits
        total += mult[lb.bit_length() - 1]
        bits ^= lb
    return total


# 4 码组合内 6 个二码对的位置（combinations 字典序，对每行相同）
_PAIR_LEFT = np.array((0, 0, 0, 1, 1, 2), dtype=np.int64)
_PAIR_RIGHT = np.array((1, 2, 3, 2, 3, 3), dtype=np.int64)


def _neumaier_sum(cols):
    """与 CPython 3.12+ 内置 sum() 相同的 Neumaier 补偿求和（逐行向量化）。

    内置 sum() 对纯 float 序列不再是朴素顺序累加（会加回舍入补偿 c），
    朴素左折叠会在 ~1/3 的行上产生 1 ULP 差异；此处逐元素复刻其分支与运算：
        t = s + x
        c += (s - t) + x   若 |s| >= |x|，否则 (x - t) + s
        s = t
    返回 s + c。cols 为按求和顺序排列的同形数组序列。
    """
    s = np.zeros_like(cols[0])
    c = np.zeros_like(cols[0])
    for x in cols:
        t = s + x
        ge = np.abs(s) >= np.abs(x)
        c = c + np.where(ge, (s - t) + x, (x - t) + s)
        s = t
    return s + c


@lru_cache(maxsize=64)
def _combo_indices(pool_size, k):
    """候选池大小 → k 码组合的池内索引 (C(pool_size,k), k)。

    行序与 itertools.combinations(range(pool_size), k) 完全一致（逐行可对拍）。
    """
    return np.array(tuple(combinations(range(pool_size), k)), dtype=np.int64)


@lru_cache(maxsize=16)
def _combo_structures(pool_size):
    """候选池大小 → 4码组合的池内索引 (R,4)（m=4 专用，语义见 _combo_indices）。"""
    return _combo_indices(pool_size, 4)


def _bootstrap_choice_matrix(rng, n, rounds):
    """一次性预生成 rounds×n 个期索引，逐元素等价于
    ``[[rng.choice(list(range(n))) for _ in range(n)] for _ in range(rounds)]``。

    等价性（CPython 3.12 random 可观察行为，单测已对拍锁定）：
      * MT 以 32 位词为单位顺序输出；``getrandbits(32*w)`` 的整数按小端拆词与
        逐次 ``getrandbits32()`` 的词序完全一致；
      * ``_randbelow_with_getrandbits`` 取每词高 k 位（k=n.bit_length()），
        值 >= n 时顺序消耗下一词重试。此处批量取词后以**相同顺序**向量化拒绝，
        接受值序列与逐次调用位级相同（跨 n=1..50000、多种子、200×56 实形对拍）；
      * 多取的尾词只影响该 rng 的后续状态；调用方用完抽样即丢弃 rng，无外部影响。

    返回 np.ndarray，shape=(rounds, n)，dtype=int64。
    """
    if not (0 < n < (1 << 31)):
        # 极端期数（现实中不可能出现）回退逐次调用，优先保正确
        idx = list(range(n))
        return np.array([[rng.choice(idx) for _ in range(n)]
                         for _ in range(rounds)], dtype=np.int64)
    needed = rounds * n
    k = n.bit_length()
    shift = np.uint32(32 - k)
    accepted = []
    got = 0
    while got < needed:
        # 接受概率 n/2^k：按 1.25 倍余量取词并加固定缓冲，正常 1~2 次取齐
        remain = needed - got
        chunk = remain + (remain >> 2) + 1024
        words = np.frombuffer(
            rng.getrandbits(32 * chunk).to_bytes(4 * chunk, "little"),
            dtype=np.uint32)
        vals = (words >> shift).astype(np.int64)
        vals = vals[vals < n]
        accepted.append(vals)
        got += vals.size
    return np.concatenate(accepted)[:needed].reshape(rounds, n)


class _RecordIndex:
    """开奖记录的位掩码索引：号码/两两共现是否出现在第 k 期用一个整数位表示。

    用于自助重采样场景：把“重建 4 码共现计数器”（每期 C(20,4)=4845 次字典插入）
    换成几次大整数按位与 + 置位求和，数值结果完全相同。
    """
    __slots__ = ("n", "nmask", "pmask", "tmask", "wtemplate", "occ")

    def __init__(self, records, with_triples=False):
        n = len(records)
        nmask = [0] * 81
        pmask = {}
        tmask = {} if with_triples else None
        occ = np.zeros((81, n), dtype=np.bool_)   # occ[号码, 期] = 该号码是否出现在该期
        for k, r in enumerate(records):
            bit = 1 << k
            nums = r["numbers"]
            for x in nums:
                nmask[x] |= bit
            occ[nums, k] = True
            for a, b in combinations(nums, 2):
                pmask[(a, b)] = pmask.get((a, b), 0) | bit
            if with_triples:
                for c in combinations(nums, 3):
                    tmask[c] = tmask.get(c, 0) | bit
        self.n = n
        self.nmask = nmask
        self.pmask = pmask
        self.tmask = tmask
        self.wtemplate = _decay_template(n)
        self.occ = occ

    def four_mask(self, c):
        bits = self.nmask[c[0]]
        for x in c[1:]:
            bits &= self.nmask[x]
        return bits


# ---------------------------------------------------------------- 跨请求观测评分缓存
# 百分位有序表只依赖数据池内容，与请求无关；同一数据池的重复分析（Web 高频场景）
# 直接复用。仅缓存浮点列表（55期四码表约8MB），按内容签名保留最近2个数据池版本。
_OBSERVED_CACHE: "OrderedDict" = OrderedDict()
_OBS_CACHE_MAX = 2


def pool_signature(records):
    """数据池内容签名：期数 + 每期(期号, 号码元组)；数据一变签名即变，杜绝脏缓存。"""
    return (len(records),
            tuple((r.get("period"), tuple(r["numbers"])) for r in records))


def get_observed_scores(records, size, builder):
    """取（或惰性构建并缓存）指定规模的历史观测组合评分有序表。"""
    sig = pool_signature(records)
    bucket = _OBSERVED_CACHE.get(sig)
    if bucket is None:
        bucket = {}
        _OBSERVED_CACHE[sig] = bucket
        _OBSERVED_CACHE.move_to_end(sig)
        while len(_OBSERVED_CACHE) > _OBS_CACHE_MAX:
            _OBSERVED_CACHE.popitem(last=False)
    if size not in bucket:
        bucket[size] = builder()
        _OBSERVED_CACHE.move_to_end(sig)
    return bucket[size]


def clear_caches():
    """清空进程级缓存（数据池被替换/测试隔离时调用）。"""
    _OBSERVED_CACHE.clear()
    _decay_template.cache_clear()


def validate_records(records, min_count=1):
    """入口级数据校验：合法路径行为不变，非法结构尽早抛出明确异常。"""
    if not isinstance(records, list):
        raise ValidationError("历史数据格式非法：应为列表")
    if len(records) < min_count:
        raise ValidationError(f"历史数据不足：至少需要{min_count}期，当前{len(records)}期")
    for r in records:
        nums = r.get("numbers") if isinstance(r, dict) else None
        if not isinstance(nums, list) or len(nums) != 20:
            raise ValidationError("历史数据格式非法：每期必须含20个号码")
        if len(set(nums)) != 20 or any(not isinstance(x, int)
                                       or isinstance(x, bool) or not 1 <= x <= 80
                                       for x in nums):
            raise ValidationError("历史数据格式非法：号码须为1-80范围内20个不重复整数")
    return records


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
        self._observed_scores = None      # 懒加载：历史观测组合评分有序表
        self._observed_scores_extra = {}  # 懒加载：1/2/3 码等其他规模的评分有序表
        self._build()
        self._init_stats()

    def _init_stats(self):
        """H0 下的均值/标准差（抽取为方法，供重采样特征复用同一套公式）。"""
        n = self.n
        self.m_f, self.sd_f = 0.25 * n, sqrt(0.1875 * n)
        weights = _decay_template(n)
        w_sum, w_sq = sum(weights), sum(w * w for w in weights)
        self.m_w, self.sd_w = 0.25 * w_sum, sqrt(0.1875 * w_sq)
        self.m_p, self.sd_p = P2 * n, sqrt(P2 * (1 - P2) * n)
        self.m_4, self.sd_4 = P4 * n, sqrt(P4 * n)

    def _build(self):
        weights = _decay_template(self.n)
        for i, r in enumerate(self.records):
            nums = r["numbers"]
            w = weights[i]
            self.freq.update(nums)
            for n in nums:
                self.wfreq[n] += w
            self.pair.update(combinations(nums, 2))
            self.four.update(combinations(nums, 4))

    def observed_scores_sorted(self):
        """历史真实出现过的全部四码组合评分（升序，懒加载并缓存）。"""
        if self._observed_scores is None:
            self._observed_scores = sorted(self.combo_score(c)[0]
                                           for c in self.four.keys())
        return self._observed_scores

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


def slide_window_features(fx, leaving_nums, entering_nums, window):
    """把基于上一滑窗的 Features 前移一期：删离开期、加进入期。

    严格等价性：
      * freq/pair/four（及 FeaturesM.triple）都是整数计数，增量加减与全量重建结果相同；
      * 衰减权重是“窗内槽位权重”，滑窗后所有权重重映射，故 wfreq 按权重模板整窗重建，
        累加顺序与模板值均与 Features(window) 位级一致；
      * 窗长固定 → _init_stats() 算出的均值/标准差不变。
    """
    four = fx.four
    pair = fx.pair
    freq = fx.freq
    for x in leaving_nums:
        freq[x] -= 1
        if freq[x] == 0:
            del freq[x]
    freq.update(entering_nums)
    for c in combinations(leaving_nums, 2):
        pair[c] -= 1
        if pair[c] == 0:
            del pair[c]
    pair.update(combinations(entering_nums, 2))
    for c in combinations(leaving_nums, 4):
        four[c] -= 1
        if four[c] == 0:
            del four[c]
    four.update(combinations(entering_nums, 4))
    triple = getattr(fx, "triple", None)
    if triple is not None:
        for c in combinations(leaving_nums, 3):
            triple[c] -= 1
            if triple[c] == 0:
                del triple[c]
        triple.update(combinations(entering_nums, 3))
    wfreq = Counter()
    weights = _decay_template(len(window))
    for i, r in enumerate(window):
        w = weights[i]
        for x in r["numbers"]:
            wfreq[x] += w
    fx.wfreq = wfreq
    fx.records = window
    fx.n = len(window)
    fx._observed_scores = None
    fx._observed_scores_extra = {}
    fx._init_stats()
    if triple is not None:                       # FeaturesM 三码维度的期望/SD
        fx._init_triple_stats()
    return fx


class _SampledFeatures(Features):
    """期级有放回重采样上的特征：与对同一份 sub 调用 Features(sub) 数值完全等价，
    但不再构建 pair/four 全量计数器，改为位掩码按需查询。

    等价性要点：
      * freq/pair/four 的计数都是整数，汇总次序不影响结果；
      * wfreq 是“槽位权重”，严格按采样槽位顺序累加，权重取自同一模板（位级一致）；
      * m_*/sd_* 只依赖期数 n（重采样长度恒等于 n），公式不变。
    """

    def __init__(self, records, chosen, index):
        self.records = records
        self.n = len(records)
        self.pair = Counter()
        self.four = Counter()
        self._observed_scores = []
        self._observed_scores_extra = {}
        # chosen 可以是 list 或预生成的 ndarray 行（OPT-9 批量抽样）；
        # bincount 与逐槽 mult[k]+=1 同为整数计数，结果严格相等。
        chosen_arr = np.asarray(chosen, dtype=np.int64)
        mult_arr = np.bincount(chosen_arr, minlength=self.n)  # (n,) int64
        mult = mult_arr.tolist()
        self._mult = mult
        self._index = index

        # ---- 频次/加权频次：occ 矩阵一次性算出（OPT-8）
        # freq：整数矩阵乘，与逐期 Counter 累加严格相等（无浮点参与）。
        freq_arr = index.occ.astype(np.int64) @ mult_arr      # (81,) int64
        # wfreq：权重按“采样槽位”计，列按 chosen 槽序排列；
        # 不能用 matmul（BLAS 归约顺序不保证 → ULP 差异），改用 cumsum——
        # 其按定义逐列顺序累加，与 Python 逐槽 += 位级一致，取最后一列即总和。
        wts = np.asarray(index.wtemplate, dtype=np.float64)   # 槽位 i 的权重
        wfreq_arr = (index.occ[:, chosen_arr] * wts).cumsum(axis=1)[:, -1]
        self._freq_arr = freq_arr
        self._wfreq_arr = wfreq_arr
        # 同步保留 Counter 视图（仅 80 次字典写入，无大整数位扫描）：
        # 键集合与朴素实现完全一致（计数为 0 的号码不写入）。
        self.freq = Counter({n: int(freq_arr[n])
                             for n in range(1, 81) if freq_arr[n]})
        self.wfreq = Counter({n: float(wfreq_arr[n])
                              for n in range(1, 81) if wfreq_arr[n] > 0.0})
        self._init_stats()
        # 81 维号码 z 向量：逐元素与标量 z_num 同序运算（位级一致）
        self._znum_arr = (0.65 * (freq_arr - self.m_f) / self.sd_f
                          + 0.35 * (wfreq_arr - self.m_w) / (self.sd_w or 1e-9))

    def z_num(self, n):
        if self.n == 0:
            return 0.0
        return float(self._znum_arr[n])

    def candidates_vectorized(self):
        """Features.candidates 的向量化版本：80 码 z 排序一次完成，
        八区保底/补足逻辑与原方法逐步相同，返回同一有序列表。"""
        nums = np.arange(1, 81, dtype=np.int64)
        order = nums[np.lexsort((nums, -self._znum_arr[1:81]))].tolist()
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

    def z_pair(self, a, b):
        if self.n == 0:
            return 0.0
        key = (a, b) if a < b else (b, a)
        cnt = _mask_weighted_count(self._index.pmask.get(key, 0), self._mult)
        return (cnt - self.m_p) / (self.sd_p or 1e-9)

    def z_four(self, c):
        if self.n == 0:
            return 0.0
        c = tuple(sorted(c))
        cnt = _mask_weighted_count(self._index.four_mask(c), self._mult)
        return (cnt - self.m_4) / (self.sd_4 or 1e-9)

    # ---- NumPy 向量化评分表（495 组一次性算完）
    def vectorized_score_table(self, pool=None):
        """与 ranked_combos 同口径的向量化评分，返回 (pool, combo_num, score, 分量)。

        严格位级一致性要点：
          * 共现/频次计数是整数：occ(bool) @ mult(int64) 为整数精确求和，与逐期
            Counter 累加、位掩码置位求和完全相同（无浮点参与）；
          * z 值为逐元素标量 IEEE 运算 ((count-m)/sd)，与 Python 标量表达式同序；
          * v_num/v_pair 不用 np.sum（其归约顺序不保证），而是显式按
            combinations 字典序逐项左结合相加，再除以 4/6，与 Python 版同序；
          * 总分按与 combo_score 完全相同的左结合顺序写出：
            W_NUM*v_num + W_PAIR*v_pair + W_FOUR*v_four + W_SPREAD*v_spread。
        """
        pool = sorted(pool if pool is not None
                      else self.candidates_vectorized())
        pool_arr = np.asarray(pool, dtype=np.int64)
        combo_idx = _combo_structures(len(pool))            # (R,4) 池内索引
        combo_num = pool_arr[combo_idx]                     # (R,4) 真实号码
        occ_p = self._index.occ[pool_arr]                   # (P,n) bool
        mult = np.asarray(self._mult, dtype=np.int64)

        # 四码共现计数：(R,4,n) 同时出现 → @ 重采样倍数
        c4 = (occ_p[combo_idx].all(axis=1) @ mult)          # (R,) int64
        # 二码共现计数：每组 6 对 (R,6,n)
        left = combo_idx[:, _PAIR_LEFT]                     # (R,6)
        right = combo_idx[:, _PAIR_RIGHT]
        cp = ((occ_p[left] & occ_p[right]) @ mult)          # (R,6) int64

        # 号码 z：直接取构造期一次性算好的 81 维向量（与标量 z_num 同序运算）
        zn = self._znum_arr[combo_num]                      # (R,4)
        # 均值复刻内置 sum() 的 Neumaier 补偿求和（见 _neumaier_sum），
        # 再除以 4/6，与 Python 版 sum(...)/len 位级一致
        v_num = _neumaier_sum([zn[:, j] for j in range(4)]) / 4.0
        zp = (cp.astype(np.float64) - self.m_p) / (self.sd_p or 1e-9)  # (R,6)
        v_pair = _neumaier_sum([zp[:, j] for j in range(6)]) / 6.0
        v_four = (c4.astype(np.float64) - self.m_4) / (self.sd_4 or 1e-9)

        # 分区离散度：逐位标记每个组合内“首次出现的区段”，与 set 去重计数一致
        R = combo_num.shape[0]
        zones = (combo_num - 1) // 10                       # (R,4) 0..7
        seen = np.zeros((R, 8), dtype=np.bool_)
        distinct = np.zeros(R, dtype=np.int64)
        rows = np.arange(R)
        for j in range(4):
            zj = zones[:, j]
            is_new = ~seen[rows, zj]
            distinct += is_new
            seen[rows, zj] = True
        v_spread = (distinct - 1) / 3.0

        score = (W_NUM * v_num + W_PAIR * v_pair
                 + W_FOUR * v_four + W_SPREAD * v_spread)
        return pool, combo_num, score, v_num, v_pair, v_four, v_spread

    def ranked_combos_vectorized(self, pool=None):
        """ranked_combos 的向量化版本：行（combo/score/parts，含排序）逐字段一致。"""
        pool, combo_num, score, v_num, v_pair, v_four, v_spread = \
            self.vectorized_score_table(pool)
        # 排序键：-score 升序（高在前），平手按组合字典序（与 sorted(key=(-s,c)) 相同）
        order = np.lexsort((combo_num[:, 3], combo_num[:, 2],
                            combo_num[:, 1], combo_num[:, 0], -score))
        rows = []
        for i in order:
            t = tuple(int(x) for x in combo_num[i])
            rows.append({"combo": t, "score": float(score[i]),
                         "parts": {"number_z": round(float(v_num[i]), 3),
                                   "pair_z": round(float(v_pair[i]), 3),
                                   "four_z": round(float(v_four[i]), 3),
                                   "spread": round(float(v_spread[i]), 3)}})
        return rows

    def top5_combos_vectorized(self, pool=None):
        """自助循环专用：只取前5组合集合，省去构造495个 dict。"""
        pool, combo_num, score, *_ = self.vectorized_score_table(pool)
        order = np.lexsort((combo_num[:, 3], combo_num[:, 2],
                            combo_num[:, 1], combo_num[:, 0], -score))
        return {tuple(int(x) for x in combo_num[i]) for i in order[:5]}

    # ---- m=1/2/3 通用向量化评分（OPT-9；与 msize.combo_score_m 同口径）
    def vectorized_score_parts_m(self, pool, m):
        """候选集内 C(|pool|, m) 组的 m=1/2/3 评分分量一次性算完。

        返回 ``(pool, combo_num, v_num, v_pair, v_tri, v_spread)``，
        不参与该规模的分量位置为 None；总分由调用方（msize）按固定权重
        以与标量路径相同的左结合顺序组装。

        严格位级一致性（与 OPT-7/8 同一套纪律）：
          * 二码/三码共现计数为整数：occ 布尔按位与/连与后 @ 重采样倍数，
            与逐期 Counter、位掩码置位求和严格相同（无浮点参与）；
          * v_num/v_pair 用 _neumaier_sum 复刻内置 sum() 后再除以 m/C(m,2)，
            与 msize.combo_score_m 中 ``sum(...)/m`` 位级一致；
          * v_tri 与标量 z_triple 同为 (count-m_3)/sd_3；
          * v_spread=(不同区段数-1)/(m-1)，逐位标记首次区段与 set 去重同结果。
        m=3 要求对象含 m_3/sd_3（msize._SampledFeaturesM）。
        """
        pool = sorted(pool)
        pool_arr = np.asarray(pool, dtype=np.int64)
        combo_idx = _combo_indices(len(pool), m)             # (R,m) 池内索引
        combo_num = pool_arr[combo_idx]                      # (R,m) 真实号码
        occ_p = self._index.occ[pool_arr]                    # (P,n) bool
        mult = np.asarray(self._mult, dtype=np.int64)

        zn = self._znum_arr[combo_num]                       # (R,m)
        v_num = _neumaier_sum([zn[:, j] for j in range(m)]) / float(m)
        if m == 1:
            return pool, combo_num, v_num, None, None, None

        pair_pos = np.array(tuple(combinations(range(m), 2)), dtype=np.int64)
        left = combo_idx[:, pair_pos[:, 0]]                  # (R,C(m,2))
        right = combo_idx[:, pair_pos[:, 1]]
        cp = ((occ_p[left] & occ_p[right]) @ mult)           # (R,C(m,2)) int64
        zp = (cp.astype(np.float64) - self.m_p) / (self.sd_p or 1e-9)
        v_pair = (_neumaier_sum([zp[:, j] for j in range(pair_pos.shape[0])])
                  / float(pair_pos.shape[0]))
        if m == 2:
            return pool, combo_num, v_num, v_pair, None, None

        # m == 3：三码同期连与 @ 倍数（走 occ，不依赖 tmask 位掩码）
        # occ_p[combo_idx] 形状 (R,m,n)，组合成员轴为 axis=1（与 m=4 同构）
        c3 = occ_p[combo_idx].all(axis=1) @ mult             # (R,) int64
        v_tri = (c3.astype(np.float64) - self.m_3) / (self.sd_3 or 1e-9)

        zones = (combo_num - 1) // 10                         # (R,m) 0..7
        R = combo_num.shape[0]
        seen = np.zeros((R, 8), dtype=np.bool_)
        distinct = np.zeros(R, dtype=np.int64)
        rows = np.arange(R)
        for j in range(m):
            zj = zones[:, j]
            is_new = ~seen[rows, zj]
            distinct += is_new
            seen[rows, zj] = True
        v_spread = (distinct - 1).astype(np.float64) / float(m - 1)
        return pool, combo_num, v_num, v_pair, v_tri, v_spread


# ---------------------------------------------------------------- 输入校验
def validate_user_combo(numbers, min_size=4, max_size=4):
    if not isinstance(numbers, (list, tuple)) or not numbers:
        raise ValidationError("请提供号码列表")
    if not (min_size <= len(numbers) <= max_size):
        raise ValidationError(f"需要{min_size}-{max_size}个号码，当前{len(numbers)}个")
    if len(set(numbers)) != len(numbers):
        raise ValidationError("号码不得重复")
    for n in numbers:
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 80:
            raise ValidationError(f"号码非法：{n!r}")
    return sorted(numbers)


# ---------------------------------------------------------------- 概率分析
def analyze_combo(numbers, records):
    """
    对用户输入的4码做概率分析
    返回：精确理论概率 + 经验指标（共现、期望、百分位、逐码、历史出现期）
    """
    combo = validate_user_combo(numbers)
    validate_records(records)
    fx = Features(records)
    score, parts = fx.combo_score(combo)

    # 经验百分位：相对历史中真实出现过的全部4码组合。
    # 有序表按数据池内容签名跨请求缓存（首次构建 O(M)，后续请求仅一次二分）。
    seen_scores = get_observed_scores(records, 4, fx.observed_scores_sorted)
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
    """固定种子的期级自助重采样：目标组合留在候选集且排名前5的比例。

    优化：抽样索引一次性批量预生成（_bootstrap_choice_matrix，与逐次
    rng.choice 位级同序列，结果可复现），每轮用位掩码特征 _SampledFeatures
    替代全量 Features(sub)（省去 200×n×C(20,4) 次四码计数）。
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValidationError(f"自助轮次非法：{rounds!r}")
    if not records:
        return {t: 0.0 for t in targets}
    rng = random.Random(seed)
    n = len(records)
    choices = _bootstrap_choice_matrix(rng, n, rounds)
    index = _RecordIndex(records)
    hits = {t: 0 for t in targets}
    target_sets = {t: set(t) for t in targets}
    for r in range(rounds):
        f = _SampledFeatures(records, choices[r], index)
        # 候选集（80 码 z 排序）与 495 组评分/排序全部走 NumPy 向量化
        pool = f.candidates_vectorized()
        top5 = f.top5_combos_vectorized(pool)
        pool_set = set(pool)
        for t in targets:
            if target_sets[t] <= pool_set and t in top5:
                hits[t] += 1
    return {t: hits[t] / rounds for t in targets}


def recommend(records, last_n=None, top_k=5, auto_strategy=False):
    """
    N+1 日推荐。
    last_n: 只使用最近N期（None=全部；会与数据池整体合并，池本身即累积数据）
    auto_strategy: True 时用 tracker 历史命中率最优的 last_n 覆盖本参数
                   （样本不足或无显著优势则仍用 last_n 默认值，安全回退）。
    返回推荐组合 + 置信度 + 方法说明
    """
    if not records:
        raise ValidationError("没有任何历史数据，无法生成推荐；请先录入或抓取数据")
    if last_n is not None and (isinstance(last_n, bool)
                               or not isinstance(last_n, int) or last_n <= 0):
        raise ValidationError(f"近N期参数非法：{last_n!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError(f"推荐条数非法：{top_k!r}")
    if auto_strategy:
        # 延迟导入避免循环依赖；无合格策略时 best_strategy 返回 None
        from tracker import best_strategy
        picked = best_strategy(4)
        if picked is not None:
            last_n = picked["last_n"]
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
    validate_records(records)
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
