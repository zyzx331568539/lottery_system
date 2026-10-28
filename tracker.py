# -*- coding: utf-8 -*-
"""
tracker.py —— 推荐结果跟踪：与每日开奖对比、命中率统计、策略自适应

职责
====
1. 把每次 ``recommend`` / ``recommend_m`` 的输出固化为"推荐快照"（data/recommendations.json），
   快照明确标注其预测的目标期号 = 数据池最新期号 + 1。
2. 当新一期开奖入库（手工录入 / 爬虫）后，``reconcile`` 将所有尚未对比的快照
   与实际开奖逐组比对，记录命中数 / 命中号码（幂等：已对比的不重复处理）。
3. ``hit_rate_report`` 聚合命中分布、平均命中、≥1/≥2 命中率，以及与超几何
   随机基准的 z 检验。
4. ``strategy_stats`` / ``best_strategy`` 按 (size, last_n) 维度统计历史命中，
   供 ``recommend`` 的 ``auto_strategy`` 选项自动选用历史表现最优的窗口。

设计约束
========
* 不修改任何已有评分逻辑的数值；只做"记录→对比→统计→选参"。
* 对比用目标期号匹配数据池，期号不存在则保持未对比（等待开奖）。
* 命中定义：推荐组合与开奖20码的交集大小（与 backtest 一致）。
* 策略调整只作用于可选参数 ``last_n``，权重等核心口径保持不变。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime
from math import comb, sqrt

from data_store import ValidationError, validate_period

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REC_PATH = os.path.join(DATA_DIR, "recommendations.json")

# 策略选择的统计门槛：至少 N 个对比样本才认为某参数组有统计意义
MIN_STRATEGY_SAMPLES = 10
# 切换参数的安全边际：新策略的 top1 平均命中需优于当前默认（last_n=None）
# 且差值大于该阈值才采纳，避免样本噪声导致频繁抖动
STRATEGY_EDGE = 0.05

_LOCK = threading.RLock()
_cache: dict | None = None
_cache_mtime: float | None = None


def ensure_store() -> None:
    """确保 recommendations.json 存在且结构合法（首次运行即创建空文件）。

    避免用户在生成任何推荐之前查询时看到"文件不存在"的误导信息。
    """
    if not os.path.exists(REC_PATH):
        _save({"records": []})


# ---------------------------------------------------------------- 存储层
def _load(force: bool = False) -> dict:
    """读取 recommendations.json（带文件 mtime 缓存，多线程安全）"""
    global _cache, _cache_mtime
    with _LOCK:
        try:
            mtime = os.path.getmtime(REC_PATH) if os.path.exists(REC_PATH) else None
        except OSError:
            mtime = None
        if not force and _cache is not None and _cache_mtime == mtime:
            return _cache
        if os.path.exists(REC_PATH):
            with open(REC_PATH, "r", encoding="utf-8") as f:
                doc = json.load(f)
        else:
            doc = {"records": []}
        if "records" not in doc or not isinstance(doc["records"], list):
            doc = {"records": []}
        _cache = doc
        _cache_mtime = mtime
        return doc


def _save(doc: dict) -> None:
    global _cache, _cache_mtime
    with _LOCK:
        os.makedirs(os.path.dirname(REC_PATH), exist_ok=True)
        tmp = REC_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        os.replace(tmp, REC_PATH)
        _cache = doc
        try:
            _cache_mtime = os.path.getmtime(REC_PATH)
        except OSError:
            _cache_mtime = None


def _make_id() -> str:
    return "%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                      hashlib.sha1(("%f" % time.time()).encode()).hexdigest()[:8])


def _target_period(records) -> int:
    """推荐预测的目标期号 = 所用数据最后一期 + 1"""
    if not records:
        raise ValidationError("无历史数据，无法确定目标期号")
    return records[-1]["period"] + 1


def _normalize_combos(recommendations) -> list:
    out = []
    for r in recommendations:
        combo = tuple(int(x) for x in r["combo"])
        out.append({"combo": list(combo),
                    "score": round(float(r.get("score", 0.0)), 6),
                    "confidence": round(float(r.get("confidence", 0.0)), 6)})
    return out


# ---------------------------------------------------------------- 写入推荐快照
def save_recommendation(records, size: int, result: dict,
                        last_n=None, top_k: int = 5) -> dict:
    """把一次推荐结果落盘为快照，返回该快照 dict。

    result 为 ``predictor.recommend`` / ``msize.recommend_m`` 的返回值。
    target_period 由 records 末期 +1 推导；同一目标期同一规模会保留多条快照
    （用户可能多次点击），查询时按生成时间取最新即可。
    """
    if not isinstance(size, int) or size not in (1, 2, 3, 4):
        raise ValidationError("size 必须是 1/2/3/4")
    rec = {
        "id": _make_id(),
        "target_period": _target_period(records),
        "target_date": None,
        "size": size,
        "recommendations": _normalize_combos(result.get("recommendations", [])),
        "candidate_pool": sorted(int(x) for x in result.get("candidate_pool", [])),
        "strategy": {"last_n": last_n, "top_k": int(top_k)},
        "basis": dict(result.get("basis", {}) or {}),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "compared": False,
        "compared_at": None,
        "result": None,
    }
    with _LOCK:
        doc = _load(force=True)
        doc["records"].append(rec)
        _save(doc)
    return rec


# ---------------------------------------------------------------- 对比
def _compare_one(rec: dict, actual_numbers) -> None:
    """就地把对比结果写入 rec['result']（不抛异常）"""
    actual = set(int(x) for x in actual_numbers)
    per = []
    best_hit = -1
    best_idx = 0
    for i, r in enumerate(rec["recommendations"]):
        combo = set(int(x) for x in r["combo"])
        hits = len(combo & actual)
        hit_nums = sorted(combo & actual)
        per.append({"combo": r["combo"], "hits": hits,
                    "hit_numbers": hit_nums,
                    "confidence": r.get("confidence", 0.0)})
        if hits > best_hit:
            best_hit = hits
            best_idx = i
    rec["compared"] = True
    rec["compared_at"] = datetime.now().isoformat(timespec="seconds")
    rec["result"] = {
        "actual": sorted(actual),
        "per_combo": per,
        "best_hit": best_hit,
        "best_hit_index": best_idx,
        "top1_hit": per[0]["hits"] if per else 0,
        "candidate_hits": len(set(rec["candidate_pool"]) & actual),
    }


def reconcile(store) -> dict:
    """扫描所有未对比快照，若目标期已在数据池则对比。幂等，返回处理摘要。

    对比用目标期号精确匹配 pool；期号不存在则保持未对比（等待开奖）。
    """
    pool = {r["period"]: r["numbers"] for r in store.all()}
    compared = skipped = missing = 0
    with _LOCK:
        doc = _load(force=True)
        for rec in doc["records"]:
            if rec.get("compared"):
                skipped += 1
                continue
            tp = rec.get("target_period")
            if tp in pool:
                _compare_one(rec, pool[tp])
                compared += 1
            else:
                missing += 1
        if compared:
            _save(doc)
    return {"compared": compared, "skipped": skipped,
            "still_pending": missing,
            "total": len(doc["records"])}


# ---------------------------------------------------------------- 查询
def list_recommendations(size: int | None = None,
                         compared: bool | None = None,
                         limit: int | None = None) -> list:
    """返回快照列表（最新在前）；compared=None 不过滤；limit 取前 N 条。"""
    doc = _load()
    recs = doc["records"]
    if size is not None:
        recs = [r for r in recs if r.get("size") == size]
    if compared is not None:
        recs = [r for r in recs if bool(r.get("compared")) == compared]
    recs = sorted(recs, key=lambda r: r.get("generated_at", ""), reverse=True)
    if limit is not None:
        recs = recs[:limit]
    return recs


def get_recommendation(rid: str) -> dict | None:
    for r in _load()["records"]:
        if r.get("id") == rid:
            return r
    return None


# ---------------------------------------------------------------- 命中率报告
def _hyper_baseline(size: int) -> dict:
    """size 码命中 k 个的超几何概率 Hypergeometric(80, 20, size)"""
    c80s = comb(80, size)
    dist = {k: comb(20, k) * comb(60, size - k) / c80s for k in range(size + 1)}
    mean = size * 20 / 80
    var = size * 0.25 * 0.75 * (80 - size) / (80 - 1)
    return {"distribution": dist, "mean": mean,
            "sd": sqrt(var),
            "at_least_1": 1.0 - dist[0],
            "at_least_2": sum(dist[k] for k in range(2, size + 1))}


def hit_rate_report(size: int | None = None, limit: int | None = None) -> dict:
    """聚合已对比快照的命中统计。

    limit：只使用最近 N 条已对比快照（None=全部），用于观察近期趋势。
    命中口径：取每个快照的 **最佳组合命中数**（best_hit）与 **top1 命中数**。
    """
    recs = [r for r in _load()["records"] if r.get("compared") and r.get("result")]
    if size is not None:
        recs = [r for r in recs if r.get("size") == size]
    recs = sorted(recs, key=lambda r: r.get("compared_at", ""), reverse=True)
    if limit is not None:
        recs = recs[:limit]

    n = len(recs)
    out = {"samples": n, "size_filter": size, "recent_limit": limit}
    if n == 0:
        out["note"] = "暂无已对比的推荐记录；生成推荐后待开奖入库即可对比。"
        return out

    best_hits = [r["result"]["best_hit"] for r in recs]
    top1_hits = [r["result"]["top1_hit"] for r in recs]
    sizes = {r["size"] for r in recs}
    rep_size = size if size is not None else (max(sizes) if sizes else 4)
    base = _hyper_baseline(rep_size)

    dist_best = {k: best_hits.count(k) for k in range(rep_size + 1)}
    mean_best = sum(best_hits) / n
    ge1_best = sum(1 for h in best_hits if h >= 1) / n
    ge2_best = sum(1 for h in best_hits if h >= 2) / n

    mean_top1 = sum(top1_hits) / n
    ge1_top1 = sum(1 for h in top1_hits if h >= 1) / n
    ge2_top1 = sum(1 for h in top1_hits if h >= 2) / n

    # 与随机基准的 z 检验（top1 命中数近似正态；n 较小时仅作参考）
    se = base["sd"] / sqrt(n)
    z_top1 = (mean_top1 - base["mean"]) / se if se > 0 else 0.0

    # 候选池命中率：12 码候选池平均命中数（理论 E = 12·20/80 = 3）
    cand_hits = [r["result"].get("candidate_hits", 0) for r in recs]
    mean_cand = sum(cand_hits) / n

    out.update({
        "best_hit": {
            "distribution": {str(k): dist_best[k] for k in range(rep_size + 1)},
            "distribution_pct": {str(k): round(dist_best[k] / n * 100, 2)
                                 for k in range(rep_size + 1)},
            "mean": round(mean_best, 4),
            "at_least_1_pct": round(ge1_best * 100, 2),
            "at_least_2_pct": round(ge2_best * 100, 2),
        },
        "top1_hit": {
            "mean": round(mean_top1, 4),
            "at_least_1_pct": round(ge1_top1 * 100, 2),
            "at_least_2_pct": round(ge2_top1 * 100, 2),
            "z_vs_baseline": round(z_top1, 3),
        },
        "candidate_pool": {
            "mean_hits": round(mean_cand, 4),
            "baseline_mean": round(12 * 20 / 80, 3),
        },
        "baseline": {
            "size": rep_size,
            "mean_hits": round(base["mean"], 3),
            "sd_hits": round(base["sd"], 3),
            "distribution_pct": {str(k): round(base["distribution"][k] * 100, 3)
                                 for k in range(rep_size + 1)},
            "at_least_1_pct": round(base["at_least_1"] * 100, 3),
            "at_least_2_pct": round(base["at_least_2"] * 100, 3),
        },
        "verdict": _verdict(mean_top1, z_top1, n, rep_size),
        "recent": [{
            "id": r["id"],
            "target_period": r["target_period"],
            "size": r["size"],
            "generated_at": r["generated_at"],
            "compared_at": r["compared_at"],
            "best_hit": r["result"]["best_hit"],
            "top1_hit": r["result"]["top1_hit"],
            "top1_combo": r["recommendations"][0]["combo"]
            if r["recommendations"] else [],
            "actual": r["result"]["actual"],
        } for r in recs[:30]],
    })
    return out


def _verdict(mean_top1: float, z: float, n: int, size: int) -> str:
    base_mean = size * 20 / 80
    if n < MIN_STRATEGY_SAMPLES:
        return ("样本仅 %d 期，结论不确定性大；需累积 ≥%d 期对比数据再判断。"
                % (n, MIN_STRATEGY_SAMPLES))
    if abs(z) <= 1.96:
        return ("top1 平均命中 %.2f，与随机基准(%.2f)无显著差异（z=%.2f），"
                "当前策略暂未证明具备超越随机的预测力。" % (mean_top1, base_mean, z))
    if z > 1.96:
        return "top1 平均命中 %.2f，显著高于随机基准（z=%.2f），策略有效。" % (mean_top1, z)
    return "top1 平均命中 %.2f，显著低于随机基准（z=%.2f），需调整策略或检查因子口径。" % (mean_top1, z)


# ---------------------------------------------------------------- 策略自适应
def strategy_stats(size: int | None = None) -> list:
    """按 (size, last_n) 聚合已对比快照的 top1 平均命中，供选参。

    last_n=None 记为 "all"（使用全部历史）。
    """
    groups: dict = {}
    for r in _load()["records"]:
        if not r.get("compared") or not r.get("result"):
            continue
        if size is not None and r.get("size") != size:
            continue
        ln = r.get("strategy", {}).get("last_n")
        key = (r["size"], "all" if ln is None else ln)
        g = groups.setdefault(key, {"size": r["size"], "last_n": ln,
                                    "samples": 0, "top1_hits": [],
                                    "best_hits": []})
        g["samples"] += 1
        g["top1_hits"].append(r["result"]["top1_hit"])
        g["best_hits"].append(r["result"]["best_hit"])
    rows = []
    for (sz, _ln), g in groups.items():
        rows.append({
            "size": sz,
            "last_n": g["last_n"],
            "samples": g["samples"],
            "mean_top1": round(sum(g["top1_hits"]) / g["samples"], 4),
            "mean_best": round(sum(g["best_hits"]) / g["samples"], 4),
            "at_least_1_pct": round(
                sum(1 for h in g["top1_hits"] if h >= 1) / g["samples"] * 100, 2),
        })
    rows.sort(key=lambda x: (x["size"], -(x["samples"]), -(x["mean_top1"])))
    return rows


def best_strategy(size: int, min_samples: int = MIN_STRATEGY_SAMPLES,
                  edge: float = STRATEGY_EDGE) -> dict | None:
    """返回 size 下历史最优的 last_n；不满足门槛则返回 None（用默认）。

    选择规则：在样本数 ≥ min_samples 的策略中，取 top1 平均命中最高者；
    若其相对默认（last_n=None）优势 < edge，则不切换（避免噪声抖动）。
    """
    rows = [r for r in strategy_stats(size) if r["samples"] >= min_samples]
    if not rows:
        return None
    rows.sort(key=lambda x: -x["mean_top1"])
    best = rows[0]
    default = next((r for r in rows if r["last_n"] is None), None)
    if default is not None and best["mean_top1"] - default["mean_top1"] < edge:
        return None
    return {"size": size, "last_n": best["last_n"],
            "samples": best["samples"],
            "mean_top1": best["mean_top1"]}
