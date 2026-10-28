# -*- coding: utf-8 -*-
"""
crawler.py —— 每日数据抓取机制

数据源（data/crawler_config.json 可配置；默认 mock，仅用于演示机制）：
  * "mock"     ：确定性演示数据（按期号做种子生成，不代表真实开奖），明确标注 DEMO
  * "url_text" ：HTTP 抓取文本，每行一期，格式同手工录入（20码 / 期号+20码 / 期号+日期+20码）
  * "url_json" ：HTTP 抓取 JSON 数组，元素 {"period": 2026253, "date": "2026-09-10",
                                            "numbers": [20个1-80整数]}
  * "url_html" ：HTTP 抓取 HTML 网页，按内置站点配置（profile）提取期号/日期/20码
                 profile="ydniu_kl8"：适配 ydniu.com 快乐8，默认抓"近10期"列表页

机制：
  * staleness 检测：数据池超过26小时未更新 -> 判定陈旧
  * update_if_stale()：陈旧才抓，供“每日定时守护线程”和页面手动按钮共用
  * dry_run=True：只抓取、解析、校验并返回预览，绝不写入数据池
  * on_duplicate：skip(默认) | replace（抓回的期号已存在时的策略）
  * Scheduler：可开关的守护线程，按间隔检查（默认每小时看一次，陈旧即抓）
所有抓回来的记录都经过 data_store 的同一套校验，脏数据不会进池。
"""
import json
import os
import re
import random
import threading
import urllib.request
import datetime as dt

from data_store import ValidationError, validate_period, validate_numbers, \
    validate_date, parse_period_text

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "data", "crawler_config.json")

DEFAULT_CONFIG = {
    "source_type": "mock",      # mock | url_text | url_json | url_html
    "url": "",
    "html_profile": "ydniu_kl8",
    "on_duplicate": "skip",     # skip | replace（抓回的期号已存在时）
    "auto_update": False,       # 是否开启定时自动抓取
    "check_interval_seconds": 3600,
    "http_timeout_seconds": 10,
}

PREVIEW_LIMIT = 3


class CrawlError(RuntimeError):
    pass


# ---------------------------------------------------------------- HTML 站点配置
# ydniu 快乐8：近10期页面每期是一个 <em>期号</em> 开头的区块，
# 号码是区块内20个 <i class="ball …">NN</i> 元素，日期取区块内首个 YYYY-MM-DD。
# 只从 ball 元素取号码，日期/期号/开机号都不会被误提取。
YDNIU_DEFAULT_URL = "https://m.ydniu.com/kaijiang/kl8-10.html"
_YDNIU_BLOCK_RE = re.compile(
    r'<em>\s*(\d{7})期\s*</em>(.*?)(?=<em>\s*\d{7}期\s*</em>|\Z)', re.S)
_YDNIU_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_YDNIU_BALL_RE = re.compile(
    r'<i\b[^>]*class="[^"]*\bball\b[^"]*"[^>]*>\s*(\d{2})\s*</i>', re.I)


def _parse_ydniu_html(html):
    """
    返回 (records, errors)；records 元素 {period, date, numbers}，文档顺序（最新在前）
    号码只从 <i class="ball …">NN</i> 元素提取，避免把日期/期号/开机号误当号码。
    """
    records, errors = [], []
    seen = set()   # 页面顶部与历史区可能重复展示最新一期，按期号保留首个
    for m in _YDNIU_BLOCK_RE.finditer(html):
        period = int(m.group(1))
        if period in seen:
            continue
        seen.add(period)
        body = m.group(2)
        date_m = _YDNIU_DATE_RE.search(body)
        nums = [int(x) for x in _YDNIU_BALL_RE.findall(body)]
        if len(nums) != 20:
            errors.append({"period": period,
                           "message": f"提取到{len(nums)}个号码（应恰好20个），已跳过"})
            continue
        records.append({"period": period,
                        "date": date_m.group(1) if date_m else None,
                        "numbers": nums})
    return records, errors


HTML_PROFILES = {
    "ydniu_kl8": {"parser": _parse_ydniu_html, "default_url": YDNIU_DEFAULT_URL},
}


# ---------------------------------------------------------------- 配置
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- HTTP
def _http_get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "lottery-system/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


# ---------------------------------------------------------------- 解析为候选记录
def _auto_numbered(store, parsed):
    """给 url_text 中无期号的行分配期号（数据池最大期号+1起递增），与 bulk_add_text 一致"""
    latest = store.latest()
    next_auto = (latest["period"] + 1) if latest else None
    out, notes = [], []
    for rec in parsed:
        if rec["_auto"]:
            if next_auto is None:
                next_auto = 2026001
                notes.append({"line": 0,
                              "message": "空库且未给期号，已从2026001开始编号"})
            rec["period"] = next_auto
            next_auto += 1
        out.append(rec)
    return out, notes


def _collect(store, cfg):
    """
    抓取并解析为候选记录（不写入）。
    返回 (records, parse_errors, meta)；record = {period, date, numbers}
    meta 至少含 {"demo": bool}
    """
    stype = cfg.get("source_type", "mock")
    timeout = cfg.get("http_timeout_seconds", 10)

    if stype == "mock":
        latest = store.latest()
        next_period = (latest["period"] + 1) if latest else 2026001
        rng = random.Random(next_period)
        numbers = sorted(rng.sample(range(1, 81), 20))
        date = ((dt.date.fromisoformat(latest["date"]) + dt.timedelta(days=1)).isoformat()
                if latest and latest.get("date") else dt.date.today().isoformat())
        return [{"period": next_period, "date": date, "numbers": numbers}], [], \
               {"demo": True, "message": "DEMO 演示数据（非真实开奖）"}

    if stype == "url_text":
        if not cfg.get("url"):
            raise CrawlError("source_type=url_text 但未配置 url")
        parsed, errors = parse_period_text(_http_get(cfg["url"], timeout))
        numbered, notes = _auto_numbered(store, parsed)
        records = [{"period": r["period"], "date": r["date"], "numbers": r["numbers"]}
                   for r in numbered]
        return records, notes + errors, {"demo": False}

    if stype == "url_json":
        if not cfg.get("url"):
            raise CrawlError("source_type=url_json 但未配置 url")
        body = _http_get(cfg["url"], timeout)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise CrawlError(f"返回内容不是合法JSON：{e}")
        if not isinstance(payload, list):
            raise CrawlError("url_json 源要求返回记录数组")
        records, errors = [], []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                errors.append({"index": i, "message": "数组元素必须是对象"})
                continue
            records.append({"period": item.get("period"), "date": item.get("date"),
                            "numbers": item.get("numbers")})
        return records, errors, {"demo": False}

    if stype == "url_html":
        profile_name = cfg.get("html_profile", "ydniu_kl8")
        profile = HTML_PROFILES.get(profile_name)
        if profile is None:
            raise CrawlError(f"未知HTML站点配置：{profile_name}")
        url = cfg.get("url") or profile["default_url"]
        html = _http_get(url, timeout)
        records, errors = profile["parser"](html)
        return records, errors, {"demo": False, "url_used": url}

    raise CrawlError(f"未知数据源类型：{stype}")


def _check_record(rec):
    """校验单条候选记录，返回规范化后的 dict（period/date/numbers）"""
    period = validate_period(rec["period"])
    numbers = validate_numbers(rec["numbers"])
    date = validate_date(rec["date"]) if rec.get("date") else None
    return {"period": period, "date": date, "numbers": numbers}


# ---------------------------------------------------------------- 统一入口
def crawl(store, cfg=None, dry_run=False, on_duplicate=None):
    """
    立即执行一次抓取。
    dry_run=True：只解析校验并返回预览，不写入数据池。
    on_duplicate：skip(默认) | replace，未给时取 cfg.on_duplicate。
    """
    cfg = cfg or load_config()
    stype = cfg.get("source_type", "mock")
    dup = on_duplicate or cfg.get("on_duplicate", "skip")
    if dup not in ("skip", "replace"):
        raise CrawlError(f"on_duplicate 只能是 skip/replace：{dup}")
    try:
        raw_records, parse_errors, meta = _collect(store, cfg)
    except CrawlError:
        raise
    except Exception as e:
        raise CrawlError(f"抓取失败（{stype}）：{e}") from e

    valid, errors = [], list(parse_errors)
    for i, rec in enumerate(raw_records):
        try:
            valid.append(_check_record(rec))
        except ValidationError as e:
            errors.append({"index": i, "period": rec.get("period"),
                           "message": str(e)})

    exists = [r for r in valid if store.has_period(r["period"])]
    fresh = [r for r in valid if not store.has_period(r["period"])]

    if dry_run:
        return {
            "ran": True, "dry_run": True, "demo": meta.get("demo", False),
            "message": meta.get("message"),
            "parsed": len(raw_records), "valid": len(valid),
            "new_count": len(fresh), "exists_count": len(exists),
            "would_add": len(fresh),
            "would_replace": len(exists) if dup == "replace" else 0,
            "would_skip": len(exists) if dup == "skip" else 0,
            "periods": sorted(r["period"] for r in valid),
            "preview": [{
                "period": r["period"], "date": r["date"],
                "numbers": r["numbers"], "exists": store.has_period(r["period"])
            } for r in valid[:PREVIEW_LIMIT]],
            "records": [{"period": r["period"], "date": r["date"],
                         "numbers": r["numbers"],
                         "exists": store.has_period(r["period"])} for r in valid],
            "errors": errors,
        }

    added = replaced = skipped = 0
    periods = []
    for r in valid:
        source = "crawl(demo)" if meta.get("demo") else "crawl"
        status = store.add({"period": r["period"], "date": r["date"],
                            "numbers": r["numbers"], "source": source},
                           on_duplicate=dup)
        added += status == "added"
        replaced += status == "replaced"
        skipped += status == "skipped"
        periods.append(r["period"])
    result = {
        "ran": True, "dry_run": False, "demo": meta.get("demo", False),
        "message": meta.get("message"),
        "added": added, "replaced": replaced, "skipped": skipped,
        "periods": sorted(periods), "errors": errors,
    }
    if meta.get("url_used"):
        result["url_used"] = meta["url_used"]
    return result


def update_if_stale(store, cfg=None, force=False, dry_run=False, on_duplicate=None):
    """陈旧才抓（force=强制；dry_run=只预览不写入）；供定时任务与手动按钮共用"""
    cfg = cfg or load_config()
    st = store.staleness()
    if not force and not dry_run and not st["is_stale"]:
        return {"ran": False, "reason": "数据池仍然新鲜", "staleness": st}
    result = crawl(store, cfg, dry_run=dry_run, on_duplicate=on_duplicate)
    result["staleness_before"] = st
    return result


# ---------------------------------------------------------------- 定时守护
class Scheduler:
    """轻量守护线程：每 interval 检查一次，陈旧则抓"""
    def __init__(self, store, cfg=None):
        self.store = store
        self.cfg = cfg or load_config()
        self._stop = threading.Event()
        self._thread = None
        self.last_result = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if not self.cfg.get("auto_update"):
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="lottery-crawler")
        self._thread.start()

    def _loop(self):
        interval = self.cfg.get("check_interval_seconds", 3600)
        while not self._stop.wait(interval):
            try:
                self.last_result = update_if_stale(self.store, self.cfg)
            except CrawlError as e:
                self.last_result = {"error": str(e)}

    def stop(self):
        self._stop.set()
