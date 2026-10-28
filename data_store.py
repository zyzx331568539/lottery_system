# -*- coding: utf-8 -*-
"""
data_store.py —— 数据池管理
职责：
  * pool.json 的读取 / 原子写入（线程安全）
  * 开奖号码与期号 / 日期的严格校验
  * 文本批量录入解析（每行一期，多种格式）
  * 陈旧度（stale）与期号缺口（gap）检测
数据格式（data/pool.json）：
  {"records": [{"period": 2026223, "date": "2026-08-11",
                "numbers": [3, ... 共20个], "source": "seed|manual|crawl"}]}
"""
import json
import os
import re
import threading
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
POOL_PATH = os.path.join(DATA_DIR, "pool.json")

STALE_HOURS = 26          # 每日一开，超过26小时视为陈旧
DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


class ValidationError(ValueError):
    """带字段信息的校验错误"""


# ---------------------------------------------------------------- 校验
def validate_numbers(numbers):
    """返回排序后的20个号码；不合法抛 ValidationError"""
    if not isinstance(numbers, (list, tuple)):
        raise ValidationError("号码必须是列表")
    if len(numbers) != 20:
        raise ValidationError(f"每期必须恰好20个号码，当前{len(numbers)}个")
    out = []
    for x in numbers:
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValidationError(f"号码必须是整数：{x!r}")
        if not 1 <= x <= 80:
            raise ValidationError(f"号码超出1-80范围：{x}")
        out.append(x)
    dup = {n for n in out if out.count(n) > 1}
    if dup:
        raise ValidationError(f"期内号码重复：{sorted(dup)}")
    return sorted(out)


def validate_date(text):
    m = DATE_RE.match(str(text or "").strip())
    if not m:
        raise ValidationError(f"日期格式应为 YYYY-MM-DD：{text!r}")
    y, mo, d = map(int, m.groups())
    try:
        return dt.date(y, mo, d).isoformat()
    except ValueError as e:
        raise ValidationError(f"无效日期 {text}：{e}")


def validate_period(p):
    if isinstance(p, bool) or not isinstance(p, int):
        raise ValidationError(f"期号必须是整数：{p!r}")
    if not 2000000 < p < 2100000:   # 7位：4位年份 + 3位年内序号
        raise ValidationError(f"期号应为7位(年份+年内序)：{p}")
    return p


# ---------------------------------------------------------------- 文本解析
def parse_period_text(text):
    """
    解析批量录入文本。
    支持每行三种格式（空格 / 逗号 / 分号分隔，#开头为注释）：
      A. 20个号码
      B. 期号 + 20个号码
      C. 期号 + 日期(YYYY-MM-DD) + 20个号码
    无期号的行按录入顺序，在“调用方给出的起始期号”基础上递增。
    返回 (records, errors)；errors = [{"line": n, "message": ...}]
    """
    records, errors = [], []
    auto_seq = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = re.split(r"[\s,，;；]+", line)
        date_val = None
        date_tokens = [t for t in tokens if DATE_RE.match(t)]
        if len(date_tokens) > 1:
            errors.append({"line": lineno, "message": "一行出现多个日期"})
            continue
        if date_tokens:
            date_val = date_tokens[0]
            tokens.remove(date_val)
        try:
            ints = [int(t) for t in tokens]
        except ValueError:
            errors.append({"line": lineno, "message": f"存在非数字内容：{line}"})
            continue

        period = None
        if len(ints) == 21:
            period, nums = ints[0], ints[1:]
        elif len(ints) == 20:
            nums = ints
        else:
            errors.append({"line": lineno,
                           "message": f"应有20个号码(可前置期号)，当前{len(ints)}个整数"})
            continue
        try:
            nums = validate_numbers(nums)
            if period is not None:
                period = validate_period(period)
            if date_val is not None:
                date_val = validate_date(date_val)
        except ValidationError as e:
            errors.append({"line": lineno, "message": str(e)})
            continue
        records.append({"period": period, "date": date_val,
                        "numbers": nums, "_auto": period is None})
    return records, errors


# ---------------------------------------------------------------- 数据池
class DataStore:
    def __init__(self, path=POOL_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._data = {"records": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        # 启动即全量校验，坏文件直接暴露
        seen = set()
        for r in self._data["records"]:
            validate_period(r["period"])
            r["numbers"] = validate_numbers(r["numbers"])
            if r["period"] in seen:
                raise ValidationError(f"pool.json中期号重复：{r['period']}")
            seen.add(r["period"])
        self._data["records"].sort(key=lambda r: r["period"])

    def _save_locked(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    # ---- 查询
    def all(self):
        with self._lock:
            return [dict(r) for r in self._data["records"]]

    def latest(self):
        with self._lock:
            return dict(self._data["records"][-1]) if self._data["records"] else None

    def count(self):
        with self._lock:
            return len(self._data["records"])

    def has_period(self, period):
        with self._lock:
            return any(r["period"] == period for r in self._data["records"])

    # ---- 写入
    def add(self, record, on_duplicate="skip"):
        """
        添加单期。record: period/date/numbers/source
        on_duplicate: skip(默认) | replace
        返回 "added" | "replaced" | "skipped"
        """
        period = validate_period(record["period"])
        numbers = validate_numbers(record["numbers"])
        date = validate_date(record["date"]) if record.get("date") else None
        with self._lock:
            rec = {"period": period, "date": date, "numbers": numbers,
                   "source": record.get("source", "manual")}
            for i, r in enumerate(self._data["records"]):
                if r["period"] == period:
                    if on_duplicate == "replace":
                        self._data["records"][i] = rec
                        self._save_locked()
                        return "replaced"
                    return "skipped"
            self._data["records"].append(rec)
            self._data["records"].sort(key=lambda r: r["period"])
            self._save_locked()
            return "added"

    def bulk_add_text(self, text, source="manual", on_duplicate="skip"):
        """
        批量文本录入。无期号行从当前最大期号+1开始自动编号。
        返回 {added, replaced, skipped, errors:[{line,message}], records:[...]}
        """
        parsed, errors = parse_period_text(text)
        added = replaced = skipped = 0
        done = []
        with self._lock:
            next_auto = (self._data["records"][-1]["period"] + 1
                         if self._data["records"] else None)
            for rec in parsed:
                if rec["_auto"]:
                    if next_auto is None:
                        next_auto = 2026001   # 空库兜底
                        errors.append({"line": 0,
                                       "message": "空库且未给期号，已从2026001开始编号"})
                    rec["period"] = next_auto
                    next_auto += 1
                status = self.add(
                    {"period": rec["period"], "date": rec["date"],
                     "numbers": rec["numbers"], "source": source},
                    on_duplicate=on_duplicate)
                if status == "added":
                    added += 1
                elif status == "replaced":
                    replaced += 1
                else:
                    skipped += 1
                done.append({"period": rec["period"], "status": status})
        return {"added": added, "replaced": replaced, "skipped": skipped,
                "errors": errors, "records": done}

    # ---- 删除
    def delete_records(self, periods=None, source=None):
        """
        按期号集合和/或来源删除记录；返回删除条数。
        典型用途：清除来源为 crawl(demo) 的演示数据。
        """
        pset = set(periods or [])
        with self._lock:
            keep, removed = [], 0
            for r in self._data["records"]:
                hit = (r["period"] in pset)
                if source is not None:
                    hit = hit or (r.get("source") == source)
                if hit:
                    removed += 1
                else:
                    keep.append(r)
            if removed:
                self._data["records"] = keep
                self._save_locked()
            return removed

    # ---- 健康度
    def gaps(self):
        """检测期号缺口（相邻期号差>1）"""
        with self._lock:
            recs = self._data["records"]
            out = []
            for a, b in zip(recs, recs[1:]):
                if b["period"] - a["period"] > 1:
                    out.append({"after": a["period"], "before": b["period"],
                                "missing": b["period"] - a["period"] - 1})
            return out

    def staleness(self, now=None):
        """
        返回 {"is_stale": bool, "age_hours": float, ...}
        优先用最后一期日期+1天判断“下一期是否该到”。
        """
        now = now or dt.datetime.now()
        latest = self.latest()
        if not latest:
            return {"is_stale": True, "age_hours": None, "reason": "empty"}
        if latest.get("date"):
            last_dt = dt.datetime.fromisoformat(latest["date"])
            age_hours = (now - last_dt).total_seconds() / 3600
            is_stale = age_hours > STALE_HOURS
        else:
            age_hours, is_stale = None, False
        return {"is_stale": is_stale, "age_hours": round(age_hours, 1) if age_hours is not None else None,
                "latest_period": latest["period"], "latest_date": latest.get("date")}

    def status(self):
        with self._lock:
            latest = self.latest()
            return {"count": self.count(),
                    "latest_period": latest["period"] if latest else None,
                    "latest_date": latest.get("date") if latest else None,
                    "gaps": self.gaps(),
                    "staleness": self.staleness()}
