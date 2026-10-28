# -*- coding: utf-8 -*-
"""
audit_log.py —— 算法透明度与系统监控基础设施

三件事：
1. setup_logging()：全局日志（logs/lottery.log，按大小滚动 + 控制台），
   抓取/校验/分析/回测/错误堆栈全部留痕。
2. AnalysisLogger：把"每次分析的完整过程数据"写为
   data/analysis_logs/YYYYMMDD_HHMMSS_<8hex>.json（数据快照/参数/中间结果/最终输出/耗时），
   文件名携带短哈希用于防抵赖引用；写入为原子替换。
3. PerfTimer / PERF_BUDGET：性能基准计时，超预算自动 WARNING。
"""
import hashlib
import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
ANALYSIS_DIR = os.path.join(HERE, "data", "analysis_logs")

# ------------------------------------------------------------ 性能基准（毫秒）
PERF_BUDGET = {
    "crawl": 15_000,        # 每日抓取+解析+校验（含网络）
    "analytics": 2_000,     # 完整分析管道（80→8→4 + 报告结构）
    "backtest": 10_000,     # 全量滑动回测
}

_LOGGER_NAME = "lottery"
_logging_ready = False
_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 1. 系统日志
# ---------------------------------------------------------------------------
def setup_logging(level=logging.INFO) -> logging.Logger:
    """幂等初始化全局 logger"""
    global _logging_ready
    with _log_lock:
        logger = logging.getLogger(_LOGGER_NAME)
        if _logging_ready:
            return logger
        os.makedirs(LOG_DIR, exist_ok=True)
        logger.setLevel(level)
        logger.propagate = False
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        fh = RotatingFileHandler(os.path.join(LOG_DIR, "lottery.log"),
                                 maxBytes=2_000_000, backupCount=5,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        _logging_ready = True
        logger.info("日志系统就绪，日志目录：%s", LOG_DIR)
        return logger


def get_logger() -> logging.Logger:
    if not _logging_ready:
        return setup_logging()
    return logging.getLogger(_LOGGER_NAME)


# ---------------------------------------------------------------------------
# 2. 性能计时
# ---------------------------------------------------------------------------
class PerfTimer:
    """
    用法：
        with PerfTimer("analytics", extra={"periods": n}) as t:
            ...
        t.elapsed_ms
    超预算自动 WARNING；异常时记录 ERROR 并继续抛出。
    """

    def __init__(self, stage: str, budget_ms: int | None = None,
                 extra: dict | None = None, logger: logging.Logger | None = None):
        self.stage = stage
        self.budget_ms = budget_ms if budget_ms is not None else PERF_BUDGET.get(stage)
        self.extra = extra or {}
        self._logger = logger or get_logger()
        self.elapsed_ms = 0.0
        self.within_budget = True

    def __enter__(self) -> "PerfTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if exc_type is None:
            if self.budget_ms is not None and self.elapsed_ms > self.budget_ms:
                self.within_budget = False
                self._logger.warning(
                    "性能超基准：%s 用时 %.1fms > 预算 %dms %s",
                    self.stage, self.elapsed_ms, self.budget_ms,
                    ("%r" % self.extra) if self.extra else "")
            else:
                self._logger.info("阶段完成：%s 用时 %.1fms %s", self.stage,
                                  self.elapsed_ms,
                                  ("%r" % self.extra) if self.extra else "")
        else:
            self._logger.error("阶段异常：%s 用时 %.1fms：%s",
                               self.stage, self.elapsed_ms, exc)
        return False  # 不吞异常


# ---------------------------------------------------------------------------
# 3. 分析过程归档
# ---------------------------------------------------------------------------
class AnalysisLogger:
    """
    归档一次分析的完整过程：
      logger = AnalysisLogger("daily_pick")
      logger.set_snapshot(...); logger.set_params(...); logger.set_intermediate(...)
      path = logger.finish(final_result, elapsed_ms, within_budget)
    """

    def __init__(self, analysis_type: str = "daily_pick",
                 logger: logging.Logger | None = None):
        self.type = analysis_type
        self._logger = logger or get_logger()
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._doc = {
            "analysis_type": analysis_type,
            "started_at": self.started_at,
            "data_snapshot": None,
            "params": None,
            "intermediate": None,
            "final": None,
            "perf": None,
        }

    def set_snapshot(self, snapshot: dict) -> None:
        self._doc["data_snapshot"] = snapshot

    def set_params(self, params: dict) -> None:
        self._doc["params"] = params

    def set_intermediate(self, intermediate: dict) -> None:
        self._doc["intermediate"] = intermediate

    @staticmethod
    def snapshot_from_records(records) -> dict:
        """从记录列表生成标准化数据快照（不复制全部号码，记录范围+逐期摘要）"""
        if not records:
            return {"count": 0}
        return {
            "count": len(records),
            "first_period": records[0]["period"],
            "last_period": records[-1]["period"],
            "first_date": records[0].get("date"),
            "last_date": records[-1].get("date"),
            "periods": [r["period"] for r in records],
            "sources": sorted({r.get("source", "unknown") for r in records}),
        }

    def finish(self, final: dict, elapsed_ms: float,
               within_budget: bool = True) -> str:
        """写入归档文件，返回绝对路径"""
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        self._doc["final"] = final
        self._doc["perf"] = {
            "elapsed_ms": round(elapsed_ms, 2),
            "budget_ms": PERF_BUDGET.get("analytics"),
            "within_budget": within_budget,
        }
        finished_at = datetime.now()
        self._doc["finished_at"] = finished_at.isoformat(timespec="seconds")

        blob = json.dumps(self._doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        self._doc["integrity_sha256"] = digest

        fname = "%s_%s.json" % (finished_at.strftime("%Y%m%d_%H%M%S"),
                                digest[:8])
        path = os.path.join(ANALYSIS_DIR, fname)
        tmp = path + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        self._logger.info("分析过程已归档：%s（%s，%.1fms）",
                          fname, self.type, elapsed_ms)
        return path


def list_analysis_logs(limit: int = 30) -> list:
    """返回最近分析归档的摘要列表"""
    if not os.path.isdir(ANALYSIS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(ANALYSIS_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(ANALYSIS_DIR, name)
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        snap = doc.get("data_snapshot") or {}
        final = doc.get("final") or {}
        out.append({
            "file": name,
            "analysis_type": doc.get("analysis_type"),
            "finished_at": doc.get("finished_at"),
            "periods_count": snap.get("count"),
            "first_period": snap.get("first_period"),
            "last_period": snap.get("last_period"),
            "final_numbers": final.get("numbers"),
            "elapsed_ms": (doc.get("perf") or {}).get("elapsed_ms"),
            "integrity_sha256": doc.get("integrity_sha256"),
        })
        if len(out) >= limit:
            break
    return out


def read_analysis_log(filename: str) -> dict:
    """按文件名读取单个归档（做路径穿越防护）"""
    base = os.path.basename(filename)
    path = os.path.join(ANALYSIS_DIR, base)
    if not os.path.isfile(path):
        raise FileNotFoundError(base)
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)
