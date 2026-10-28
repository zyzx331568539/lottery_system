# -*- coding: utf-8 -*-
"""
app.py —— 80码 / 4码组合分析与推荐系统（Flask Web 服务）

启动：
    cd lottery_system
    python app.py
    浏览器打开 http://127.0.0.1:5000

路由：
    GET  /                     页面
    GET  /api/status           数据池状态 + 抓取配置
    POST /api/analyze          4码概率分析      {"numbers":[..]}
    POST /api/optimize         自动补充优化     {"numbers":[1-10个]}
    POST /api/pool/manual      手工批量录入     {"text":"...", "on_duplicate":"skip"}
    POST /api/crawl            立即抓取/陈旧更新 {"force":true|false}
    POST /api/recommend        N+1日推荐        {"last_n": 30}
    POST /api/crawler/config   更新抓取配置     {"source_type":..., "url":..., "auto_update":...}
"""
import os
import sys

from flask import Flask, jsonify, render_template, request

from crawler import (CrawlError, Scheduler, load_config, save_config,
                     update_if_stale)
from data_store import DataStore, ValidationError
from predictor import analyze_combo, optimize_combo, recommend
from analytics import run_analysis
from backtest import run_backtest
from msize import (analyze_combo_m, recommend_m, run_analysis_m,
                   run_backtest_m, theory_table)
from audit_log import (list_analysis_logs, read_analysis_log,
                       setup_logging, get_logger)
import tracker

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# 系统日志（抓取/校验/分析/回测/错误全留痕）
setup_logging()
log = get_logger()

store = DataStore()
scheduler = Scheduler(store)
scheduler.start()
# 启动时确保推荐快照文件存在，并把历史快照与已有开奖对齐（幂等，已对比的跳过）
tracker.ensure_store()
tracker.reconcile(store)
log.info("数据池加载完成：%d 期（最新 %s）",
         store.count(),
         store.latest()["period"] if store.latest() else None)


# ---------------------------------------------------------------- 错误处理
@app.errorhandler(ValidationError)
def _handle_validation(e):
    return jsonify({"ok": False, "error_type": "validation", "error": str(e)}), 400


@app.errorhandler(CrawlError)
def _handle_crawl(e):
    return jsonify({"ok": False, "error_type": "crawl", "error": str(e)}), 502


@app.errorhandler(404)
def _handle_404(e):
    return jsonify({"ok": False, "error": "not found"}), 404


def _payload():
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        raise ValidationError("请求体必须是JSON对象")
    return data


@app.after_request
def _no_cache(resp):
    """开发模式：页面与静态资源始终校验最新版本，避免浏览器缓存导致脚本/页面版本错配"""
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


# ---------------------------------------------------------------- 页面
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------- 状态
@app.route("/api/status", methods=["GET"])
def api_status():
    cfg = load_config()
    return jsonify({"ok": True, "pool": store.status(),
                    "crawler": {"source_type": cfg["source_type"],
                                "url": cfg.get("url", ""),
                                "html_profile": cfg.get("html_profile", "ydniu_kl8"),
                                "on_duplicate": cfg.get("on_duplicate", "skip"),
                                "auto_update": cfg.get("auto_update", False),
                                "last_result": scheduler.last_result}})


# ---------------------------------------------------------------- 概率分析
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    numbers = _payload().get("numbers")
    return jsonify({"ok": True, "result": analyze_combo(numbers, store.all())})


# ---------------------------------------------------------------- 补充优化
@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    numbers = _payload().get("numbers")
    return jsonify({"ok": True, "result": optimize_combo(numbers, store.all())})


# ---------------------------------------------------------------- 手工录入
@app.route("/api/pool/manual", methods=["POST"])
def api_manual():
    data = _payload()
    text = data.get("text", "")
    if not text.strip():
        raise ValidationError("录入内容为空")
    mode = data.get("on_duplicate", "skip")
    if mode not in ("skip", "replace"):
        raise ValidationError("on_duplicate 仅支持 skip / replace")
    result = store.bulk_add_text(text, source="manual",
                                 on_duplicate=mode)
    log.info("手工录入完成：added=%d replaced=%d skipped=%d errors=%d",
             result["added"], result["replaced"], result["skipped"],
             len(result["errors"]))
    if result["added"]:
        rec = tracker.reconcile(store)
        log.info("开奖对比完成：compared=%d pending=%d",
                 rec["compared"], rec["still_pending"])
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------- 抓取
@app.route("/api/crawl", methods=["POST"])
def api_crawl():
    data = _payload()
    force = bool(data.get("force", False))
    dry_run = bool(data.get("dry_run", False))
    on_duplicate = data.get("on_duplicate")
    if on_duplicate is not None and on_duplicate not in ("skip", "replace"):
        raise ValidationError("on_duplicate 仅支持 skip / replace")

    # 可选的“临时覆盖”：用界面当前选择试跑，不必先保存配置
    override_keys = ("source_type", "url", "html_profile")
    cfg = None
    if any(k in data for k in override_keys):
        cfg = load_config()
        for k in override_keys:
            if k in data:
                cfg[k] = data[k]
        if cfg["source_type"] not in ("mock", "url_text", "url_json", "url_html"):
            raise ValidationError("source_type 仅支持 mock/url_text/url_json/url_html")
        if cfg["source_type"] == "url_html" and cfg["html_profile"] not in ("ydniu_kl8",):
            raise ValidationError("html_profile 仅支持 ydniu_kl8")

    crawl_result = update_if_stale(store, cfg=cfg, force=force,
                                   dry_run=dry_run,
                                   on_duplicate=on_duplicate)
    if crawl_result.get("ran"):
        if dry_run:
            log.info("测试抓取完成：parsed=%s exists=%s",
                     crawl_result.get("parsed"), crawl_result.get("exists_count"))
        else:
            log.info("抓取完成：added=%s replaced=%s skipped=%s",
                     crawl_result.get("added"), crawl_result.get("replaced"),
                     crawl_result.get("skipped"))
            # 新开奖入库后，把待对比的推荐快照自动对齐
            if crawl_result.get("added"):
                rec = tracker.reconcile(store)
                log.info("开奖对比完成：compared=%d pending=%d",
                         rec["compared"], rec["still_pending"])
    return jsonify({"ok": True, "result": crawl_result})


# ---------------------------------------------------------------- 推荐
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = _payload()
    last_n = data.get("last_n")
    if last_n is not None:
        if not isinstance(last_n, int) or last_n < 5:
            raise ValidationError("last_n 必须是 ≥5 的整数（或省略表示全部）")
    top_k = data.get("top_k", 5)
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValidationError("top_k 取值1-10")
    auto_strategy = bool(data.get("auto_strategy", False))
    records = store.all()
    result = recommend(records, last_n, top_k, auto_strategy=auto_strategy)
    # 落盘快照，目标期号 = 池最新期号 + 1；后续开奖入库自动对比
    tracker.save_recommendation(records, 4, result,
                                last_n=last_n, top_k=top_k)
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------- 抓取配置
@app.route("/api/crawler/config", methods=["POST"])
def api_crawler_config():
    data = _payload()
    cfg = load_config()
    for key in ("source_type", "url", "auto_update", "check_interval_seconds",
                "html_profile", "on_duplicate"):
        if key in data:
            cfg[key] = data[key]
    if cfg["source_type"] not in ("mock", "url_text", "url_json", "url_html"):
        raise ValidationError("source_type 仅支持 mock/url_text/url_json/url_html")
    if cfg["html_profile"] not in ("ydniu_kl8",):
        raise ValidationError("html_profile 仅支持 ydniu_kl8")
    if cfg["on_duplicate"] not in ("skip", "replace"):
        raise ValidationError("on_duplicate 仅支持 skip / replace")
    save_config(cfg)
    log.info("爬虫配置已更新：source_type=%s", cfg["source_type"])
    return jsonify({"ok": True, "config": cfg})


# ---------------------------------------------------------------- 核心筛选（80→4）
@app.route("/api/analytics/full", methods=["GET"])
def api_analytics_full():
    """
    完整科学筛选报告：冷热识别/号段分布/多因子评分/Wilson区间/最终4码。
    每次调用自动归档完整分析过程（data/analysis_logs/）。
    """
    try:
        result = run_analysis(store.all())
    except ValueError as e:
        raise ValidationError(str(e))
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------- 历史回测
@app.route("/api/backtest", methods=["POST"])
def api_backtest_run():
    data = _payload()
    tw = data.get("train_window")
    if tw is not None:
        if isinstance(tw, bool) or not isinstance(tw, int) or tw < 5:
            raise ValidationError("train_window 必须是 >=5 的整数（或省略）")
    try:
        result = run_backtest(store.all(), train_window=tw)
    except ValueError as e:
        raise ValidationError(str(e))
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------- 多规模扩展（1/2/3/4码）
# 说明：以下为纯增量端点；size=4 时 msize 内部委派原有函数，4 码口径不变。
def _parse_size(data):
    size = data.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size not in (1, 2, 3, 4):
        raise ValidationError("size 必须是 1/2/3/4 的整数")
    return size


@app.route("/api/picksize/theory", methods=["GET"])
def api_picksize_theory():
    """m=1..4 理论命中概率对比（超几何精确值，与选号策略无关）"""
    return jsonify({"ok": True, "result": theory_table()})


@app.route("/api/analyze/m", methods=["POST"])
def api_analyze_m():
    """多规模概率分析：{"numbers":[..]} 号码个数 1-4 即为规模"""
    numbers = _payload().get("numbers")
    return jsonify({"ok": True, "result": analyze_combo_m(numbers, store.all())})


@app.route("/api/analytics/m", methods=["POST"])
def api_analytics_m():
    """多规模科学筛选：{"size":1-4}，与4码同一管道（80→前8→枚举组合）"""
    size = _parse_size(_payload())
    try:
        result = run_analysis_m(store.all(), size)
    except ValueError as e:
        raise ValidationError(str(e))
    return jsonify({"ok": True, "result": result})


@app.route("/api/recommend/m", methods=["POST"])
def api_recommend_m():
    """多规模 N+1 日推荐：{"size":1-4, "last_n":可选, "top_k":1-10, "auto_strategy":bool}"""
    data = _payload()
    size = _parse_size(data)
    last_n = data.get("last_n")
    if last_n is not None:
        if not isinstance(last_n, int) or last_n < 5:
            raise ValidationError("last_n 必须是 ≥5 的整数（或省略表示全部）")
    top_k = data.get("top_k", 5)
    if not isinstance(top_k, int) or not 1 <= top_k <= 10:
        raise ValidationError("top_k 取值1-10")
    auto_strategy = bool(data.get("auto_strategy", False))
    records = store.all()
    result = recommend_m(records, size, last_n, top_k,
                         auto_strategy=auto_strategy)
    tracker.save_recommendation(records, size, result,
                                last_n=last_n, top_k=top_k)
    return jsonify({"ok": True, "result": result})


@app.route("/api/backtest/m", methods=["POST"])
def api_backtest_m():
    """多规模历史回测：{"size":1-4, "train_window":可选>=5}"""
    data = _payload()
    size = _parse_size(data)
    tw = data.get("train_window")
    if tw is not None:
        if isinstance(tw, bool) or not isinstance(tw, int) or tw < 5:
            raise ValidationError("train_window 必须是 >=5 的整数（或省略）")
    try:
        result = run_backtest_m(store.all(), size, train_window=tw)
    except ValueError as e:
        raise ValidationError(str(e))
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------- 分析过程日志
@app.route("/api/analytics/logs", methods=["GET"])
def api_analytics_logs():
    return jsonify({"ok": True, "logs": list_analysis_logs(30)})


@app.route("/api/analytics/logs/<path:filename>", methods=["GET"])
def api_analytics_log_detail(filename):
    try:
        doc = read_analysis_log(filename)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "日志不存在"}), 404
    return jsonify({"ok": True, "log": doc})


# ---------------------------------------------------------------- 推荐命中追踪
@app.route("/api/recommendations", methods=["GET"])
def api_recommendations():
    """推荐快照列表。查询参数：size(1-4)、compared(true/false/省略)、limit。"""
    size = request.args.get("size", type=int)
    if size is not None and size not in (1, 2, 3, 4):
        raise ValidationError("size 必须是 1/2/3/4")
    cmp_raw = request.args.get("compared")
    compared = None
    if cmp_raw is not None:
        compared = cmp_raw.lower() == "true"
    limit = request.args.get("limit", type=int)
    recs = tracker.list_recommendations(size=size, compared=compared,
                                        limit=limit)
    return jsonify({"ok": True, "recommendations": recs,
                    "count": len(recs)})


@app.route("/api/recommendations/<rid>", methods=["GET"])
def api_recommendation_detail(rid):
    rec = tracker.get_recommendation(rid)
    if rec is None:
        return jsonify({"ok": False, "error": "推荐记录不存在"}), 404
    return jsonify({"ok": True, "recommendation": rec})


@app.route("/api/recommendations/reconcile", methods=["POST"])
def api_reconcile():
    """手动触发一次开奖对比（幂等）。"""
    return jsonify({"ok": True, "result": tracker.reconcile(store)})


@app.route("/api/hitrate", methods=["GET"])
def api_hitrate():
    """命中率聚合报告。查询参数：size、limit（最近N条）。"""
    size = request.args.get("size", type=int)
    if size is not None and size not in (1, 2, 3, 4):
        raise ValidationError("size 必须是 1/2/3/4")
    limit = request.args.get("limit", type=int)
    return jsonify({"ok": True, "report": tracker.hit_rate_report(
        size=size, limit=limit)})


@app.route("/api/strategy/stats", methods=["GET"])
def api_strategy_stats():
    """按 (size, last_n) 统计的历史命中表 + 推荐策略。"""
    size = request.args.get("size", type=int)
    if size is not None and size not in (1, 2, 3, 4):
        raise ValidationError("size 必须是 1/2/3/4")
    stats = tracker.strategy_stats(size=size)
    suggested = {}
    for sz in (size,) if size else (1, 2, 3, 4):
        suggested[sz] = tracker.best_strategy(sz)
    return jsonify({"ok": True, "stats": stats, "suggested": suggested})


if __name__ == "__main__":
    print("Lottery 4-digit system running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
