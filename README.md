# lottery_system —— 80 选 20 彩票数据分析与推荐系统

对"每天从 80 个号码中开出 20 个"的彩票玩法，提供**历史数据管理、确定性统计分析、
组合推荐、回测验证与命中率跟踪**的完整 Web 系统。

核心原则：

- **确定性**：所有算法无随机采样，相同输入永远得到相同输出。
- **口径分离**：真实中奖概率（超几何理论值，对所有组合相同）与"模型相对评分"
  严格区分，前端与接口字段不混淆。
- **可验证**：回测与线上推荐走同一条代码路径，并与随机基准做显著性检验。

> **免责声明**：彩票开奖相互独立，历史数据无法改变任何组合的真实中奖概率
> （任意 4 码全中概率恒为 0.3063%，约 1/326）。本系统是统计分析与工程实践工具，
> 不构成任何投注建议。

---

## 快速开始

```bash
# 1. 安装依赖（Python 3.12+）
pip install -r requirements.txt

# 2. 启动服务
python app.py
# 浏览器打开 http://127.0.0.1:5000

# 3. 运行全部测试
python -m unittest tests.test_analytics tests.test_msize tests.test_robustness \
  tests.test_tracker tests.test_feature_test tests.test_weight_opt \
  tests.test_perf tests.test_optimization_equiv
```

> 提示：跑测试前先关闭正在运行的 Flask 进程，否则 `logs/lottery.log`
> 会被占用导致日志轮转失败（Windows 下 `PermissionError [WinError 32]`）。

---

## 文件说明

### 后端核心（根目录）

| 文件 | 作用 |
|---|---|
| `app.py` | **Web 服务入口**。Flask 应用，提供页面渲染与全部 REST API（数据分析、推荐、回测、抓取、跟踪统计等）；启动时初始化日志、数据池与抓取调度器。 |
| `predictor.py` | **推荐引擎核心**。`Features` 负责特征统计（号码频次 z、近期加权 z、两两/四码共现 z、分区离散、一阶转移 z）；在 12 码候选集内枚举 `C(12,4)=495` 组并按权重排序，输出 `recommend()` 推荐结果与置信度；另含单组合分析 `analyze_combo` 与补充优化 `optimize_combo`。 |
| `analytics.py` | **科学筛选管道（80 → 8 → 4）**。频次/冷热/遗漏/八区特征 → 多因子单码评分 → 取前 8 名枚举 `C(8,4)=70` 组；`AnalysisParams` 可调权重；含 `gap_ablation_analysis`（遗漏回补特征消融对比）。 |
| `msize.py` | **多规模扩展（1/2/3/4 码）**。`size=4` 时完全委派 `predictor`，仅在外层附加 `size`/`theory_m` 字段；1/2/3 码复用同一套特征与候选集，权重口径按 m 分别定义。 |
| `picksize.py` | **选号数量的理论概率分析**。用超几何分布精确计算 m=1~4 时的"全中概率 / 至少中 1 个概率 / 期望命中数"，纯标准库整数组合数，无近似。 |
| `backtest.py` | **历史回测**。严格时间切片：第 i 期只用 `records[:i]` 训练，调用与线上相同的 `recommend(top_k=1)` 预测，再核对命中数；输出命中分布、≥1 命中率、与超几何随机基准的对比及 bootstrap 置信区间。 |
| `tracker.py` | **推荐结果跟踪与自适应**。把每次推荐固化为快照（`data/recommendations.json`），新开奖入库后自动对比；提供命中率报告、按 `(size, last_n, candidate_size)` 的策略统计与最优策略选择、置信度校准（ECE）。 |
| `data_store.py` | **数据池管理**。`data/pool.json` 的线程安全原子读写、期号/日期/号码严格校验、文本批量录入解析、陈旧度（stale）与期号缺口检测。 |
| `crawler.py` | **开奖数据抓取**。支持 mock / url_text / url_json / url_html 四种数据源（配置在 `data/crawler_config.json`）；提供 dry-run 预览、重复期号策略、陈旧自动更新与后台定时调度线程。 |
| `audit_log.py` | **日志与可追溯基础设施**。全局滚动日志 `logs/lottery.log`；`AnalysisLogger` 把每次分析的完整过程（输入快照/参数/中间结果/耗时）落盘为带短哈希的 JSON，便于复核；`PerfTimer` 做性能预算监控。 |
| `requirements.txt` | 运行时依赖声明（仅 `numpy`，其余为标准库）。 |

### 特征检验与权重优化（本轮新增）

| 文件 | 作用 |
|---|---|
| `feature_test.py` | **特征预测力检验（IC）**。逐期滚动计算 `freq_z / recent_z / gap_score / segment_z / z_num_composite` 与下一期开出结果的 Spearman 秩相关，输出 `mean_ic / sd_ic / t_stat / p_value / positive_ratio / verdict`，用于识别噪声特征。 |
| `weight_opt.py` | **评分权重优化**。嵌套 walk-forward 网格搜索：每期只用历史数据选最优权重再预测当期，对比"自适应权重"与默认权重的 top1 命中表现；另提供事后最优权重的参考值。 |

### 前端

| 文件 | 作用 |
|---|---|
| `templates/index.html` | 单页界面：数据池状态、批量录入、抓取、分析、推荐、回测、命中率与策略统计等面板。 |
| `static/app.js` | 前端主交互逻辑：调用各 API、渲染结果表格与状态提示。 |
| `static/charts.js` | 图表绘制（频次分布、命中分布等）。 |
| `static/i18n.js` | 界面中英文文案切换。 |
| `static/style.css` | 页面样式。 |

### 数据

| 文件 | 作用 |
|---|---|
| `data/pool.json` | **开奖数据池**（核心数据）。记录格式：`{period, date, numbers(20个), source}`。 |
| `data/recommendations.json` | **推荐快照**。每次推荐的目标期号、组合、置信度、策略参数，以及后续对比出的命中结果。 |
| `data/crawler_config.json` | 抓取配置：数据源类型、URL、是否自动更新、间隔等。 |
| `data/analysis_logs/` | 每次分析的完整过程留痕（运行产物，已在 `.gitignore` 中忽略）。 |

### 测试

| 文件 | 作用 |
|---|---|
| `tests/test_analytics.py` | 分析管道、参数合法性、回测与真实数据池的测试。 |
| `tests/test_msize.py` | 1/2/3/4 码多规模推荐的正确性与一致性测试。 |
| `tests/test_robustness.py` | 鲁棒性测试：向量化与纯 Python 路径位级一致、异常输入拦截等。 |
| `tests/test_tracker.py` | 快照落盘、开奖对比、命中率统计、策略选择、置信度校准测试。 |
| `tests/test_feature_test.py` | IC 检验模块的秩相关与统计量正确性测试。 |
| `tests/test_weight_opt.py` | 权重网格合法性、walk-forward 输出结构测试。 |
| `tests/test_perf.py` | 性能基准测试（各接口/算法耗时上限）。 |
| `tests/test_optimization_equiv.py` | **新旧实现对拍**：m=1/2/3 逐字段与 legacy 实现比对，确保优化未改变输出。 |
| `tests/perf_benchmark.py` | 性能基准脚本（非单元测试，可单独运行产出耗时报告）。 |
| `tests/__init__.py` | 使 `tests` 成为可导入包。 |

### 文档

| 文件 | 作用 |
|---|---|
| `README.md` | 本文件：项目说明、文件清单、启动与测试方式。 |
| `使用文档.md` | 面向使用者的操作说明（页面功能与接口用法）。 |
| `算法说明.md` | 算法原理与统计口径的详细推导说明。 |
| `算法优化说明.md` | 性能优化与能力增强记录（复杂度分析、优化前后对比、接口兼容性约束）。 |

---

## 算法管线概览

```
数据池 pool.json
      │
      ├─ analytics.py   80码特征 → 多因子评分 → 前8名 → C(8,4)=70 组
      │
      └─ predictor.py   80码特征 → 12码候选集(前16 + 八区保底) → C(12,4)=495 组
                                   ↓
                        加权评分：号码z / 共现z / 四码共现z / 分区离散
                                   ↓
                        recommend() → top-k 组合 + 置信度
                                   ↓
                   backtest.py 严格时间切片验证 + 随机基准对比
                                   ↓
                   tracker.py 快照落盘 → 开奖后对比 → 命中率/策略统计
```

- **随机基准**：命中数服从 `Hypergeometric(80, 20, 4)`，平均命中 `1.0`。
  任何筛选若跑不赢基准，即不具备实证有效性。
- **已实证的结论**：IC 检验显示现有号码级特征均无统计显著预测力
  （|t| < 1.96），其中 `gap_score` 方向为负；权重网格优化与默认权重结果一致，
  说明**候选集规模而非评分权重**才是真正的杠杆。

---

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 页面 |
| GET | `/api/status` | 数据池状态 + 抓取配置 |
| POST | `/api/analyze` | 指定 4 码组合的概率/特征分析 |
| POST | `/api/optimize` | 自动补充优化（给定 1~10 码，补足到 4 码） |
| POST | `/api/pool/manual` | 手工批量录入开奖数据 |
| POST | `/api/crawl` | 立即抓取 / 陈旧时更新 |
| POST | `/api/crawler/config` | 更新抓取配置 |
| POST | `/api/recommend` | N+1 日推荐（支持 `last_n`、`top_k`、`auto_strategy`、`candidate_size`） |
| GET | `/api/analytics/full` | 完整科学筛选分析 |
| POST | `/api/backtest` | 4 码回测 |
| GET | `/api/picksize/theory` | 1~4 码理论概率 |
| POST | `/api/analyze/m` | 多规模（m=1/2/3/4）单组分析 |
| POST | `/api/analytics/m` | 多规模科学筛选分析 |
| POST | `/api/recommend/m` | 多规模推荐 |
| POST | `/api/backtest/m` | 多规模回测 |
| GET | `/api/analytics/logs` | 分析留痕列表 |
| GET | `/api/analytics/logs/<filename>` | 单条留痕详情 |
| GET | `/api/recommendations` | 推荐快照列表 |
| GET | `/api/recommendations/<rid>` | 单条快照详情 |
| POST | `/api/recommendations/reconcile` | 手动触发开奖对比 |
| GET | `/api/hitrate` | 命中率聚合报告（含随机基准 z 检验） |
| GET | `/api/confidence/calibrate` | 置信度校准（分箱 + ECE） |
| GET | `/api/strategy/stats` | 策略（窗口/候选集）统计 |
| GET | `/api/feature/ic` | 特征预测力检验（IC） |
| POST | `/api/weights/optimize` | 权重 walk-forward 优化 |
| GET | `/api/gap/ablation` | gap 特征消融对比 |

---

## 目录结构

```
lottery_system/
├── app.py                  # Web 入口与 API
├── predictor.py            # 推荐引擎核心
├── analytics.py            # 科学筛选管道
├── msize.py                # 多规模推荐
├── picksize.py             # 理论概率
├── backtest.py             # 回测
├── tracker.py              # 推荐跟踪与统计
├── data_store.py           # 数据池管理
├── crawler.py              # 数据抓取
├── audit_log.py            # 日志与留痕
├── feature_test.py         # 特征 IC 检验
├── weight_opt.py           # 权重优化
├── requirements.txt
├── templates/index.html
├── static/{app.js, charts.js, i18n.js, style.css}
├── data/{pool.json, recommendations.json, crawler_config.json}
├── tests/                  # 单元测试与性能基准
├── logs/                   # 运行日志（不入库）
└── *.md                    # 项目/使用/算法/优化文档
```
