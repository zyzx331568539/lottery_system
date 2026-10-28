/* app.js —— 前端交互（原生 fetch，无第三方依赖；支持中英文切换） */
const $ = id => document.getElementById(id);

/* 缓存最近一次结果，切换语言时按缓存重新渲染，避免半中半英 */
const cache = {};

async function api(url, body){
  const opt = body ? {method:"POST", headers:{"Content-Type":"application/json"},
                      body: JSON.stringify(body)} : {};
  const res = await fetch(url, opt);
  const data = await res.json().catch(() => ({ok:false, error:"Bad server response"}));
  if(!res.ok || data.ok === false){
    throw new Error(data.error || ("HTTP " + res.status));
  }
  return data.result ?? data;
}

function parseCombo(text, exact){
  const nums = (text.trim().match(/\d+/g) || []).map(Number);
  if(exact !== undefined && nums.length !== exact)
    throw new Error(t("err_need", {n:exact, m:nums.length}));
  if(nums.some(n => n < 1 || n > 80)) throw new Error(t("err_range"));
  if(new Set(nums).size !== nums.length) throw new Error(t("err_unique"));
  return nums;
}

function balls(arr){
  return `<div class="balls">${arr.map(n =>
    `<span class="ball">${n}</span>`).join("")}</div>`;
}

function msg(kind, text){ return `<div class="note ${kind}">${text}</div>`; }

/* 轻量浮动提示：在页面顶部闪现一条消息，3 秒后自动消失 */
let _flashTimer = null;
function flash(text, isError){
  let el = document.getElementById("flashToast");
  if(!el){
    el = document.createElement("div");
    el.id = "flashToast";
    el.className = "flash-toast";
    document.body.appendChild(el);
  }
  el.textContent = text;
  el.classList.toggle("err", !!isError);
  el.classList.add("show");
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(()=>el.classList.remove("show"), 2600);
}

/* ------------------------------------------------------------ status */
function renderStatus(s){
  const p = s.pool, c = s.crawler;
  const tags = [];
  tags.push(p.staleness.is_stale
    ? `<span class="badge stale">${t("badge_stale", {h:p.staleness.age_hours ?? "?"})}</span>`
    : `<span class="badge ok">${t("badge_fresh")}</span>`);
  if(p.gaps.length) tags.push(`<span class="badge gap">${t("badge_gaps", {n:p.gaps.length})}</span>`);
  $("status").innerHTML =
    `<div><b>${p.count}</b> ${t("status_periods")} <b>${p.latest_period ?? "-"}</b>
     (${p.latest_date ?? "-"})</div>
     <div style="margin-top:4px">${t("status_source")}: ${c.source_type} ${tags.join(" ")}</div>`;
  $("srcType").value = c.source_type;
  $("srcUrl").value = c.url || "";
  $("autoUpdate").checked = !!c.auto_update;
  $("crawlReplace").checked = c.on_duplicate === "replace";
}

async function refreshStatus(){
  try{
    const s = await api("/api/status");
    cache.status = s;
    renderStatus(s);
  }catch(e){ $("status").textContent = t("status_error") + ": " + e.message; }
}

/* ------------------------------------------------------------ analyze */
function renderAnalyze(r){
  const th = r.theory, e = r.empirical;
  $("analyzeOut").innerHTML =
    `<div class="rec">${balls(r.combo)}</div>
     <dl class="kv">
       <dt>${t("exact_prob")}</dt><dd>${th.percent.toFixed(4)}% (1/${th.odds_1_in.toFixed(0)})</dd>
       <dt>${t("observed_co")}</dt><dd>${e.co_occurrence} ${t("times_word")}
           (${t("expected_word")} ${e.co_expected}) — z ${e.score_parts.four_z}</dd>
       <dt>${t("emp_score")}</dt><dd>${e.score}
           (${e.percentile_among_observed}% · ${t("pct_among", {n:e.observed_basis})})</dd>
       <dt>${t("hit_history")}</dt><dd>${e.hit_periods.length ? e.hit_periods.join(", ") : t("never")}</dd>
     </dl>
     <table class="table"><tr>
       <th>${t("th_num")}</th><th>${t("th_count")}</th>
       <th>${t("th_expected")}</th><th>${t("th_z")}</th></tr>
       ${r.numbers.map(n =>
         `<tr><td>${n.number}</td><td>${n.count}</td><td>${n.expected}</td><td>${n.z}</td></tr>`).join("")}
     </table>
     <div class="note warn">${t("disclaimer")}</div>`;
}

$("btnAnalyze").onclick = async () => {
  const out = $("analyzeOut");
  try{
    const nums = parseCombo($("comboInput").value, 4);
    out.innerHTML = msg("", t("msg_analyzing"));
    const r = await api("/api/analyze", {numbers: nums});
    cache.analyze = r;
    renderAnalyze(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ optimize */
function renderOptimize(r){
  $("optimizeOut").innerHTML =
    `<div class="wheel">${r.wheel.map(n => `<span class="ball">${n}</span>`).join("")}</div>
     <p class="small">${t("companions_added")}: ${r.companions.join(", ") || t("none_word")}</p>
     <table class="table"><tr><th>${t("best_with_input")}</th><th>${t("score_word")}</th></tr>
       ${r.best_with_input.map(x =>
         `<tr><td>${x.combo.join(" ")}</td><td>${x.score}</td></tr>`).join("")}</table>
     <table class="table"><tr><th>${t("best_in_wheel")}</th><th>${t("score_word")}</th></tr>
       ${r.best_global.map(x =>
         `<tr><td>${x.combo.join(" ")}</td><td>${x.score}</td></tr>`).join("")}</table>
     <div class="note warn">${t("optimize_note")}</div>`;
}

$("btnOptimize").onclick = async () => {
  const out = $("optimizeOut");
  try{
    const nums = parseCombo($("comboInput").value);   // 1-10 allowed
    if(nums.length > 10) throw new Error(t("err_max10"));
    out.innerHTML = msg("", t("msg_optimizing"));
    const r = await api("/api/optimize", {numbers: nums});
    cache.optimize = r;
    renderOptimize(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ recommend */
function renderRecommend(r){
  $("recommendOut").innerHTML =
    `<p class="small">${t("basis_word")}: ${r.basis.periods_used} ${t("periods_word")}
      (${r.basis.first_period} → ${r.basis.last_period}) · ${t("candidate_pool_word")}
      ${r.candidate_pool.join(" ")}</p>` +
    r.recommendations.map(rec =>
      `<div class="rec">
         ${balls(rec.combo)}
         <div class="conf">
           <div class="track"><div class="fill ${rec.confidence_label}"
             style="width:${(rec.confidence*100).toFixed(0)}%"></div></div>
           <div class="meta">${t("conf_score")} ${rec.score} · ${t("conf_stability")} ${rec.stability}
             · ${t("conf_margin")} ${rec.margin} · ${t("conf_parts")} ${JSON.stringify(rec.score_parts)}</div>
         </div>
         <span class="label ${rec.confidence_label}">${t("label_" + rec.confidence_label)}
           ${(rec.confidence*100).toFixed(0)}%</span>
       </div>`).join("") +
    `<details class="small" style="margin-top:8px"><summary>${t("methodology")}</summary>
       <p>${t("recommend_method")}</p><p class="note warn">${t("disclaimer")}</p></details>`;
}

$("btnRecommend").onclick = async () => {
  const out = $("recommendOut");
  try{
    const body = {top_k: Number($("topK").value) || 5};
    const n = Number($("lastN").value);
    if(Number.isInteger(n) && n > 0) body.last_n = n;
    body.auto_strategy = $("autoStrategy").checked;
    out.innerHTML = msg("", t("msg_generating"));
    const r = await api("/api/recommend", body);
    cache.recommend = r;
    renderRecommend(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ manual */
function renderManual(r){
  const errs = r.errors.length
    ? r.errors.map(x => `<div>${t("line_word", {n:x.line})}: ${x.message}</div>`).join("") : "";
  $("manualOut").innerHTML = msg(r.errors.length ? "warn" : "",
    `${t("added")} ${r.added} · ${t("replaced")} ${r.replaced} · ${t("skipped")} ${r.skipped}`)
    + (errs ? `<div class="note err">${errs}</div>` : "");
}

$("btnManual").onclick = async () => {
  const out = $("manualOut");
  try{
    const r = await api("/api/pool/manual", {
      text: $("manualText").value,
      on_duplicate: $("replaceDup").checked ? "replace" : "skip"});
    cache.manual = r;
    renderManual(r);
    refreshStatus();
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ crawler */
function renderCrawlFresh(r){
  $("crawlOut").innerHTML = msg("",
    t("fresh_skip", {p:r.staleness.latest_period, h:r.staleness.age_hours ?? "?"}));
}

function renderCrawlDone(r){
  $("crawlOut").innerHTML = msg(r.demo ? "warn" : "",
    `${r.demo ? t("demo_warn") + " · " : ""}${t("added")} ${r.added} · ${t("replaced")} ${r.replaced ?? 0}
     · ${t("skipped")} ${r.skipped} · ${t("crawl_periods")} ${(r.periods||[]).join(", ")}`)
    + ((r.errors||[]).length
      ? `<div class="note err">${r.errors.map(x => JSON.stringify(x)).join("<br>")}</div>` : "");
}

/* dry-run：只预览，不写入 */
function renderDryRun(r){
  const head = `<div class="note">${t("dr_head")}</div>
    <p class="small">${t("dr_parsed", {n:r.parsed})} · ${t("dr_new", {n:r.new_count})}
      · ${t("dr_exists", {n:r.exists_count})}</p>`;
  const rows = (r.records || []).map(x =>
    `<tr><td>${x.period}</td><td>${x.date ?? "-"}</td>
      <td style="font-size:12px">${x.numbers.map(n=>String(n).padStart(2,"0")).join(" ")}</td>
      <td><span class="badge ${x.exists ? "stale" : "ok"}">${x.exists ? t("tag_exists") : t("tag_new")}</span></td>
    </tr>`).join("");
  const table = `<table class="table"><tr><th>${t("th_num")}</th><th>${t("date_word")}</th>
      <th>${t("numbers_word")}</th><th></th></tr>${rows}</table>`;
  const errs = (r.errors||[]).length
    ? `<div class="note err">${t("dr_errors")}: ${r.errors.map(x=>JSON.stringify(x)).join("<br>")}</div>` : "";
  $("crawlOut").innerHTML = head + table + errs;
}

$("btnCrawl").onclick = async () => {
  const out = $("crawlOut");
  try{
    out.innerHTML = msg("", t("msg_crawling"));
    const r = await api("/api/crawl", {
      force: $("forceCrawl").checked,
      on_duplicate: $("crawlReplace").checked ? "replace" : "skip"});
    cache.crawl = r;
    if(!r.ran){ renderCrawlFresh(r); return; }
    renderCrawlDone(r);
    refreshStatus();
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

$("btnTestCrawl").onclick = async () => {
  const out = $("crawlOut");
  try{
    out.innerHTML = msg("", t("msg_crawling"));
    // 测试抓取：直接使用界面当前选择（无需先保存配置），不受陈旧状态限制
    const r = await api("/api/crawl", {
      force: true, dry_run: true,
      source_type: $("srcType").value,
      url: $("srcUrl").value.trim(),
      on_duplicate: $("crawlReplace").checked ? "replace" : "skip"});
    cache.crawl = r;
    renderDryRun(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

$("btnSaveCfg").onclick = async () => {
  try{
    await api("/api/crawler/config", {
      source_type: $("srcType").value, url: $("srcUrl").value.trim(),
      on_duplicate: $("crawlReplace").checked ? "replace" : "skip",
      auto_update: $("autoUpdate").checked});
    $("crawlOut").innerHTML = msg("", t("config_saved"));
    refreshStatus();
  }catch(e){ $("crawlOut").innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ core scientific pick */
function renderFull(r){
  $("fullOut").innerHTML =
    `<div class="final4">
       ${r.numbers.map(n => `<span class="ball gold">${n}</span>`).join("")}
       <span class="small">${t("combo_score_word")} ${r.combo_score}
         · ${JSON.stringify(r.combo_score_parts)}</span>
     </div>
     <table class="table wrap-table">
       <tr><th>${t("th_num")}</th><th>${t("th_rank")}</th><th>${t("th_total")}</th>
         <th>${t("th_full_z")}</th><th>${t("th_recent_z")}</th><th>${t("th_gap_s")}</th>
         <th>${t("th_seg_s")}</th><th>${t("th_model")}</th><th>${t("th_freq")}</th>
         <th>${t("th_wilson")}</th><th>${t("th_hotcold")}</th></tr>
       ${r.ranking_basis.map(x => `<tr>
          <td><b>${x.number}</b></td><td>${x.rank_of_80}</td><td>${x.total_score}</td>
          <td>${x.factors.full_freq_z}</td><td>${x.factors.recent_z}</td>
          <td>${x.factors.gap_score}</td><td>${x.factors.segment_score}</td>
          <td>${x.model_index_pct}%</td><td>${x.empirical_freq_pct}%</td>
          <td>${x.wilson95_ci_pct[0]}–${x.wilson95_ci_pct[1]}%</td>
          <td><span class="mini-lab rc-${x.recent_label}">${t("lab_r_" + x.recent_label)}</span>
              <span class="mini-lab gl-${x.gap_label}">${t("lab_g_" + x.gap_label)}</span></td>
        </tr>`).join("")}
     </table>
     <div class="charts-grid">
       <div class="chart-box"><h4 data-i18n="chart_hotcold">80码冷热分布（红=热 蓝=冷 蓝框=遗漏冷）</h4>
         <canvas id="cvHotCold"></canvas></div>
       <div class="chart-box"><h4 data-i18n="chart_gap">遗漏值（连续未出期数）</h4>
         <canvas id="cvGap"></canvas></div>
       <div class="chart-box"><h4 data-i18n="chart_zone">八区段 · 期均出现数（期望2.5）</h4>
         <canvas id="cvZone"></canvas></div>
     </div>
     <details class="small"><summary data-i18n="param_detail">参数 / 组合枚举 Top10 / 口径声明</summary>
       <pre class="json-pre">${JSON.stringify(
         {params: r.params, shortlist: r.shortlist, combo_ranked: r.combo_ranked},
         null, 2)}</pre>
       ${r.disclaimers.map(d => `<div class="note warn">${d}</div>`).join("")}
     </details>`;
  applyI18n();
  drawFullCharts(r);
}

function drawFullCharts(r){
  const p = r.params;
  Charts.heatmap($("cvHotCold"), r.hot_cold, row =>
    `<b>#${row.number}</b><br>${t("tip_freq")}: ${row.freq} (${row.freq_pct}%)<br>` +
    `${t("tip_recent")}: ${row.recent_count}/${row.recent_window} z=${row.recent_z}<br>` +
    `${t("tip_gap")}: ${row.gap} (tail ${row.gap_tail_pct}%)`);

  Charts.barChart($("cvGap"), {
    title: t("chart_gap"),
    labels: r.hot_cold.map(x => x.number),
    series: [{name: "gap",
              values: r.hot_cold.map(x => x.gap),
              color: v => v >= p.gap_cold ? "#3c6edc"
                         : v <= p.gap_active ? "#dc603c" : "#aab2bd"}],
    refs: [{y: 3, color: "#9aa4b0"}, {y: p.gap_cold, color: "#3c6edc"}],
    height: 230,
    tip: (si, v, lab) => `#${lab} · gap=${v}`
  });

  Charts.barChart($("cvZone"), {
    title: t("chart_zone"),
    labels: r.segments.zones.map(z => `${z.range[0]}-${z.range[1]}`),
    series: [{name: "per draw",
              values: r.segments.zones.map(z => z.per_draw),
              color: (v, i) => {
                const zz = r.segments.zones[i].z;
                return zz > 1.28 ? "#dc603c" : zz < -1.28 ? "#3c6edc" : "#aab2bd";
              }}],
    refs: [{y: 2.5, color: "#697583"}],
    height: 230,
    tip: (si, v, lab) => `${lab}<br>${v} /draw · χ²=${r.segments.chi_square}`
  });
}

$("btnFull").onclick = async () => {
  const out = $("fullOut");
  try{
    out.innerHTML = msg("", t("msg_full"));
    const r = await api("/api/analytics/full");
    cache.full = r;
    renderFull(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ backtest */
function verdictText(r){
  const z = r.z_vs_baseline;
  if(z > 1.96) return t("verdict_pos", {m: r.mean_hits, z});
  if(z < -1.96) return t("verdict_neg", {m: r.mean_hits, z});
  return t("verdict_none", {m: r.mean_hits, z, n: r.trials});
}

function renderBacktest(r){
  $("backtestOut").innerHTML =
    `<table class="table"><tr><th>${t("th_hits")}</th>
       ${[0,1,2,3,4].map(k => `<th>${k}</th>`).join("")}</tr>
      <tr><td>${t("bt_count")}</td>
       ${[0,1,2,3,4].map(k => `<td>${r.hit_distribution[k]}</td>`).join("")}</tr>
      <tr><td>${t("bt_pct")}</td>
       ${[0,1,2,3,4].map(k => `<td>${r.hit_distribution_pct[k]}%</td>`).join("")}</tr>
      <tr><td>${t("bt_base_pct")}</td>
       ${[0,1,2,3,4].map(k => `<td>${r.baseline.distribution_pct[k]}%</td>`).join("")}</tr>
     </table>
     <p class="small">${t("bt_mean")}: <b>${r.mean_hits}</b>
       (95% CI ${r.mean_hits_bootstrap95.join(" – ")}) ·
       ${t("bt_base_mean")}: ${r.baseline.mean_hits} · z=${r.z_vs_baseline}</p>
     <p class="small">${t("bt_ge1")}: ${r.hit_at_least_1_pct}%
       (${t("base_word")} ${r.baseline.at_least_1_pct}%) ·
       ${t("bt_ge2")}: ${r.hit_at_least_2_pct}%
       (${t("base_word")} ${r.baseline.at_least_2_pct}%)</p>
     <div class="note ${Math.abs(r.z_vs_baseline) > 1.96
        ? (r.z_vs_baseline > 0 ? "" : "err") : "warn"}">${verdictText(r)}</div>
     <p class="small">${t("bt_note", {n: r.trials})}</p>
     <canvas id="cvBacktest" style="margin-top:8px"></canvas>`;
  applyI18n();
  Charts.barChart($("cvBacktest"), {
    title: t("chart_bt"),
    labels: ["0", "1", "2", "3", "4"],
    series: [
      {name: t("bt_strategy"),
       values: [0,1,2,3,4].map(k => r.hit_distribution_pct[k]), color: "#2f6fed"},
      {name: t("bt_baseline"),
       values: [0,1,2,3,4].map(k => r.baseline.distribution_pct[k]), color: "#c9d2dd"}
    ],
    height: 230,
    tip: (si, v) => `${si === 0 ? t("bt_strategy") : t("bt_baseline")}: ${v}%`
  });
}

$("btnBacktest").onclick = async () => {
  const out = $("backtestOut");
  try{
    const tw = Number($("btWindow").value);
    if(!Number.isInteger(tw) || tw < 5) throw new Error(t("err_bt_window"));
    out.innerHTML = msg("", t("msg_bt"));
    const r = await api("/api/backtest", {train_window: tw});
    cache.backtest = r;
    renderBacktest(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ------------------------------------------------------------ analysis logs */
function renderLogs(logs){
  if(!logs.length){ $("logsOut").innerHTML = msg("", t("no_logs")); return; }
  $("logsOut").innerHTML =
    `<table class="table"><tr>
       <th>${t("th_finished")}</th><th>${t("th_type")}</th><th>${t("th_range")}</th>
       <th>${t("th_final")}</th><th>${t("th_elapsed")}</th></tr>
       ${logs.map(l => `<tr class="log-row" data-file="${l.file}">
         <td>${l.finished_at ?? "-"}</td><td>${l.analysis_type}</td>
         <td>${l.first_period}→${l.last_period} (${l.periods_count})</td>
         <td>${(l.final_numbers || []).join(" ")}</td>
         <td>${l.elapsed_ms}ms</td></tr>`).join("")}</table>`;
  document.querySelectorAll(".log-row").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.onclick = async () => {
      try{
        const r = await api("/api/analytics/logs/" + encodeURIComponent(tr.dataset.file));
        $("logDetailOut").innerHTML =
          `<details open><summary>${tr.dataset.file}</summary>
           <pre class="json-pre">${JSON.stringify(r.log, null, 2)}</pre></details>`;
      }catch(e){ $("logDetailOut").innerHTML = msg("err", e.message); }
    };
  });
}

async function refreshLogs(){
  try{
    const r = await api("/api/analytics/logs");
    cache.logs = r.logs;
    renderLogs(r.logs);
  }catch(e){ $("logsOut").innerHTML = msg("err", e.message); }
}

$("btnLoadLogs").onclick = refreshLogs;

/* ------------------------------------------------------------ language switch */
function syncLangButtons(){
  document.querySelectorAll(".lang-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.lang === I18N.lang));
}

document.querySelectorAll(".lang-btn").forEach(b => {
  b.addEventListener("click", () => I18N.setLang(b.dataset.lang));
});

/* 语言切换：静态文案由 i18n.js 处理，这里把已显示的动态内容按缓存重渲染 */
document.addEventListener("i18n:change", () => {
  syncLangButtons();
  if(cache.status) renderStatus(cache.status);
  if(cache.full) renderFull(cache.full);
  if(cache.analyze) renderAnalyze(cache.analyze);
  if(cache.optimize) renderOptimize(cache.optimize);
  if(cache.recommend) renderRecommend(cache.recommend);
  if(cache.backtest) renderBacktest(cache.backtest);
  if(cache.logs) renderLogs(cache.logs);
  if(cache.manual) renderManual(cache.manual);
  if(cache.crawl){
    if(cache.crawl.dry_run) renderDryRun(cache.crawl);
    else if(cache.crawl.ran) renderCrawlDone(cache.crawl);
    else renderCrawlFresh(cache.crawl);
  }
});

applyI18n();
syncLangButtons();
refreshStatus();
refreshLogs();

/* ------------------------------------------------------------ 多规模扩展（1/2/3/4 码） */
let msSize = 1;
const msCache = {};

function msSyncSize(){
  document.querySelectorAll(".size-btn").forEach(b =>
    b.classList.toggle("active", Number(b.dataset.size) === msSize));
  $("msSizeNote").textContent = t("sm_size_note", {m: msSize});
}

document.querySelectorAll(".size-btn").forEach(b => {
  b.addEventListener("click", () => {
    msSize = Number(b.dataset.size);
    msSyncSize();
    if(msCache.theory) renderMsTheory(msCache.theory);
  });
});

/* ---- 理论概率对比表 ---- */
function renderMsTheory(r){
  const rows = r.rows.map(x =>
    `<tr class="${x.m === msSize ? "ms-active" : ""}">
       <td><b>${x.m}</b></td><td>${x.all_hit_pct}%</td><td>1/${x.odds_1_in}</td>
       <td>${x.at_least_1_pct}%</td><td>${x.mean_hits}</td><td>${x.sd_hits}</td></tr>`).join("");
  $("msTheoryOut").innerHTML =
    `<table class="table"><tr>
       <th>${t("sm_th_m")}</th><th>${t("sm_th_all_hit")}</th><th>${t("sm_th_odds")}</th>
       <th>${t("sm_th_ge1")}</th><th>${t("sm_th_mean")}</th><th>${t("sm_th_sd")}</th></tr>
       ${rows}</table>
     <canvas id="cvMsTheory" style="margin-top:8px"></canvas>`;
  Charts.barChart($("cvMsTheory"), {
    title: t("sm_chart_theory"),
    labels: r.rows.map(x => x.m),
    series: [
      {name: t("sm_th_all_hit"), values: r.rows.map(x => x.all_hit_pct), color: "#2f6fed"},
      {name: t("sm_th_ge1"), values: r.rows.map(x => x.at_least_1_pct), color: "#c9d2dd"}
    ],
    height: 220,
    tip: (si, v, lab) => `${si === 0 ? t("sm_th_all_hit") : t("sm_th_ge1")} · m=${lab}: ${v}%`
  });
}

async function refreshMsTheory(){
  try{
    const r = await api("/api/picksize/theory");
    msCache.theory = r;
    renderMsTheory(r);
  }catch(e){ $("msTheoryOut").innerHTML = msg("err", e.message); }
}

/* ---- 概率分析（个数=规模） ---- */
function renderMsAnalyze(r){
  const th = r.theory_m, e = r.empirical;
  const coLabel = e.co_kind === "four" ? t("observed_co") : t("sm_co_" + e.co_kind);
  const pctTxt = (e.percentile_among_observed == null) ? "-"
    : `${e.percentile_among_observed}% · ${t("pct_among", {n: e.observed_basis})}`;
  const pairsTable = (r.pairs && r.pairs.length)
    ? `<table class="table"><tr><th>${t("sm_th_combo")}</th><th>${t("th_count")}</th>
         <th>${t("th_expected")}</th><th>${t("th_z")}</th></tr>
         ${r.pairs.map(p => `<tr><td>${p.pair.join(" ")}</td><td>${p.count}</td>
           <td>${p.expected}</td><td>${p.z}</td></tr>`).join("")}</table>` : "";
  $("msAnalyzeOut").innerHTML =
    `<div class="rec">${balls(r.combo)}</div>
     <dl class="kv">
       <dt>${t("exact_prob")}</dt><dd>${th.all_hit_pct}% (1/${th.odds_1_in})</dd>
       <dt>${t("sm_at_least_1")}</dt><dd>${th.at_least_1_pct}%</dd>
       <dt>${coLabel}</dt><dd>${e.co_occurrence} ${t("times_word")}
           (${t("expected_word")} ${e.co_expected}) — z ${e.co_z}</dd>
       <dt>${t("emp_score")}</dt><dd>${e.score} (${pctTxt})</dd>
       <dt>${t("hit_history")}</dt><dd>${e.hit_periods.length ? e.hit_periods.join(", ") : t("never")}</dd>
     </dl>
     <table class="table"><tr>
       <th>${t("th_num")}</th><th>${t("th_count")}</th>
       <th>${t("th_expected")}</th><th>${t("th_z")}</th></tr>
       ${r.numbers.map(n => `<tr><td>${n.number}</td><td>${n.count}</td>
         <td>${n.expected}</td><td>${n.z}</td></tr>`).join("")}
     </table>
     ${pairsTable}
     <div class="note warn">${t("sm_disclaimer", {m: r.size, pct: th.all_hit_pct, o: th.odds_1_in})}</div>`;
}

$("btnMsAnalyze").onclick = async () => {
  const out = $("msAnalyzeOut");
  try{
    const nums = parseCombo($("msInput").value, msSize);
    out.innerHTML = msg("", t("msg_analyzing"));
    const r = await api("/api/analyze/m", {numbers: nums});
    msCache.analyze = r;
    renderMsAnalyze(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ---- 科学筛选（80 → N 码） ---- */
function renderMsPick(r){
  const th = r.theory_m;
  $("msPickOut").innerHTML =
    `<div class="final4">
       ${r.numbers.map(n => `<span class="ball gold">${n}</span>`).join("")}
       <span class="small">${t("combo_score_word")} ${r.combo_score}
         · ${JSON.stringify(r.combo_score_parts)}</span>
     </div>
     <p class="small">${t("exact_prob")}: ${th.all_hit_pct}% (1/${th.odds_1_in})
       · ${t("sm_at_least_1")}: ${th.at_least_1_pct}%</p>
     <table class="table wrap-table">
       <tr><th>${t("th_num")}</th><th>${t("th_rank")}</th><th>${t("th_total")}</th>
         <th>${t("th_full_z")}</th><th>${t("th_recent_z")}</th><th>${t("th_gap_s")}</th>
         <th>${t("th_seg_s")}</th><th>${t("th_model")}</th><th>${t("th_freq")}</th>
         <th>${t("th_wilson")}</th><th>${t("th_hotcold")}</th></tr>
       ${r.ranking_basis.map(x => `<tr>
          <td><b>${x.number}</b></td><td>${x.rank_of_80}</td><td>${x.total_score}</td>
          <td>${x.factors.full_freq_z}</td><td>${x.factors.recent_z}</td>
          <td>${x.factors.gap_score}</td><td>${x.factors.segment_score}</td>
          <td>${x.model_index_pct}%</td><td>${x.empirical_freq_pct}%</td>
          <td>${x.wilson95_ci_pct[0]}–${x.wilson95_ci_pct[1]}%</td>
          <td><span class="mini-lab rc-${x.recent_label}">${t("lab_r_" + x.recent_label)}</span>
              <span class="mini-lab gl-${x.gap_label}">${t("lab_g_" + x.gap_label)}</span></td>
        </tr>`).join("")}
     </table>
     <details class="small"><summary>${t("param_detail")}</summary>
       <pre class="json-pre">${JSON.stringify(
         {size: r.size, params: r.params, shortlist: r.shortlist, combo_ranked: r.combo_ranked},
         null, 2)}</pre>
       ${r.disclaimers.map(d => `<div class="note warn">${d}</div>`).join("")}
     </details>`;
  applyI18n();
}

$("btnMsPick").onclick = async () => {
  const out = $("msPickOut");
  try{
    out.innerHTML = msg("", t("msg_full"));
    const r = await api("/api/analytics/m", {size: msSize});
    msCache.pick = r;
    renderMsPick(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ---- 第 N+1 日推荐 ---- */
function renderMsRecommend(r){
  $("msRecommendOut").innerHTML =
    `<p class="small">${t("basis_word")}: ${r.basis.periods_used} ${t("periods_word")}
      (${r.basis.first_period} → ${r.basis.last_period}) · ${t("candidate_pool_word")}
      ${r.candidate_pool.join(" ")}</p>` +
    r.recommendations.map(rec =>
      `<div class="rec">
         ${balls(rec.combo)}
         <div class="conf">
           <div class="track"><div class="fill ${rec.confidence_label}"
             style="width:${(rec.confidence*100).toFixed(0)}%"></div></div>
           <div class="meta">${t("conf_score")} ${rec.score} · ${t("conf_stability")} ${rec.stability}
             · ${t("conf_margin")} ${rec.margin} · ${t("conf_parts")} ${JSON.stringify(rec.score_parts)}</div>
         </div>
         <span class="label ${rec.confidence_label}">${t("label_" + rec.confidence_label)}
           ${(rec.confidence*100).toFixed(0)}%</span>
       </div>`).join("") +
    `<details class="small" style="margin-top:8px"><summary>${t("methodology")}</summary>
       <p>${r.method}</p><p class="note warn">${r.disclaimer}</p></details>`;
}

$("btnMsRecommend").onclick = async () => {
  const out = $("msRecommendOut");
  try{
    const body = {size: msSize, top_k: Number($("msTopK").value) || 5};
    const n = Number($("msLastN").value);
    if(Number.isInteger(n) && n > 0) body.last_n = n;
    body.auto_strategy = $("msAutoStrategy").checked;
    out.innerHTML = msg("", t("msg_generating"));
    const r = await api("/api/recommend/m", body);
    msCache.recommend = r;
    renderMsRecommend(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* ---- 历史回测 ---- */
function msVerdictText(r){
  const z = r.z_vs_baseline, b = r.baseline.mean_hits;
  if(z > 1.96) return t("sm_verdict_pos", {m: r.mean_hits, z, b});
  if(z < -1.96) return t("sm_verdict_neg", {m: r.mean_hits, z, b});
  return t("sm_verdict_none", {m: r.mean_hits, z, n: r.trials, b});
}

function renderMsBacktest(r){
  const ks = [...Array(r.size + 1).keys()];
  $("msBacktestOut").innerHTML =
    `<table class="table"><tr><th>${t("th_hits")}</th>
       ${ks.map(k => `<th>${k}</th>`).join("")}</tr>
      <tr><td>${t("bt_count")}</td>
       ${ks.map(k => `<td>${r.hit_distribution[k]}</td>`).join("")}</tr>
      <tr><td>${t("bt_pct")}</td>
       ${ks.map(k => `<td>${r.hit_distribution_pct[k]}%</td>`).join("")}</tr>
      <tr><td>${t("bt_base_pct")}</td>
       ${ks.map(k => `<td>${r.baseline.distribution_pct[k]}%</td>`).join("")}</tr>
     </table>
     <p class="small">${t("bt_mean")}: <b>${r.mean_hits}</b>
       (95% CI ${r.mean_hits_bootstrap95.join(" – ")}) ·
       ${t("bt_base_mean")}: ${r.baseline.mean_hits} · z=${r.z_vs_baseline}</p>
     <p class="small">${t("bt_ge1")}: ${r.hit_at_least_1_pct}%
       (${t("base_word")} ${r.baseline.at_least_1_pct}%) ·
       ${t("bt_ge2")}: ${r.hit_at_least_2_pct}%
       (${t("base_word")} ${r.baseline.at_least_2_pct}%)</p>
     <div class="note ${Math.abs(r.z_vs_baseline) > 1.96
        ? (r.z_vs_baseline > 0 ? "" : "err") : "warn"}">${msVerdictText(r)}</div>
     <p class="small">${t("bt_note", {n: r.trials})}</p>
     <canvas id="cvMsBacktest" style="margin-top:8px"></canvas>`;
  applyI18n();
  Charts.barChart($("cvMsBacktest"), {
    title: t("chart_bt"),
    labels: ks.map(String),
    series: [
      {name: t("bt_strategy"),
       values: ks.map(k => r.hit_distribution_pct[k]), color: "#2f6fed"},
      {name: t("bt_baseline"),
       values: ks.map(k => r.baseline.distribution_pct[k]), color: "#c9d2dd"}
    ],
    height: 230,
    tip: (si, v) => `${si === 0 ? t("bt_strategy") : t("bt_baseline")}: ${v}%`
  });
}

$("btnMsBacktest").onclick = async () => {
  const out = $("msBacktestOut");
  try{
    const tw = Number($("msBtWindow").value);
    if(!Number.isInteger(tw) || tw < 5) throw new Error(t("err_bt_window"));
    out.innerHTML = msg("", t("msg_bt"));
    const r = await api("/api/backtest/m", {size: msSize, train_window: tw});
    msCache.backtest = r;
    renderMsBacktest(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
};

/* 语言切换：按缓存重渲染多规模区块（独立于既有4码缓存，互不影响） */
document.addEventListener("i18n:change", () => {
  msSyncSize();
  if(msCache.theory) renderMsTheory(msCache.theory);
  if(msCache.analyze) renderMsAnalyze(msCache.analyze);
  if(msCache.pick) renderMsPick(msCache.pick);
  if(msCache.recommend) renderMsRecommend(msCache.recommend);
  if(msCache.backtest) renderMsBacktest(msCache.backtest);
});

msSyncSize();
refreshMsTheory();

/* ============================================================ 推荐命中追踪 */
const trackCache = {hit:null, strategy:null, recent:null};

async function refreshTrackHit(){
  const sz = document.getElementById("trackSize").value;
  const out = document.getElementById("trackHitOut");
  out.innerHTML = msg("loading");
  try{
    const q = sz ? `?size=${sz}` : "";
    const r = await api("/api/hitrate" + q);
    trackCache.hit = r;
    renderTrackHit(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
}

function renderTrackHit(r){
  const out = document.getElementById("trackHitOut");
  const rep = r.report;
  if(rep.samples === 0){
    out.innerHTML = `<p class="note">${rep.note || "暂无已对比的推荐记录。"}</p>`;
    return;
  }
  const b = rep.baseline;
  const t = rep.top1_hit;
  const bh = rep.best_hit;
  const badge = (z)=>{
    if(Math.abs(z)<=1.96) return '<span class="tag">≈ 基准</span>';
    return z>1.96 ? '<span class="tag good">优于基准</span>' : '<span class="tag bad">弱于基准</span>';
  };
  out.innerHTML = `
    <div class="row">
      <div class="stat"><b>${rep.samples}</b><span>对比样本</span></div>
      <div class="stat"><b>${t.mean}</b><span>top1 平均命中</span></div>
      <div class="stat"><b>${bh.mean}</b><span>最佳组合平均命中</span></div>
      <div class="stat"><b>${rep.candidate_pool.mean_hits}</b><span>12码候选池平均命中</span></div>
    </div>
    <table class="tbl">
      <tr><th>指标</th><th>top1 组合</th><th>最佳组合</th><th>随机基准(${b.size}码)</th></tr>
      <tr><td>平均命中</td><td>${t.mean}</td><td>${bh.mean}</td><td>${b.mean_hits}</td></tr>
      <tr><td>≥1 命中率</td><td>${t.at_least_1_pct}%</td><td>${bh.at_least_1_pct}%</td><td>${b.at_least_1_pct}%</td></tr>
      <tr><td>≥2 命中率</td><td>${t.at_least_2_pct}%</td><td>${bh.at_least_2_pct}%</td><td>${b.at_least_2_pct}%</td></tr>
      <tr><td>z(vs基准)</td><td>${t.z_vs_baseline} ${badge(t.z_vs_baseline)}</td><td colspan="2"></td></tr>
    </table>
    <p class="note">${rep.verdict}</p>`;
}

async function refreshTrackStrategy(){
  const out = document.getElementById("trackStrategyOut");
  out.innerHTML = msg("loading");
  try{
    const r = await api("/api/strategy/stats");
    trackCache.strategy = r;
    renderTrackStrategy(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
}

function renderTrackStrategy(r){
  const out = document.getElementById("trackStrategyOut");
  if(!r.stats || r.stats.length===0){
    out.innerHTML = `<p class="note">暂无策略统计数据。生成推荐并等待开奖对比后即可查看。</p>`;
    return;
  }
  const suggested = r.suggested || {};
  const rows = r.stats.map(s=>{
    const sug = suggested[String(s.size)];
    const isSug = sug && sug.last_n === s.last_n;
    return `<tr${isSug?' class="hl"':''}>
      <td>${s.size}码</td>
      <td>${s.last_n===null?'全部':s.last_n}</td>
      <td>${s.samples}</td>
      <td>${s.mean_top1}</td>
      <td>${s.mean_best}</td>
      <td>${s.at_least_1_pct}%</td>
      <td>${isSug?'★ 推荐':''}</td>
    </tr>`;
  }).join("");
  out.innerHTML = `
    <table class="tbl">
      <tr><th>规模</th><th>窗口last_n</th><th>样本</th><th>top1均值</th><th>最佳均值</th><th>≥1命中</th><th>自适应</th></tr>
      ${rows}
    </table>
    <p class="note">自适应规则：样本≥10 的策略中取 top1 均值最高者，且相对默认(全部)优势≥0.05 才切换；否则保持默认。生成推荐时勾选 auto_strategy 即启用。</p>`;
}

async function refreshTrackRecent(){
  const sz = document.getElementById("trackSize").value;
  const out = document.getElementById("trackRecentOut");
  out.innerHTML = msg("loading");
  try{
    const q = `?compared=true&limit=20${sz?'&size='+sz:''}`;
    const r = await api("/api/recommendations" + q);
    trackCache.recent = r;
    renderTrackRecent(r);
  }catch(e){ out.innerHTML = msg("err", e.message); }
}

function renderTrackRecent(r){
  const out = document.getElementById("trackRecentOut");
  const recs = r.recommendations || [];
  if(!recs.length){ out.innerHTML = `<p class="note">暂无已对比记录。</p>`; return; }
  const rows = recs.map(x=>{
    const combo = (x.recommendations && x.recommendations[0]) ? x.recommendations[0].combo.join(",") : "-";
    const res = x.result || {};
    return `<tr>
      <td>${x.target_period}</td>
      <td>${x.size}码</td>
      <td>${combo}</td>
      <td><b>${res.top1_hit}</b></td>
      <td>${res.best_hit}</td>
      <td>${(res.actual||[]).slice(0,8).join(",")}…</td>
    </tr>`;
  }).join("");
  out.innerHTML = `
    <table class="tbl">
      <tr><th>目标期</th><th>规模</th><th>top1 组合</th><th>top1命中</th><th>最佳命中</th><th>开奖(前8)</th></tr>
      ${rows}
    </table>`;
}

function refreshTrackAll(){ refreshTrackHit(); refreshTrackStrategy(); refreshTrackRecent(); }

document.getElementById("btnTrackRefresh").addEventListener("click", refreshTrackAll);
document.getElementById("btnTrackReconcile").addEventListener("click", async ()=>{
  try{
    const r = await api("/api/recommendations/reconcile", {});
    flash(`对比完成：新增 ${r.compared} 条，待对比 ${r.still_pending} 条`);
    refreshTrackAll();
  }catch(e){ flash("对比失败: "+e.message, true); }
});
document.getElementById("trackSize").addEventListener("change", ()=>{ refreshTrackHit(); refreshTrackRecent(); });

refreshTrackAll();

