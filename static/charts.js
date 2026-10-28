/* charts.js —— 辅助分析图表（原生 Canvas，零第三方依赖，支持高DPI）
 * 提供：
 *   barChart(canvas, opts)  通用分组柱状图（参考线、每柱自定义颜色）
 *   heatmap(canvas, rows)   80码冷热网格（8区×10列，悬浮明细）
 */
(function () {

  /* ---- 高 DPI 适配：按 devicePixelRatio 放大位图，CSS 尺寸不变 ---- */
  function prepare(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.parentElement.clientWidth;
    const cssH = cssHeight || 260;
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = "11px sans-serif";
    return {ctx, W: cssW, H: cssH};
  }

  /* ---- 惰性创建的全局 tooltip ---- */
  let tipEl = null;
  function tip(text, x, y) {
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.id = "chartTip";
      document.body.appendChild(tipEl);
    }
    if (!text) { tipEl.style.display = "none"; return; }
    tipEl.innerHTML = text;
    tipEl.style.display = "block";
    tipEl.style.left = Math.min(x + 12, window.innerWidth - 180) + "px";
    tipEl.style.top = (y + 12) + "px";
  }
  function hideTip() { if (tipEl) tipEl.style.display = "none"; }

  /* ============================================================ 柱状图 */
  function barChart(canvas, opts) {
    /*
     * opts = {
     *   labels: [...],
     *   series: [{ name, values:[], color: "#hex" | (value,i)=>cssColor }],
     *   refs:   [{ y, color, dash:[4,4] }],
     *   title:  "...",
     *   tip:    (seriesIndex, value, label) => html   // 可省略
     * }
     */
    const {ctx, W, H} = prepare(canvas, opts.height || 250);
    const pad = {l: 42, r: 12, t: opts.title ? 30 : 12, b: 46};
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    let yMax = 0, yMin = 0;
    opts.series.forEach(s => s.values.forEach(v => { yMax = Math.max(yMax, v); }));
    (opts.refs || []).forEach(r => { yMax = Math.max(yMax, r.y); });
    yMax *= 1.12;

    const x = px => pad.l + px;
    const y = v => pad.t + plotH - (v - yMin) / (yMax - yMin || 1) * plotH;

    /* 背景与标题 */
    ctx.clearRect(0, 0, W, H);
    if (opts.title) {
      ctx.fillStyle = "#1f2933";
      ctx.font = "bold 12px sans-serif";
      ctx.fillText(opts.title, pad.l, 16);
      ctx.font = "11px sans-serif";
    }

    /* 网格 + Y 轴刻度 */
    ctx.strokeStyle = "#e5e9ee";
    ctx.fillStyle = "#697583";
    ctx.lineWidth = 1;
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const v = yMin + (yMax - yMin) * i / ticks;
      const yy = y(v);
      ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
      ctx.fillText(v.toFixed(yMax < 4 ? 2 : 0), 6, yy + 4);
    }

    /* 参考线 */
    (opts.refs || []).forEach(r => {
      ctx.save();
      ctx.strokeStyle = r.color;
      ctx.setLineDash(r.dash || [5, 4]);
      ctx.beginPath(); ctx.moveTo(pad.l, y(r.y)); ctx.lineTo(W - pad.r, y(r.y)); ctx.stroke();
      ctx.restore();
    });

    /* 柱 */
    const nGroups = opts.labels.length;
    const groupW = plotW / nGroups;
    const nS = opts.series.length;
    const barW = Math.min(26, groupW / (nS + 1));
    const hitAreas = [];
    opts.series.forEach((s, si) => {
      s.values.forEach((v, i) => {
        const cx = pad.l + groupW * (i + 0.5) + (si - (nS - 1) / 2) * barW;
        const color = typeof s.color === "function" ? s.color(v, i, si) : s.color;
        ctx.fillStyle = color;
        const yy = y(v);
        ctx.fillRect(cx - barW / 2 + 1, yy, barW - 2, pad.t + plotH - yy);
        hitAreas.push({x: cx - barW / 2, w: barW, si, i, v});
      });
    });

    /* X 轴标签 */
    ctx.fillStyle = "#697583";
    opts.labels.forEach((lab, i) => {
      const cx = pad.l + groupW * (i + 0.5);
      const text = String(lab);
      if (text.length > 5) {
        ctx.save();
        ctx.translate(cx, H - pad.b + 14);
        ctx.rotate(-Math.PI / 5);
        ctx.fillText(text, 0, 0);
        ctx.restore();
      } else {
        ctx.fillText(text, cx - ctx.measureText(text).width / 2, H - pad.b + 16);
      }
    });

    /* 图例 */
    if (nS > 1) {
      let lx = pad.l;
      ctx.font = "11px sans-serif";
      opts.series.forEach(s => {
        const color = typeof s.color === "function" ? "#888" : s.color;
        ctx.fillStyle = color;
        ctx.fillRect(lx, pad.t - 12, 10, 10);
        ctx.fillStyle = "#3c4856";
        ctx.fillText(s.name, lx + 14, pad.t - 3);
        lx += ctx.measureText(s.name).width + 34;
      });
    }

    /* 悬浮交互（每次重绘绑定一次） */
    canvas.onmousemove = (ev) => {
      if (!opts.tip) { hideTip(); return; }
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const hit = hitAreas.find(a => mx >= a.x && mx <= a.x + a.w && my >= pad.t && my <= pad.t + plotH);
      if (hit) tip(opts.tip(hit.si, hit.v, opts.labels[hit.i]), ev.clientX, ev.clientY);
      else hideTip();
    };
    canvas.onmouseleave = hideTip;
  }

  /* ============================================================ 冷热网格 */
  function heatmap(canvas, rows, tipFn) {
    const {ctx, W, H} = prepare(canvas, 260);
    const pad = {l: 30, r: 10, t: 10, b: 26};
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;
    const cw = plotW / 10;
    const ch = plotH / 8;
    ctx.clearRect(0, 0, W, H);

    const cells = [];
    for (let z = 0; z < 8; z++) {
      for (let c = 0; c < 10; c++) {
        const num = z * 10 + c + 1;
        const r = rows[num - 1];
        const rz = Math.max(-2.5, Math.min(2.5, r.recent_z));
        let fill;
        if (r.recent_label === "hot")
          fill = `rgba(220,70,60,${0.25 + 0.28 * Math.abs(rz)})`;
        else if (r.recent_label === "cold_recent")
          fill = `rgba(60,110,220,${0.25 + 0.28 * Math.abs(rz)})`;
        else
          fill = "rgba(150,160,170,0.22)";
        const x0 = pad.l + c * cw, y0 = pad.t + z * ch;
        ctx.fillStyle = fill;
        ctx.fillRect(x0 + 2, y0 + 2, cw - 4, ch - 4);
        if (r.gap_label === "cold_gap") {
          ctx.strokeStyle = "#1d4ed8";
          ctx.lineWidth = 2;
          ctx.strokeRect(x0 + 2, y0 + 2, cw - 4, ch - 4);
        }
        ctx.fillStyle = "#2a3340";
        ctx.font = "11px sans-serif";
        ctx.fillText(String(num), x0 + cw / 2 - 7, y0 + ch / 2 + 4);
        cells.push({x: x0, y: y0, w: cw, h: ch, num});
      }
      /* 行（区）标签 */
      ctx.fillStyle = "#697583";
      ctx.font = "10px sans-serif";
      ctx.fillText("Z" + (z + 1), 6, pad.t + z * ch + ch / 2 + 3);
    }

    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
      const hit = cells.find(a => mx >= a.x && mx <= a.x + a.w && my >= a.y && my <= a.y + a.h);
      if (hit) tip(rows[hit.num - 1], ev.clientX, ev.clientY, tipFn);
      else hideTip();
    };
    canvas.onmouseleave = hideTip;
  }

  function tip(row, clientX, clientY, tipFn) {
    tip(tipFn ? tipFn(row) : defaultTip(row), clientX, clientY);
  }

  function defaultTip(r) {
    return `<b>${r.number}</b><br>freq ${r.freq} (${r.freq_pct}%)` +
           `<br>recent ${r.recent_count}/${r.recent_window} (z ${r.recent_z})` +
           `<br>gap ${r.gap} (tail ${r.gap_tail_pct}%)`;
  }

  window.Charts = {barChart, heatmap, hideTip};
})();
