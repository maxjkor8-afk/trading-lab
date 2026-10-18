"use strict";

// The dashboard never computes a result. It asks the harness for one and draws
// what comes back, so there is no second implementation of the arithmetic.
// Layout follows the QuantStats tearsheet convention: charts left, dense
// metrics right, strategy against a control column throughout.

const LWC = window.LightweightCharts;

const C = {
  up: "#2ebd85",
  down: "#e5484d",
  strategy: "#6ea8fe",
  control: "#6f7c98",
  text: "#6f7c98",
  grid: "#141b28",
  border: "#1e2636",
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const el = (id) => document.getElementById(id);
const statusEl = el("status");

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
}

// -- formatting ------------------------------------------------------------

const isNum = (x) => x !== null && x !== undefined && !Number.isNaN(x);
const pct = (x, d = 2) => (isNum(x) ? (x * 100).toFixed(d) + "%" : "n/a");
const money = (x) =>
  isNum(x) ? x.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "n/a";
const num = (x, d = 2) => (isNum(x) ? x.toFixed(d) : "n/a");
const day = (iso) => (iso ? iso.slice(0, 10) : "n/a");
const cls = (x) => (!isNum(x) ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");

// -- charts ----------------------------------------------------------------

function baseOptions() {
  return {
    autoSize: true,
    layout: {
      background: { color: "transparent" },
      textColor: C.text,
      fontFamily: "ui-monospace, Consolas, monospace",
      fontSize: 10,
      attributionLogo: false,
    },
    grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
    rightPriceScale: { borderColor: C.border, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderColor: C.border, timeVisible: false, secondsVisible: false },
    crosshair: { mode: LWC.CrosshairMode.Normal },
    // The wheel belongs to the page: four stacked charts that each swallow it
    // make everything below the first one unreachable. Drag still pans and the
    // axes still zoom.
    handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: false, pinch: true, axisPressedMouseMove: true },
  };
}

const percentFormat = { type: "percent", precision: 2, minMove: 0.01 };

const returnsChart = LWC.createChart(el("returns-chart"), baseOptions());
const drawdownChart = LWC.createChart(el("drawdown-chart"), baseOptions());
const sharpeChart = LWC.createChart(el("sharpe-chart"), baseOptions());
const priceChart = LWC.createChart(el("price-chart"), baseOptions());

const strategyLine = returnsChart.addSeries(LWC.LineSeries, {
  color: C.strategy,
  lineWidth: 2,
  priceFormat: percentFormat,
});

const controlLine = returnsChart.addSeries(LWC.LineSeries, {
  color: C.control,
  lineWidth: 1,
  lineStyle: 2,
  priceFormat: percentFormat,
});

const underwater = drawdownChart.addSeries(LWC.BaselineSeries, {
  baseValue: { type: "price", price: 0 },
  topLineColor: "rgba(0,0,0,0)",
  topFillColor1: "rgba(0,0,0,0)",
  topFillColor2: "rgba(0,0,0,0)",
  bottomLineColor: C.down,
  bottomFillColor1: "rgba(229, 72, 77, 0.05)",
  bottomFillColor2: "rgba(229, 72, 77, 0.5)",
  lineWidth: 1,
  priceFormat: percentFormat,
});

const sharpeLine = sharpeChart.addSeries(LWC.BaselineSeries, {
  baseValue: { type: "price", price: 0 },
  topLineColor: C.up,
  topFillColor1: "rgba(46, 189, 133, 0.28)",
  topFillColor2: "rgba(46, 189, 133, 0.02)",
  bottomLineColor: C.down,
  bottomFillColor1: "rgba(229, 72, 77, 0.02)",
  bottomFillColor2: "rgba(229, 72, 77, 0.28)",
  lineWidth: 1,
  priceFormat: { type: "price", precision: 2, minMove: 0.01 },
});

const candles = priceChart.addSeries(LWC.CandlestickSeries, {
  upColor: C.up,
  downColor: C.down,
  borderUpColor: C.up,
  borderDownColor: C.down,
  wickUpColor: C.up,
  wickDownColor: C.down,
});

let markersPrimitive = null;

// Keep every pane on the same window so a drawdown lines up with the bar that
// caused it.
const charts = [returnsChart, drawdownChart, sharpeChart, priceChart];
let syncing = false;
charts.forEach((source) => {
  source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range || syncing) return;
    syncing = true;
    charts.forEach((target) => {
      if (target !== source) target.timeScale().setVisibleLogicalRange(range);
    });
    syncing = false;
  });
});

// -- rendering -------------------------------------------------------------

function renderHeadline(data) {
  const m = data.metrics;
  el("run-title").textContent =
    data.symbol + "  ·  " + day(m.start) + " to " + day(m.end) + "  ·  " + m.bars + " bars";
  const items = [
    ["return", pct(m.cumulative_return), cls(m.cumulative_return)],
    ["sharpe", num(m.sharpe), ""],
    ["max dd", pct(m.max_drawdown), "neg"],
    ["net p&l", money(m.net_pnl), cls(m.net_pnl)],
    ["costs", money(m.total_costs), ""],
  ];
  el("headline").innerHTML = items
    .map(
      ([k, v, c]) =>
        '<div class="h"><b class="' + c + '">' + v + "</b><span>" + k + "</span></div>"
    )
    .join("");
}

function row(label, strategyValue, controlValue, options) {
  const o = options || {};
  return (
    '<tr class="' + (o.rowClass || "") + '"><td class="key">' + label + "</td>" +
    '<td class="' + (o.colour ? cls(o.rawStrategy) : "") + '">' + strategyValue + "</td>" +
    '<td class="' + (o.colour ? cls(o.rawControl) : "") + '">' +
    (controlValue === undefined || controlValue === null ? "–" : controlValue) +
    "</td></tr>"
  );
}

function renderMetrics(data) {
  const m = data.metrics;
  const c = data.control_metrics;
  const get = (key) => (c ? c[key] : null);

  const lines = [
    ["Cumulative return", pct(m.cumulative_return), c ? pct(c.cumulative_return) : null, true, m.cumulative_return, get("cumulative_return")],
    ["Annualised return", pct(m.annualized_return), c ? pct(c.annualized_return) : null, true, m.annualized_return, get("annualized_return")],
    ["Sharpe", num(m.sharpe), c ? num(c.sharpe) : null, false],
    ["Sortino", num(m.sortino), c ? num(c.sortino) : null, false],
    ["Max drawdown", pct(m.max_drawdown), c ? pct(c.max_drawdown) : null, false],
    ["Max DD peak", day(m.max_drawdown_start), c ? day(c.max_drawdown_start) : null, false],
    ["Max DD valley", day(m.max_drawdown_end), c ? day(c.max_drawdown_end) : null, false],
    ["Initial equity", money(m.initial_equity), c ? money(c.initial_equity) : null, false],
    ["Final equity", money(m.final_equity), c ? money(c.final_equity) : null, false],
    ["Gross P&L", money(m.gross_pnl), c ? money(c.gross_pnl) : null, true, m.gross_pnl, get("gross_pnl")],
    ["Net P&L", money(m.net_pnl), c ? money(c.net_pnl) : null, true, m.net_pnl, get("net_pnl")],
    ["Fees", money(m.fees_paid), c ? money(c.fees_paid) : null, false],
    ["Slippage", money(m.slippage_paid), c ? money(c.slippage_paid) : null, false],
    ["Funding", money(m.funding_paid), c ? money(c.funding_paid) : null, false],
    ["Total costs", money(m.total_costs), c ? money(c.total_costs) : null, false],
    ["Costs / gross P&L", pct(m.cost_ratio), c ? pct(c.cost_ratio) : null, false],
    ["Trades", String(m.trades), c ? String(c.trades) : null, false],
    ["Win rate", pct(m.win_rate), c ? pct(c.win_rate) : null, false],
    ["Turnover", num(m.turnover) + "x", c ? num(c.turnover) + "x" : null, false],
    ["Bars", String(m.bars), c ? String(c.bars) : null, false],
  ];

  let body = lines
    .map(([label, s, ctl, colour, rawS, rawC]) =>
      row(label, s, ctl, { colour: colour, rawStrategy: rawS, rawControl: rawC })
    )
    .join("");

  if (c) {
    const drag = m.cumulative_return - c.cumulative_return;
    body +=
      '<tr class="rule highlight"><td class="key">Cost drag on return</td>' +
      '<td colspan="2" class="neg">' + pct(drag) + "</td></tr>";
  }

  el("metrics").innerHTML =
    "<thead><tr><th>metric</th><th>after costs</th><th>control</th></tr></thead><tbody>" +
    body +
    "</tbody>";
}

function renderHeatmap(data) {
  const rows = data.monthly_returns;
  if (!rows.length) {
    el("heatmap").innerHTML = '<div class="empty">No monthly data.</div>';
    return;
  }
  const years = [...new Set(rows.map((r) => r.year))].sort();
  const peak = Math.max(...rows.map((r) => Math.abs(r.value)), 0.0001);
  const lookup = new Map(rows.map((r) => [r.year + "-" + r.month, r.value]));

  const head =
    "<tr><th></th>" + MONTHS.map((m) => "<th>" + m + "</th>").join("") + "</tr>";

  const body = years
    .map((year) => {
      const cells = MONTHS.map((_, i) => {
        const value = lookup.get(year + "-" + (i + 1));
        if (value === undefined) return '<td class="blank"></td>';
        const weight = Math.min(Math.abs(value) / peak, 1) * 0.82 + 0.08;
        const base = value >= 0 ? "46, 189, 133" : "229, 72, 77";
        return (
          '<td style="background:rgba(' + base + "," + weight.toFixed(3) + ')" title="' +
          year + " " + MONTHS[i] + ": " + pct(value) + '">' +
          (value * 100).toFixed(1) +
          "</td>"
        );
      }).join("");
      return '<tr><td class="year">' + year + "</td>" + cells + "</tr>";
    })
    .join("");

  el("heatmap").innerHTML =
    '<div class="heat"><table>' + head + body + "</table></div>";
}

function renderYearBars(data) {
  const rows = data.eoy_returns;
  if (!rows.length) {
    el("eoy").innerHTML = '<div class="empty">No completed years.</div>';
    return;
  }
  const peak = Math.max(...rows.map((r) => Math.abs(r.value)), 0.0001);
  el("eoy").innerHTML =
    '<div class="bars">' +
    rows
      .map((r) => {
        const width = (Math.abs(r.value) / peak) * 50;
        const positive = r.value >= 0;
        const style = positive
          ? "left:50%;width:" + width + "%;background:" + C.up
          : "right:50%;width:" + width + "%;background:" + C.down;
        return (
          '<div class="bar-row"><div class="key">' + r.year + "</div>" +
          '<div class="bar-track"><div class="bar-fill" style="' + style + '"></div></div>' +
          '<div class="v ' + cls(r.value) + '">' + pct(r.value) + "</div></div>"
        );
      })
      .join("") +
    "</div>";
}

function renderDrawdowns(data) {
  const rows = data.worst_drawdowns;
  if (!rows.length) {
    el("drawdowns").innerHTML = '<div class="empty">No drawdown: equity never fell below its peak.</div>';
    return;
  }
  el("drawdowns").innerHTML =
    "<table><thead><tr><th>peak</th><th>valley</th><th>recovered</th><th>depth</th><th>days</th></tr></thead><tbody>" +
    rows
      .map(
        (d) =>
          "<tr><td>" + day(d.start) + "</td><td>" + day(d.valley) + "</td><td>" +
          (d.end ? day(d.end) : "open") + '</td><td class="neg">' + pct(d.depth) +
          "</td><td>" + d.days + "</td></tr>"
      )
      .join("") +
    "</tbody></table>";
}

function renderFills(data) {
  if (!data.fills.length) {
    el("fills").innerHTML = '<div class="empty">No fills in this run.</div>';
    return;
  }
  el("fills").innerHTML =
    '<div class="scroll"><table><thead><tr><th>date</th><th>side</th><th>qty</th>' +
    "<th>fill</th><th>slip bps</th><th>fee</th></tr></thead><tbody>" +
    data.fills
      .map(
        (f) =>
          "<tr><td>" + day(f.iso) + '</td><td class="' +
          (f.side === "buy" ? "pos" : "neg") + '">' + f.side + "</td><td>" +
          f.quantity.toFixed(6) + "</td><td>" + money(f.price) + "</td><td>" +
          num(f.slippage_bps) + "</td><td>" + money(f.fee) + "</td></tr>"
      )
      .join("") +
    "</tbody></table></div>";
}

function renderCharts(data) {
  const base = data.metrics.initial_equity;
  // A "percent" price format labels the value as given, so these series carry
  // percent units (155.41), not fractions (1.5541). The drawdown curve already
  // arrives in percent from the server.
  const toReturn = (points) =>
    points.map((p) => ({ time: p.time, value: (p.value / base - 1) * 100 }));

  strategyLine.setData(toReturn(data.equity));
  controlLine.setData(data.control_equity ? toReturn(data.control_equity) : []);
  underwater.setData(data.drawdown);
  sharpeLine.setData(data.rolling_sharpe);
  candles.setData(data.price);

  const markers = data.fills.map((f) => ({
    time: f.time,
    position: f.side === "buy" ? "belowBar" : "aboveBar",
    color: f.side === "buy" ? C.up : C.down,
    shape: f.side === "buy" ? "arrowUp" : "arrowDown",
    text: f.side + " " + f.quantity.toFixed(4),
  }));
  if (markersPrimitive) {
    markersPrimitive.setMarkers(markers);
  } else {
    markersPrimitive = LWC.createSeriesMarkers(candles, markers);
  }

  charts.forEach((c) => c.timeScale().fitContent());
}

// -- data ------------------------------------------------------------------

function query() {
  return new URLSearchParams({
    dataset: el("dataset").value,
    strategy: el("strategy").value,
    cash: el("cash").value,
    instrument: el("instrument").value,
    venue: el("venue").value,
    half_spread_bps: el("half_spread_bps").value,
    depth_quote: el("depth_quote").value,
    zero_costs: el("zero_costs").checked ? "1" : "0",
    compare: el("compare").checked ? "1" : "0",
  }).toString();
}

async function run() {
  const button = el("run");
  button.disabled = true;
  setStatus("running", "busy");
  try {
    const response = await fetch("/api/run?" + query());
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);

    el("cost-description").textContent = data.cost_description;
    el("report").textContent = data.report;
    renderHeadline(data);
    renderMetrics(data);
    renderHeatmap(data);
    renderYearBars(data);
    renderDrawdowns(data);
    renderFills(data);
    renderCharts(data);
    setStatus("ok", "ok");
  } catch (err) {
    setStatus(String(err.message || err), "error");
  } finally {
    button.disabled = false;
  }
}

async function boot() {
  if (!LWC) {
    setStatus("chart library did not load", "error");
    return;
  }
  try {
    const response = await fetch("/api/datasets");
    const { datasets } = await response.json();
    el("dataset").innerHTML = datasets
      .map((d) => '<option value="' + d.file + '">' + d.label + "</option>")
      .join("");
    const preferred = datasets.findIndex((d) => d.file.includes("BTC-USD-1d-2023"));
    if (preferred >= 0) el("dataset").selectedIndex = preferred;
    await run();
  } catch (err) {
    setStatus("could not reach the server: " + err, "error");
  }
}

el("run").addEventListener("click", run);
["dataset", "strategy", "instrument", "venue", "zero_costs", "compare"].forEach((id) =>
  el(id).addEventListener("change", run)
);

boot();
