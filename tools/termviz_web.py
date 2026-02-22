#!/usr/bin/env python3
"""Web-based decision-trace dashboard served on localhost.

Usage
-----
    python -m tools.termviz_web                                   # defaults
    python -m tools.termviz_web --path logs/decision_trace.jsonl   # custom
    python -m tools.termviz_web --port 8777 --no-open              # options

Opens http://localhost:8777 with a live-updating dashboard that reads
the JSONL decision-trace file and displays:

  * Equity curve + drawdown (Chart.js)
  * Win-rate donut
  * Counters: decisions / accepted / rejected / fills
  * Top reject-reason bar chart
  * Recent decisions table (colour-coded)
  * Recent fills table
  * Score distribution histogram
  * Alarms (micro-loss streak, swap > 12 h)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state shared between reader-thread and HTTP handlers
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "decisions": [],
    "fills": [],
    "equity": [],
    "drawdown": [],
    "scores": [],
    "reject_reasons": {},
    "exit_reasons": {},
    "counters": {"decisions": 0, "accepted": 0, "rejected": 0, "fills": 0},
    "alarms": [],
    "last_update": 0,
}

MAX_RECENT = 200   # keep last N events in tables
MAX_CHART = 2000   # keep last N points in charts


def _ingest(event: dict[str, Any]) -> None:
    with _LOCK:
        etype = event.get("type")
        if etype == "decision":
            _STATE["counters"]["decisions"] += 1
            rr = event.get("reject_reason")
            if rr:
                _STATE["counters"]["rejected"] += 1
                _STATE["reject_reasons"][rr] = _STATE["reject_reasons"].get(rr, 0) + 1
            else:
                _STATE["counters"]["accepted"] += 1
            sc = event.get("score")
            if sc is not None:
                _STATE["scores"].append(float(sc))
                if len(_STATE["scores"]) > MAX_CHART:
                    _STATE["scores"] = _STATE["scores"][-MAX_CHART:]
            _STATE["decisions"].append(event)
            if len(_STATE["decisions"]) > MAX_RECENT:
                _STATE["decisions"] = _STATE["decisions"][-MAX_RECENT:]

        elif etype == "fill":
            _STATE["counters"]["fills"] += 1
            eq = event.get("equity_after")
            if eq is not None:
                _STATE["equity"].append(float(eq))
                peak = max(_STATE["equity"]) if _STATE["equity"] else 0
                dd = peak - float(eq)
                _STATE["drawdown"].append(dd)
                if len(_STATE["equity"]) > MAX_CHART:
                    _STATE["equity"] = _STATE["equity"][-MAX_CHART:]
                    _STATE["drawdown"] = _STATE["drawdown"][-MAX_CHART:]
            rc = event.get("reason_close", "")
            if rc:
                _STATE["exit_reasons"][rc] = _STATE["exit_reasons"].get(rc, 0) + 1
            _STATE["fills"].append(event)
            if len(_STATE["fills"]) > MAX_RECENT:
                _STATE["fills"] = _STATE["fills"][-MAX_RECENT:]

            # alarms
            pnl = float(event.get("pnl") or 0)
            hm = float(event.get("holding_min") or 0)
            alarms = _STATE["alarms"]
            if hm > 720:
                alarms.append({"type": "SWAP_12H", "ts": event.get("ts"), "holding_min": hm})
            if len(alarms) > 50:
                _STATE["alarms"] = alarms[-50:]

        _STATE["last_update"] = time.time()


# ---------------------------------------------------------------------------
# File reader thread
# ---------------------------------------------------------------------------
def _read_loop(path: Path, stop: threading.Event) -> None:
    while not stop.is_set() and not path.exists():
        stop.wait(0.5)
    if stop.is_set():
        return
    with path.open("r", encoding="utf-8") as fh:
        while not stop.is_set():
            line = fh.readline()
            if not line:
                stop.wait(0.25)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                _ingest(json.loads(line))
            except json.JSONDecodeError:
                pass


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *a: Any) -> None:
        pass  # silence request logs

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._serve_json()
        elif self.path == "/health":
            self._serve_health()
        else:
            self._serve_html()

    def _serve_health(self) -> None:
        with _LOCK:
            decisions = len(_STATE.get("decisions", []))
            fills = len(_STATE.get("fills", []))
        body = json.dumps({
            "status": "ok",
            "decisions_count": decisions,
            "fills_count": fills,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self) -> None:
        with _LOCK:
            payload = json.dumps(_STATE, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_html(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))


# ---------------------------------------------------------------------------
# Embedded HTML / JS / CSS
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradingBot — Decision Trace Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --fg: #e6edf3; --dim: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --yellow: #d29922;
    --cyan: #39d2c0; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  .header { background: linear-gradient(90deg, #1a1e2e 0%, #0d1117 100%); padding: 16px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header h1 span { color: var(--blue); }
  .header .badge { background: var(--green); color: #000; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .header .ts { color: var(--dim); margin-left: auto; font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 16px 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .card h3 { font-size: 12px; text-transform: uppercase; color: var(--dim); margin-bottom: 8px; letter-spacing: 0.5px; }
  .stat-row { display: flex; gap: 16px; }
  .stat { flex: 1; text-align: center; }
  .stat .num { font-size: 28px; font-weight: 700; line-height: 1.2; }
  .stat .lbl { font-size: 11px; color: var(--dim); }
  .green { color: var(--green); }
  .red { color: var(--red); }
  .blue { color: var(--blue); }
  .cyan { color: var(--cyan); }
  .yellow { color: var(--yellow); }
  .wide { grid-column: span 2; }
  .full { grid-column: span 4; }
  .chart-wrap { position: relative; width: 100%; height: 220px; }
  .chart-wrap canvas { width: 100% !important; height: 100% !important; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--dim); font-weight: 500; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 5px 8px; border-bottom: 1px solid #21262d; white-space: nowrap; }
  tr.rej td { color: var(--red); opacity: 0.85; }
  tr.acc td { color: var(--green); }
  tr.fill-win td { color: var(--green); }
  tr.fill-loss td { color: var(--red); opacity: 0.85; }
  .alarm-box { background: #2d1b1b; border-left: 3px solid var(--red); padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 12px; }
  .alarm-box.warn { background: #2d2a1b; border-left-color: var(--yellow); }
  .no-alarm { color: var(--green); opacity: 0.7; }
  .tbl-scroll { max-height: 360px; overflow-y: auto; }
  @media (max-width: 1200px) { .grid { grid-template-columns: repeat(2, 1fr); } .wide { grid-column: span 2; } .full { grid-column: span 2; } }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } .wide, .full { grid-column: span 1; } }
</style>
</head>
<body>

<div class="header">
  <h1>📊 <span>TradingBot</span> Decision Trace</h1>
  <span class="badge" id="badge">LIVE</span>
  <span class="ts" id="headerTs">—</span>
</div>

<div class="grid">

  <!-- Row 1: Counters -->
  <div class="card">
    <h3>Decisions</h3>
    <div class="stat"><div class="num blue" id="cDecisions">0</div><div class="lbl">total bars evaluated</div></div>
  </div>
  <div class="card">
    <h3>Accepted</h3>
    <div class="stat"><div class="num green" id="cAccepted">0</div><div class="lbl">orders placed</div></div>
  </div>
  <div class="card">
    <h3>Rejected</h3>
    <div class="stat"><div class="num red" id="cRejected">0</div><div class="lbl">filtered out</div></div>
  </div>
  <div class="card">
    <h3>Fills</h3>
    <div class="stat"><div class="num cyan" id="cFills">0</div><div class="lbl">positions opened/closed</div></div>
  </div>

  <!-- Row 2: Equity + Reject reasons -->
  <div class="card wide">
    <h3>Equity Curve & Drawdown</h3>
    <div class="chart-wrap"><canvas id="chartEquity"></canvas></div>
  </div>
  <div class="card">
    <h3>Top Reject Reasons</h3>
    <div class="chart-wrap"><canvas id="chartRejects"></canvas></div>
  </div>
  <div class="card">
    <h3>Win / Loss &amp; Score</h3>
    <div style="display:flex;gap:8px;height:100%">
      <div style="flex:1;position:relative"><canvas id="chartWinLoss"></canvas></div>
      <div style="flex:1;position:relative"><canvas id="chartScore"></canvas></div>
    </div>
  </div>

  <!-- Row 3: Tables -->
  <div class="card wide">
    <h3>Recent Decisions</h3>
    <div class="tbl-scroll">
      <table><thead><tr>
        <th>Time</th><th>Symbol</th><th>Signal</th><th>Score</th><th>Spread</th><th>Size</th><th>Reject</th>
      </tr></thead><tbody id="tblDecisions"></tbody></table>
    </div>
  </div>
  <div class="card wide">
    <h3>Recent Fills</h3>
    <div class="tbl-scroll">
      <table><thead><tr>
        <th>Time</th><th>Symbol</th><th>Side</th><th>PnL</th><th>Equity</th><th>Reason</th><th>Hold (min)</th>
      </tr></thead><tbody id="tblFills"></tbody></table>
    </div>
  </div>

  <!-- Row 4: Alarms -->
  <div class="card full">
    <h3>Alarms</h3>
    <div id="alarmBox"><span class="no-alarm">✓ No alarms</span></div>
  </div>
</div>

<script>
// ------ Charts setup ------
const COLORS = { green: '#3fb950', red: '#f85149', blue: '#58a6ff', dim: '#8b949e',
                 cyan: '#39d2c0', yellow: '#d29922', purple: '#bc8cff' };

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';

const ctxEq = document.getElementById('chartEquity').getContext('2d');
const chartEquity = new Chart(ctxEq, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Equity', data: [], borderColor: COLORS.green, backgroundColor: 'rgba(63,185,80,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
      { label: 'Drawdown', data: [], borderColor: COLORS.red, backgroundColor: 'rgba(248,81,73,0.08)', fill: true, tension: 0.3, pointRadius: 0, borderWidth: 1, yAxisID: 'dd' },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } } },
    scales: {
      x: { display: false },
      y: { position: 'left', grid: { color: '#21262d' } },
      dd: { position: 'right', reverse: true, grid: { drawOnChartArea: false } }
    }
  }
});

const ctxRej = document.getElementById('chartRejects').getContext('2d');
const chartRejects = new Chart(ctxRej, {
  type: 'bar',
  data: { labels: [], datasets: [{ data: [], backgroundColor: COLORS.red, borderRadius: 3 }] },
  options: {
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { color: '#21262d' } }, y: { ticks: { font: { size: 10 } } } }
  }
});

const ctxWL = document.getElementById('chartWinLoss').getContext('2d');
const chartWinLoss = new Chart(ctxWL, {
  type: 'doughnut',
  data: { labels: ['Win', 'Loss'], datasets: [{ data: [0, 0], backgroundColor: [COLORS.green, COLORS.red], borderWidth: 0 }] },
  options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } } }
});

const ctxSc = document.getElementById('chartScore').getContext('2d');
const chartScore = new Chart(ctxSc, {
  type: 'bar',
  data: { labels: [], datasets: [{ data: [], backgroundColor: COLORS.purple, borderRadius: 2 }] },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { ticks: { font: { size: 9 } } }, y: { grid: { color: '#21262d' } } }
  }
});

// ------ Update loop ------
function fmt(v, d) { return v != null ? Number(v).toFixed(d || 2) : '—'; }

function updateDashboard(s) {
  // Counters
  document.getElementById('cDecisions').textContent = s.counters.decisions.toLocaleString();
  document.getElementById('cAccepted').textContent = s.counters.accepted.toLocaleString();
  document.getElementById('cRejected').textContent = s.counters.rejected.toLocaleString();
  document.getElementById('cFills').textContent = s.counters.fills.toLocaleString();

  // Header timestamp
  const lu = s.last_update;
  document.getElementById('headerTs').textContent = lu ? new Date(lu * 1000).toLocaleTimeString() : '—';

  // Equity chart
  const eqLabels = s.equity.map((_, i) => i);
  chartEquity.data.labels = eqLabels;
  chartEquity.data.datasets[0].data = s.equity;
  chartEquity.data.datasets[1].data = s.drawdown;
  chartEquity.update('none');

  // Reject reasons bar chart (top 12)
  const rr = Object.entries(s.reject_reasons).sort((a, b) => b[1] - a[1]).slice(0, 12);
  chartRejects.data.labels = rr.map(r => r[0]);
  chartRejects.data.datasets[0].data = rr.map(r => r[1]);
  chartRejects.update('none');

  // Win/Loss donut
  let wins = 0, losses = 0;
  s.fills.forEach(f => { (parseFloat(f.pnl || 0) >= 0) ? wins++ : losses++; });
  chartWinLoss.data.datasets[0].data = [wins, losses];
  chartWinLoss.update('none');

  // Score histogram
  if (s.scores.length > 0) {
    const bins = {};
    s.scores.forEach(sc => { const b = Math.floor(sc / 5) * 5; bins[b] = (bins[b] || 0) + 1; });
    const sorted = Object.entries(bins).sort((a, b) => Number(a[0]) - Number(b[0]));
    chartScore.data.labels = sorted.map(e => e[0]);
    chartScore.data.datasets[0].data = sorted.map(e => e[1]);
    chartScore.update('none');
  }

  // Decisions table
  const dTbl = document.getElementById('tblDecisions');
  dTbl.innerHTML = s.decisions.slice(-60).reverse().map(d => {
    const rr = d.reject_reason || '';
    const cls = rr ? 'rej' : 'acc';
    return `<tr class="${cls}">
      <td>${(d.ts || '').substring(0, 19)}</td>
      <td>${d.symbol || ''}</td>
      <td>${d.signal || '—'}</td>
      <td>${fmt(d.score)}</td>
      <td>${fmt(d.spread_points)}</td>
      <td>${fmt(d.size_final)}</td>
      <td>${rr || '✓ ACCEPTED'}</td>
    </tr>`;
  }).join('');

  // Fills table
  const fTbl = document.getElementById('tblFills');
  fTbl.innerHTML = s.fills.slice(-60).reverse().map(f => {
    const pnl = parseFloat(f.pnl || 0);
    const cls = pnl >= 0 ? 'fill-win' : 'fill-loss';
    return `<tr class="${cls}">
      <td>${(f.ts || '').substring(0, 19)}</td>
      <td>${f.symbol || ''}</td>
      <td>${f.side || ''}</td>
      <td>${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</td>
      <td>${fmt(f.equity_after)}</td>
      <td>${f.reason_close || ''}</td>
      <td>${fmt(f.holding_min, 1)}</td>
    </tr>`;
  }).join('');

  // Alarms
  const ab = document.getElementById('alarmBox');
  if (s.alarms.length === 0) {
    ab.innerHTML = '<span class="no-alarm">✓ No alarms</span>';
  } else {
    ab.innerHTML = s.alarms.slice(-10).reverse().map(a => {
      if (a.type === 'SWAP_12H') return `<div class="alarm-box warn">⚠ Swap &gt; 12h — holding ${fmt(a.holding_min, 0)} min at ${a.ts}</div>`;
      return `<div class="alarm-box">⚠ ${a.type} at ${a.ts}</div>`;
    }).join('');
  }
}

let prevUpdate = 0;
async function poll() {
  try {
    const resp = await fetch('/api/state');
    const data = await resp.json();
    if (data.last_update !== prevUpdate) {
      prevUpdate = data.last_update;
      updateDashboard(data);
      document.getElementById('badge').textContent = 'LIVE';
      document.getElementById('badge').style.background = '#3fb950';
    }
  } catch (e) {
    document.getElementById('badge').textContent = 'OFFLINE';
    document.getElementById('badge').style.background = '#f85149';
  }
}
setInterval(poll, 800);
poll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------
def _run_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_dashboard(
    path: str | Path = "logs/decision_trace.jsonl",
    port: int = 8777,
    open_browser: bool = True,
    blocking: bool = True,
) -> HTTPServer:
    """Start the web dashboard.  Returns the HTTPServer instance.

    If *blocking* is True (CLI usage), blocks until Ctrl+C.
    If *blocking* is False (programmatic usage), returns immediately.
    """
    trace_path = Path(path)
    stop = threading.Event()

    reader = threading.Thread(target=_read_loop, args=(trace_path, stop), daemon=True)
    reader.start()

    server = _run_server(port)
    url = f"http://localhost:{port}"
    print(f"\n  Dashboard → {url}\n  Trace     → {trace_path}\n")

    if open_browser:
        webbrowser.open(url)

    if blocking:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            server.shutdown()
            print("Dashboard stopped.")

    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Web dashboard for decision_trace.jsonl")
    parser.add_argument("--path", default="logs/decision_trace.jsonl", help="Path to JSONL file")
    parser.add_argument("--port", type=int, default=8777, help="Port to listen on")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    start_dashboard(
        path=args.path,
        port=args.port,
        open_browser=not args.no_open,
        blocking=True,
    )


if __name__ == "__main__":
    main()
