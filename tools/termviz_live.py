#!/usr/bin/env python3
"""Live Rich TUI that tails ``logs/decision_trace.jsonl``.

Usage
-----
    python -m tools.termviz_live                       # default path
    python -m tools.termviz_live --path logs/decision_trace.jsonl
    python -m tools.termviz_live --refresh 0.5         # 500 ms refresh

Panels
------
* **Equity sparkline** – last 200 fill-events equity_after + drawdown.
* **Counters** – decisions / fills / rejects (rolling).
* **Top-10 reject reasons** – horizontal bar chart.
* **Recent decisions** – last 30 rows.
* **Recent fills** – last 30 rows.
* **Micro-loss alarm** – consecutive small-loss fills.
* **Swap > 12 h alarm** – fills with holding_min > 720.

Press **q** to quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from threading import Event, Thread
from typing import Any

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("ERROR: 'rich' is required.  Install with:  pip install rich")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SPARK = 200
MAX_RECENT = 30
MICRO_LOSS_THRESHOLD = -0.5   # pnl threshold for "micro-loss"
MICRO_LOSS_STREAK = 3         # consecutive micro-losses before alarm
SWAP_ALARM_MIN = 720.0        # holding_min threshold (12 hours)

SPARK_CHARS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------------------
# Sparkline helper
# ---------------------------------------------------------------------------
def _sparkline(values: list[float], width: int = 60) -> str:
    """Return a Unicode sparkline string for *values*."""
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi != lo else 1.0
    return "".join(SPARK_CHARS[min(int((v - lo) / span * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)] for v in recent)


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
class DashboardState:
    """Accumulates JSONL events and exposes data for rendering."""

    def __init__(self) -> None:
        self.decisions: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT)
        self.fills: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT)
        self.equity_series: deque[float] = deque(maxlen=MAX_SPARK)
        self.drawdown_series: deque[float] = deque(maxlen=MAX_SPARK)
        self.reject_reasons: Counter[str] = Counter()
        self.decision_count = 0
        self.fill_count = 0
        self.reject_count = 0
        self.accept_count = 0
        self._peak_equity = 0.0
        self._micro_loss_streak = 0
        self.micro_loss_alarm = False
        self.swap_alarm_fills: deque[dict[str, Any]] = deque(maxlen=10)

    # ---- ingest one parsed event ------------------------------------------
    def ingest(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "decision":
            self._ingest_decision(event)
        elif etype == "fill":
            self._ingest_fill(event)

    def _ingest_decision(self, ev: dict[str, Any]) -> None:
        self.decision_count += 1
        self.decisions.append(ev)
        rr = ev.get("reject_reason")
        if rr:
            self.reject_count += 1
            self.reject_reasons[rr] += 1
        else:
            self.accept_count += 1

    def _ingest_fill(self, ev: dict[str, Any]) -> None:
        self.fill_count += 1
        self.fills.append(ev)

        eq = ev.get("equity_after")
        if eq is not None:
            self.equity_series.append(float(eq))
            self._peak_equity = max(self._peak_equity, float(eq))
            dd = self._peak_equity - float(eq)
            self.drawdown_series.append(dd)

        pnl = ev.get("pnl", 0.0)
        if pnl is not None and float(pnl) < MICRO_LOSS_THRESHOLD:
            self._micro_loss_streak += 1
        else:
            self._micro_loss_streak = 0
        self.micro_loss_alarm = self._micro_loss_streak >= MICRO_LOSS_STREAK

        hm = ev.get("holding_min", 0.0)
        if hm is not None and float(hm) > SWAP_ALARM_MIN:
            self.swap_alarm_fills.append(ev)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def _render_counters(st: DashboardState) -> Panel:
    txt = Text()
    txt.append(f"  Decisions: {st.decision_count:>8,}\n", style="bold white")
    txt.append(f"  Accepted:  {st.accept_count:>8,}\n", style="bold green")
    txt.append(f"  Rejected:  {st.reject_count:>8,}\n", style="bold red")
    txt.append(f"  Fills:     {st.fill_count:>8,}\n", style="bold cyan")
    return Panel(txt, title="[bold]Counters[/bold]", border_style="blue")


def _render_equity_spark(st: DashboardState) -> Panel:
    eq_line = _sparkline(list(st.equity_series))
    dd_line = _sparkline(list(st.drawdown_series))
    last_eq = f"{st.equity_series[-1]:,.2f}" if st.equity_series else "—"
    last_dd = f"{st.drawdown_series[-1]:,.2f}" if st.drawdown_series else "—"
    txt = Text()
    txt.append(f"  Equity   ({last_eq}): ", style="green")
    txt.append(eq_line + "\n", style="bright_green")
    txt.append(f"  Drawdown ({last_dd}): ", style="red")
    txt.append(dd_line, style="bright_red")
    return Panel(txt, title="[bold]Equity & Drawdown[/bold]", border_style="green")


def _render_reject_histogram(st: DashboardState) -> Panel:
    top10 = st.reject_reasons.most_common(10)
    if not top10:
        return Panel("[dim]No rejects yet[/dim]", title="[bold]Top Reject Reasons[/bold]", border_style="red")
    max_count = top10[0][1] if top10 else 1
    bar_width = 30
    lines: list[Text] = []
    for reason, cnt in top10:
        bar_len = max(1, int(cnt / max_count * bar_width))
        line = Text()
        line.append(f"  {reason:<30s} ", style="white")
        line.append("█" * bar_len, style="bright_red")
        line.append(f" {cnt:,}", style="dim")
        lines.append(line)
    group = Group(*lines)
    return Panel(group, title="[bold]Top Reject Reasons[/bold]", border_style="red")


def _render_recent_decisions(st: DashboardState) -> Panel:
    tbl = Table(show_header=True, header_style="bold magenta", expand=True, padding=(0, 1))
    tbl.add_column("Time", width=20)
    tbl.add_column("Symbol", width=8)
    tbl.add_column("Signal", width=7)
    tbl.add_column("Score", width=8, justify="right")
    tbl.add_column("Reject", width=28)
    tbl.add_column("Spread", width=8, justify="right")
    tbl.add_column("Size", width=8, justify="right")
    for ev in reversed(list(st.decisions)):
        rr = ev.get("reject_reason") or ""
        style = "red" if rr else "green"
        tbl.add_row(
            str(ev.get("ts", ""))[:19],
            str(ev.get("symbol", "")),
            str(ev.get("signal", "—")),
            f'{ev.get("score", ""):>8}',
            rr or "✓ ACCEPTED",
            f'{ev.get("spread_points", ""):>8}',
            f'{ev.get("size_final", ""):>8}',
            style=style,
        )
    return Panel(tbl, title="[bold]Recent Decisions[/bold]", border_style="magenta")


def _render_recent_fills(st: DashboardState) -> Panel:
    tbl = Table(show_header=True, header_style="bold cyan", expand=True, padding=(0, 1))
    tbl.add_column("Time", width=20)
    tbl.add_column("Symbol", width=8)
    tbl.add_column("Side", width=6)
    tbl.add_column("PnL", width=10, justify="right")
    tbl.add_column("Equity", width=12, justify="right")
    tbl.add_column("Reason", width=20)
    tbl.add_column("Hold(min)", width=10, justify="right")
    for ev in reversed(list(st.fills)):
        pnl = ev.get("pnl", 0)
        pnl_style = "green" if float(pnl or 0) >= 0 else "red"
        tbl.add_row(
            str(ev.get("ts", ""))[:19],
            str(ev.get("symbol", "")),
            str(ev.get("side", "")),
            f'{float(pnl or 0):>+.2f}',
            f'{float(ev.get("equity_after", 0)):>,.2f}',
            str(ev.get("reason_close", "")),
            f'{float(ev.get("holding_min", 0)):>.1f}',
            style=pnl_style,
        )
    return Panel(tbl, title="[bold]Recent Fills[/bold]", border_style="cyan")


def _render_alarms(st: DashboardState) -> Panel:
    txt = Text()
    if st.micro_loss_alarm:
        txt.append(f"  ⚠ MICRO-LOSS STREAK ({st._micro_loss_streak} consecutive)\n", style="bold bright_red")
    if st.swap_alarm_fills:
        txt.append(f"  ⚠ SWAP > 12h on {len(st.swap_alarm_fills)} fill(s)\n", style="bold yellow")
    if not txt.plain.strip():
        txt.append("  ✓ No alarms", style="dim green")
    return Panel(txt, title="[bold]Alarms[/bold]", border_style="yellow")


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------
def _build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=6),
        Layout(name="middle", ratio=1),
        Layout(name="bottom", ratio=1),
        Layout(name="footer", size=5),
    )
    layout["top"].split_row(
        Layout(name="counters", ratio=1),
        Layout(name="equity", ratio=2),
    )
    layout["middle"].split_row(
        Layout(name="rejects", ratio=1),
        Layout(name="decisions", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="alarms", ratio=1),
        Layout(name="fills", ratio=2),
    )
    return layout


def _render(layout: Layout, st: DashboardState) -> Layout:
    layout["header"].update(
        Panel(
            Align.center(Text("🔍  TradingBot Decision-Trace LIVE  🔍", style="bold bright_white")),
            border_style="bright_blue",
        )
    )
    layout["counters"].update(_render_counters(st))
    layout["equity"].update(_render_equity_spark(st))
    layout["rejects"].update(_render_reject_histogram(st))
    layout["decisions"].update(_render_recent_decisions(st))
    layout["alarms"].update(_render_alarms(st))
    layout["fills"].update(_render_recent_fills(st))
    layout["footer"].update(
        Panel(
            Align.center(Text("Press Ctrl+C to quit", style="dim")),
            border_style="dim",
        )
    )
    return layout


# ---------------------------------------------------------------------------
# File-tail thread
# ---------------------------------------------------------------------------
def _tail_file(path: Path, state: DashboardState, stop: Event) -> None:
    """Tail a JSONL file, ingesting new lines into *state*."""
    while not stop.is_set():
        if not path.exists():
            stop.wait(0.5)
            continue
        break
    if stop.is_set():
        return

    with path.open("r", encoding="utf-8") as fh:
        # Skip to end on startup
        fh.seek(0, 2)
        while not stop.is_set():
            line = fh.readline()
            if not line:
                stop.wait(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                state.ingest(event)
            except json.JSONDecodeError:
                pass


def _tail_file_from_start(path: Path, state: DashboardState, stop: Event) -> None:
    """Read all existing lines, then tail for new ones."""
    while not stop.is_set():
        if not path.exists():
            stop.wait(0.5)
            continue
        break
    if stop.is_set():
        return

    with path.open("r", encoding="utf-8") as fh:
        while not stop.is_set():
            line = fh.readline()
            if not line:
                stop.wait(0.2)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                state.ingest(event)
            except json.JSONDecodeError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Live TUI for decision_trace.jsonl")
    parser.add_argument("--path", default="logs/decision_trace.jsonl", help="Path to JSONL file")
    parser.add_argument("--refresh", type=float, default=0.4, help="Screen refresh interval (seconds)")
    parser.add_argument("--from-start", action="store_true", help="Read all existing lines first (not just tail)")
    args = parser.parse_args()

    trace_path = Path(args.path)
    state = DashboardState()
    stop_event = Event()

    tail_fn = _tail_file_from_start if args.from_start else _tail_file
    tail_thread = Thread(target=tail_fn, args=(trace_path, state, stop_event), daemon=True)
    tail_thread.start()

    console = Console()
    layout = _build_layout()

    try:
        with Live(layout, console=console, refresh_per_second=1.0 / args.refresh, screen=True):
            while True:
                _render(layout, state)
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        tail_thread.join(timeout=2)
        console.print("\n[bold]Stopped.[/bold]")


if __name__ == "__main__":
    main()
