#!/usr/bin/env python3
"""Replay a completed backtest from ``trades.csv`` + ``equity.csv``.

Usage
-----
    python -m tools.termviz_replay --trades runs/latest/trades.csv --equity runs/latest/equity.csv
    python -m tools.termviz_replay --trades runs/latest/trades.csv --speed 5 --last 50

The tool animates each trade/equity row as if it were happening live,
rendering the same Rich TUI layout as ``termviz_live`` but driven from
CSV files instead of a JSONL tail.

Options
-------
--trades PATH   Path to trades.csv (required).
--equity PATH   Path to equity.csv (optional – computed from trades if missing).
--speed  N      Playback speed multiplier (default 1.0).
--last   N      Only replay the last *N* trades.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque
from pathlib import Path
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
# Sparkline & constants (shared with termviz_live)
# ---------------------------------------------------------------------------
MAX_SPARK = 200
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        SPARK_CHARS[min(int((v - lo) / span * (len(SPARK_CHARS) - 1)), len(SPARK_CHARS) - 1)]
        for v in recent
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ReplayState:
    def __init__(self) -> None:
        self.trades: deque[dict[str, Any]] = deque(maxlen=30)
        self.equity_series: deque[float] = deque(maxlen=MAX_SPARK)
        self.drawdown_series: deque[float] = deque(maxlen=MAX_SPARK)
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.exit_reasons: Counter[str] = Counter()
        self.progress_current = 0
        self.progress_total = 0

    def add_trade(self, row: dict[str, Any]) -> None:
        self.trade_count += 1
        self.trades.append(row)
        pnl = float(row.get("pnl") or row.get("pnl_net") or 0)
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        reason = str(row.get("reason_close") or row.get("reason") or "")
        if reason:
            self.exit_reasons[reason] += 1
        eq = row.get("equity_after")
        if eq is not None and str(eq).strip():
            eq_val = float(eq)
            self.equity_series.append(eq_val)
            self.peak_equity = max(self.peak_equity, eq_val)
            self.drawdown_series.append(self.peak_equity - eq_val)

    def add_equity_point(self, eq: float) -> None:
        self.equity_series.append(eq)
        self.peak_equity = max(self.peak_equity, eq)
        self.drawdown_series.append(self.peak_equity - eq)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def _render_header(st: ReplayState) -> Panel:
    pct = int(st.progress_current / st.progress_total * 100) if st.progress_total else 0
    bar_width = 40
    filled = int(pct / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    txt = Text()
    txt.append(f"  🎬 REPLAY  [{bar}] {pct}%  ", style="bold bright_white")
    txt.append(f"  Trade {st.progress_current}/{st.progress_total}", style="dim")
    return Panel(txt, border_style="bright_blue")


def _render_stats(st: ReplayState) -> Panel:
    wr = st.wins / st.trade_count * 100 if st.trade_count else 0
    txt = Text()
    txt.append(f"  Trades:   {st.trade_count:>6}\n", style="bold white")
    txt.append(f"  Wins:     {st.wins:>6}\n", style="bold green")
    txt.append(f"  Losses:   {st.losses:>6}\n", style="bold red")
    txt.append(f"  Win Rate: {wr:>5.1f}%\n", style="bold yellow")
    txt.append(f"  PnL:      {st.total_pnl:>+10.2f}\n", style="bold cyan")
    return Panel(txt, title="[bold]Statistics[/bold]", border_style="blue")


def _render_equity(st: ReplayState) -> Panel:
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


def _render_exit_reasons(st: ReplayState) -> Panel:
    top = st.exit_reasons.most_common(10)
    if not top:
        return Panel("[dim]No trades yet[/dim]", title="[bold]Exit Reasons[/bold]", border_style="yellow")
    mx = top[0][1]
    bar_w = 25
    lines: list[Text] = []
    for reason, cnt in top:
        bl = max(1, int(cnt / mx * bar_w))
        line = Text()
        line.append(f"  {reason:<24s} ", style="white")
        line.append("█" * bl, style="bright_yellow")
        line.append(f" {cnt}", style="dim")
        lines.append(line)
    return Panel(Group(*lines), title="[bold]Exit Reasons[/bold]", border_style="yellow")


def _render_trades(st: ReplayState) -> Panel:
    tbl = Table(show_header=True, header_style="bold cyan", expand=True, padding=(0, 1))
    tbl.add_column("Exit Time", width=20)
    tbl.add_column("Side", width=6)
    tbl.add_column("Entry", width=10, justify="right")
    tbl.add_column("Exit", width=10, justify="right")
    tbl.add_column("Size", width=8, justify="right")
    tbl.add_column("PnL", width=10, justify="right")
    tbl.add_column("R", width=6, justify="right")
    tbl.add_column("Reason", width=18)
    for row in reversed(list(st.trades)):
        pnl = float(row.get("pnl") or row.get("pnl_net") or 0)
        style = "green" if pnl >= 0 else "red"
        tbl.add_row(
            str(row.get("exit_time", ""))[:19],
            str(row.get("side", "")),
            str(row.get("entry_price", "")),
            str(row.get("exit_price", "")),
            str(row.get("size", "")),
            f"{pnl:>+.2f}",
            str(row.get("r_multiple", "")),
            str(row.get("reason_close") or row.get("reason", "")),
            style=style,
        )
    return Panel(tbl, title="[bold]Recent Trades[/bold]", border_style="cyan")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _build_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=8),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="equity", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="reasons", ratio=1),
        Layout(name="trades", ratio=2),
    )
    return layout


def _render(layout: Layout, st: ReplayState) -> None:
    layout["header"].update(_render_header(st))
    layout["stats"].update(_render_stats(st))
    layout["equity"].update(_render_equity(st))
    layout["reasons"].update(_render_exit_reasons(st))
    layout["trades"].update(_render_trades(st))


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------
def _read_trades_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _read_equity_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay backtest trades.csv as animated TUI")
    parser.add_argument("--trades", required=True, help="Path to trades.csv")
    parser.add_argument("--equity", default=None, help="Path to equity.csv (optional)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (higher = faster)")
    parser.add_argument("--last", type=int, default=None, help="Only replay last N trades")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found")
        sys.exit(1)

    rows = _read_trades_csv(trades_path)
    if args.last and args.last > 0:
        rows = rows[-args.last:]

    equity_rows: list[dict[str, str]] = []
    if args.equity:
        eq_path = Path(args.equity)
        if eq_path.exists():
            equity_rows = _read_equity_csv(eq_path)

    state = ReplayState()
    state.progress_total = len(rows)

    # Base delay per trade – adjusted by speed
    base_delay = 0.3 / max(0.01, args.speed)

    console = Console()
    layout = _build_layout()

    try:
        with Live(layout, console=console, refresh_per_second=10, screen=True):
            # Pre-load equity points (if provided separately)
            for eq_row in equity_rows:
                eq_val = eq_row.get("equity") or eq_row.get("equity_after")
                if eq_val is not None and str(eq_val).strip():
                    state.add_equity_point(float(eq_val))

            for idx, row in enumerate(rows):
                state.progress_current = idx + 1
                state.add_trade(row)
                _render(layout, state)
                time.sleep(base_delay)

            # Hold final frame
            _render(layout, state)
            time.sleep(2)

    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[bold]Replay finished.[/bold]")
        wr = state.wins / state.trade_count * 100 if state.trade_count else 0
        console.print(
            f"  Trades: {state.trade_count}  |  "
            f"Win Rate: {wr:.1f}%  |  "
            f"PnL: {state.total_pnl:+.2f}  |  "
            f"Max DD: {max(state.drawdown_series) if state.drawdown_series else 0:.2f}"
        )


if __name__ == "__main__":
    main()
