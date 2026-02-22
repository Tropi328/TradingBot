#!/usr/bin/env python3
"""Monte Carlo Terminal Viewer — real-time ASCII fan chart in the terminal.

Usage (standalone):
    python tools/termviz_mc.py \
        --json reports/live/monte_carlo.json \
        --refresh 1.0

The viewer polls the JSON file for changes (mtime check) and redraws the
fan chart + stats panel every refresh cycle.  Uses **plotext** for terminal-
native line charts with proper axes.

Dependencies: plotext (>=5.2), rich (optional — for coloured stats panel).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_json_safe(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    except Exception:
        return {}


def _fmt_pct(v: Any, scale_01: bool = True) -> str:
    """Format a value as percentage string."""
    if v is None:
        return "?"
    try:
        f = float(v)
        if scale_01 and f <= 1.0:
            f *= 100.0
        return f"{f:.1f}%"
    except (ValueError, TypeError):
        return "?"


def _fmt_num(v: Any, decimals: int = 0) -> str:
    if v is None:
        return "?"
    try:
        f = float(v)
        if decimals == 0:
            return f"{f:,.0f}"
        return f"{f:,.{decimals}f}"
    except (ValueError, TypeError):
        return "?"


# ---------------------------------------------------------------------------
# Stats summary (plain text, no Rich dependency)
# ---------------------------------------------------------------------------

def _build_stats_text(data: dict[str, Any]) -> str:
    """Build a compact multi-line stats summary from MC JSON data."""
    lines: list[str] = []

    trades = data.get("num_trades", "?")
    sims = data.get("num_simulations", "?")
    lines.append(f"  Trades: {trades}  |  Simulations: {sims}")

    ruin_dd = data.get("ruin_dd", 0.5)
    try:
        ruin_pct = f"{float(ruin_dd) * 100:.0f}" if float(ruin_dd) <= 1 else f"{float(ruin_dd):.0f}"
    except (ValueError, TypeError):
        ruin_pct = "50"
    prob_ruin = _fmt_pct(data.get("prob_ruin"))
    health = data.get("health_score")
    health_str = f"{float(health):.0%}" if health is not None else "?"
    lines.append(f"  P(ruin ≥ {ruin_pct}%) = {prob_ruin}  |  Health = {health_str}")

    eq_p5 = _fmt_num(data.get("equity_end_p5"))
    eq_p50 = _fmt_num(data.get("equity_end_p50"))
    eq_p95 = _fmt_num(data.get("equity_end_p95"))
    lines.append(f"  Equity  p5={eq_p5}  p50={eq_p50}  p95={eq_p95}")

    dd_p50 = _fmt_pct(data.get("max_dd_p50"))
    dd_p95 = _fmt_pct(data.get("max_dd_p95"))
    min_eq = _fmt_num(data.get("min_equity_p5"))
    lines.append(f"  Max DD  p50={dd_p50}  p95={dd_p95}  |  Min Equity p5={min_eq}")

    ret = _fmt_pct(data.get("median_return_pct"), scale_01=False)
    wr = _fmt_pct(data.get("input_win_rate"))
    pf = data.get("input_profit_factor")
    pf_str = f"{float(pf):.2f}" if pf is not None and float(pf) < 9999 else "∞"
    cl = data.get("max_consecutive_loss_p95", "?")
    lines.append(f"  Return={ret}  WR={wr}  PF={pf_str}  ConsecLoss p95={cl}")

    gen = data.get("generated_at", "")
    if gen:
        lines.append(f"  Generated: {gen}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fan chart rendering via plotext
# ---------------------------------------------------------------------------

def _draw_fan_chart(data: dict[str, Any], title: str) -> None:
    """Render MC equity fan chart in the terminal using plotext."""
    import plotext as plt

    step_pcts = data.get("step_percentiles")
    if not step_pcts or not isinstance(step_pcts, dict):
        # Fallback: no per-step data — show stats only
        plt.clear_figure()
        plt.title(f"{title} — waiting for step_percentiles data…")
        plt.show()
        return

    p5 = step_pcts.get("p5", [])
    p25 = step_pcts.get("p25", [])
    p50 = step_pcts.get("p50", [])
    p75 = step_pcts.get("p75", [])
    p95 = step_pcts.get("p95", [])

    if not p50:
        plt.clear_figure()
        plt.title(f"{title} — no data yet")
        plt.show()
        return

    n = len(p50)
    x = list(range(n))

    plt.clear_figure()
    plt.theme("dark")
    plt.plot_size(None, None)  # auto-fit terminal

    # Plot percentile bands (outer to inner)
    if p5:
        plt.plot(x, p5, label="p5", color="red")
    if p25:
        plt.plot(x, p25, label="p25", color=(180, 100, 100))
    plt.plot(x, p50, label="p50 (median)", color="magenta")
    if p75:
        plt.plot(x, p75, label="p75", color=(100, 180, 100))
    if p95:
        plt.plot(x, p95, label="p95", color="green")

    # Ruin threshold line
    starting_eq = data.get("starting_equity", 0)
    ruin_dd = data.get("ruin_dd", 0.5)
    if starting_eq and ruin_dd:
        ruin_eq = float(starting_eq) * (1.0 - float(ruin_dd))
        plt.hline(ruin_eq, color="red")

    # Starting equity reference
    if starting_eq:
        plt.hline(float(starting_eq), color=(100, 100, 100))

    # Labels
    prob_ruin = data.get("prob_ruin", 0)
    health = data.get("health_score", 0)
    trades = data.get("num_trades", 0)
    sims = data.get("num_simulations", 0)
    try:
        pr_str = f"{float(prob_ruin) * 100:.1f}%" if float(prob_ruin) <= 1 else f"{float(prob_ruin):.1f}%"
    except (ValueError, TypeError):
        pr_str = "?"
    try:
        h_str = f"{float(health):.0%}"
    except (ValueError, TypeError):
        h_str = "?"

    plt.title(f"{title}  —  {sims} paths × {trades} trades  |  P(ruin)={pr_str}  Health={h_str}")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")

    plt.show()


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def run_viewer(
    json_path: str | Path,
    refresh_seconds: float = 1.0,
    title: str = "IGNACY BOT — Monte Carlo",
) -> None:
    """Poll *json_path* and redraw the terminal chart on every change.

    Blocks until Ctrl+C.
    """
    import plotext as plt

    jsn = Path(json_path)
    last_mtime = 0.0
    update_count = 0
    _spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _spin_idx = 0

    # Initial "waiting" screen
    plt.clear_figure()
    plt.title(f"{title} — waiting for Monte Carlo data…")
    plt.show()

    last_data: dict[str, Any] = {}

    try:
        while True:
            mt = _get_mtime(jsn)
            _spin_idx = (_spin_idx + 1) % len(_spinner)

            if mt > last_mtime and mt > 0.0:
                last_mtime = mt
                data = _load_json_safe(jsn)
                if data:
                    last_data = data
                    update_count += 1

            if last_data:
                # Clear terminal for clean redraw
                plt.clear_terminal()

                # Draw the fan chart
                _draw_fan_chart(last_data, title)

                # Print stats below chart
                print()
                print(_build_stats_text(last_data))

                # Status badge
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"\n  {_spinner[_spin_idx]}  updates: {update_count}  |  polled: {now_str}  |  Ctrl+C to exit")
            else:
                plt.clear_terminal()
                plt.clear_figure()
                plt.title(f"{title} — waiting for data… {_spinner[_spin_idx]}")
                plt.show()
                print(f"\n  Watching: {jsn}")
                print(f"  {_spinner[_spin_idx]}  polled: {datetime.now().strftime('%H:%M:%S')}")

            time.sleep(refresh_seconds)

    except KeyboardInterrupt:
        plt.clear_terminal()
        print("\nMonte Carlo terminal viewer stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Terminal Viewer — ASCII fan chart in the terminal.",
    )
    parser.add_argument(
        "--json",
        default="reports/live/monte_carlo.json",
        help="Path to the MC summary JSON (default: reports/live/monte_carlo.json)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--title",
        default="IGNACY BOT — Monte Carlo",
        help="Title text for the chart",
    )
    args = parser.parse_args()

    run_viewer(
        json_path=args.json,
        refresh_seconds=args.refresh,
        title=args.title,
    )


if __name__ == "__main__":
    main()
