#!/usr/bin/env python3
"""Monte Carlo Live Viewer — real-time desktop window showing MC simulation PNG.

Usage (standalone):
    python tools/monte_carlo_live_viewer.py \
        --png reports/live/monte_carlo.png \
        --json reports/live/monte_carlo.json \
        --refresh 1.0

The window refreshes when the PNG file changes (mtime check) or at most
every *refresh* seconds.  Key stats from the companion JSON are shown as
an overlay text or in the window title.

Dependencies: matplotlib, Pillow (PIL).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON summary parsing — importable utility for tests
# ---------------------------------------------------------------------------

def parse_mc_summary(data: dict[str, Any]) -> str:
    """Turn a Monte Carlo JSON blob into a compact one-line summary.

    Expected keys (all optional — missing values render as '?'):
        prob_ruin      – probability of ruin (0–1 float  *or* 0–100 percent)
        ruin_dd        – drawdown level used as ruin threshold (e.g. 0.5)
        equity_end_p5  – 5th-percentile ending equity
        equity_end_p50 – median ending equity
        equity_end_p95 – 95th-percentile ending equity
        max_dd_p95     – 95th-percentile maximum drawdown (0–1 *or* 0–100)

    Returns a human-readable summary string such as:
        "P(ruin>=50%)=12.4% | EqEnd p50=132.5 p5=88.1 p95=210.7 | maxDD p95=43.0%"
    """
    def _pct(v: Any, default: str = "?") -> str:
        if v is None:
            return default
        try:
            f = float(v)
        except (ValueError, TypeError):
            return default
        # auto-detect 0-1 vs 0-100 range
        if f <= 1.0:
            f *= 100.0
        return f"{f:.1f}%"

    def _num(v: Any, default: str = "?") -> str:
        if v is None:
            return default
        try:
            return f"{float(v):.1f}"
        except (ValueError, TypeError):
            return default

    ruin_dd = data.get("ruin_dd")
    ruin_dd_str = _num(ruin_dd, "50")
    # handle ruin_dd as fraction vs percent for display
    try:
        ruin_dd_f = float(ruin_dd) if ruin_dd is not None else 0.5
        if ruin_dd_f <= 1.0:
            ruin_dd_pct = f"{ruin_dd_f * 100:.0f}"
        else:
            ruin_dd_pct = f"{ruin_dd_f:.0f}"
    except (ValueError, TypeError):
        ruin_dd_pct = "50"

    prob_ruin = _pct(data.get("prob_ruin"), "?")

    eq_p5 = _num(data.get("equity_end_p5"))
    eq_p50 = _num(data.get("equity_end_p50"))
    eq_p95 = _num(data.get("equity_end_p95"))

    max_dd = _pct(data.get("max_dd_p95"))

    parts: list[str] = []
    parts.append(f"P(ruin>={ruin_dd_pct}%)={prob_ruin}")
    parts.append(f"EqEnd p50={eq_p50} p5={eq_p5} p95={eq_p95}")
    parts.append(f"maxDD p95={max_dd}")

    # Health score (0-1 from MCAdaptiveModel) when available
    health = data.get("health_score")
    if health is not None:
        try:
            h_val = float(health)
            parts.append(f"Health={h_val:.0%}")
        except (ValueError, TypeError):
            pass

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def _get_mtime(path: Path) -> float:
    """Return file mtime or 0.0 if the file does not exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _load_json_safe(path: Path) -> dict[str, Any]:
    """Read and parse JSON; return empty dict on any failure."""
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    except Exception:
        return {}


def _load_image(path: Path):
    """Load a PNG into a numpy-compatible array via PIL."""
    from PIL import Image
    import numpy as np
    img = Image.open(path)
    return np.asarray(img)


def run_viewer(
    png_path: str | Path,
    json_path: str | Path,
    refresh_seconds: float = 1.0,
    window_title: str = "IGNACY BOT — Monte Carlo Live",
    show_stats_overlay: bool = True,
    max_fps: int = 2,
) -> None:
    """Open a matplotlib window and keep refreshing from *png_path*.

    Blocks until the user closes the window.
    """
    import matplotlib
    # Use TkAgg for interactive window on Windows; fallback gracefully
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import numpy as np

    png = Path(png_path)
    jsn = Path(json_path)

    min_interval = 1.0 / max(max_fps, 1)
    interval = max(refresh_seconds, min_interval)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title(window_title)  # type: ignore[union-attr]

    # "Waiting" placeholder
    ax.set_facecolor("#1a1a2e")
    fig.set_facecolor("#1a1a2e")
    wait_text = ax.text(
        0.5, 0.5, "Waiting for Monte Carlo data…\n(chart updates after every few trades)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=18, color="#e0e0e0",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    img_artist = None
    overlay_text = None
    status_text = None       # bottom-right live status indicator
    last_png_mtime = 0.0
    last_json_mtime = 0.0
    last_summary = ""
    update_count = 0         # track how many times the chart has refreshed
    _spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _spin_idx = 0

    def _update_overlay(summary: str) -> None:
        nonlocal overlay_text
        if not show_stats_overlay:
            return
        if overlay_text is not None:
            overlay_text.remove()
        overlay_text = ax.text(
            0.01, 0.99, summary,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000", alpha=0.65),
            zorder=10,
        )

    def _update_status(updated: bool) -> None:
        """Show a live status badge at the bottom-right so the user
        can see the viewer is alive and when the last data arrived."""
        nonlocal status_text, _spin_idx, update_count
        _spin_idx = (_spin_idx + 1) % len(_spinner)
        if updated:
            update_count += 1
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M:%S")
        badge = f"{_spinner[_spin_idx]}  updates: {update_count}  |  polled: {now_str}"
        if status_text is not None:
            try:
                status_text.remove()
            except Exception:
                pass
        status_text = ax.text(
            0.99, 0.01, badge,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#aaaaaa",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000", alpha=0.50),
            zorder=10,
        )

    try:
        while plt.fignum_exists(fig.number):
            png_mt = _get_mtime(png)
            json_mt = _get_mtime(jsn)

            need_redraw = False
            data_updated = False

            # Reload JSON if changed
            if json_mt > last_json_mtime:
                last_json_mtime = json_mt
                data = _load_json_safe(jsn)
                if data:
                    last_summary = parse_mc_summary(data)
                    title = f"{window_title} | {last_summary}"
                    fig.canvas.manager.set_window_title(title)  # type: ignore[union-attr]
                    need_redraw = True
                    data_updated = True

            # Reload PNG if changed
            if png_mt > last_png_mtime and png_mt > 0.0:
                last_png_mtime = png_mt
                try:
                    arr = _load_image(png)
                except Exception:
                    time.sleep(0.1)
                    try:
                        arr = _load_image(png)
                    except Exception:
                        arr = None

                if arr is not None:
                    # Remove wait text on first successful load
                    if wait_text is not None:
                        try:
                            wait_text.remove()
                        except Exception:
                            pass
                        wait_text = None  # type: ignore[assignment]

                    if img_artist is None:
                        ax.clear()
                        ax.set_xticks([])
                        ax.set_yticks([])
                        img_artist = ax.imshow(arr, aspect="auto")
                    else:
                        img_artist.set_data(arr)
                        img_artist.set_extent([0, arr.shape[1], arr.shape[0], 0])

                    _update_overlay(last_summary)
                    need_redraw = True
                    data_updated = True

            # Always update the status badge (spinner + timestamp)
            _update_status(data_updated)
            need_redraw = True

            if need_redraw:
                fig.canvas.draw_idle()

            fig.canvas.flush_events()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Live Viewer — watch MC simulation PNG in real time.",
    )
    parser.add_argument(
        "--png",
        default="reports/live/monte_carlo.png",
        help="Path to the MC simulation PNG (default: reports/live/monte_carlo.png)",
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
        default="IGNACY BOT — Monte Carlo Live",
        help="Window title base text",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable stats overlay text on the image",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        default=2,
        help="Maximum refresh rate in frames per second (default: 2)",
    )
    args = parser.parse_args()

    run_viewer(
        png_path=args.png,
        json_path=args.json,
        refresh_seconds=args.refresh,
        window_title=args.title,
        show_stats_overlay=not args.no_overlay,
        max_fps=args.max_fps,
    )


if __name__ == "__main__":
    main()
