from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import compute_drawdown_series


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _save_or_placeholder(fig, ax, *, title: str, empty: bool, path: Path) -> None:
    ax.set_title(title)
    if empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def generate_plots(
    trades: Sequence[Mapping[str, Any]],
    equity: Sequence[Mapping[str, Any]],
    outdir: str | Path,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required for PNG backtest reports") from exc

    charts_dir = Path(outdir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, str] = {}

    equity_series = compute_drawdown_series(equity)
    eq_x = [point["idx"] for point in equity_series]
    eq_y = [float(point["equity"]) for point in equity_series]
    dd_y = [float(point["drawdown"]) for point in equity_series]
    pnl_values = [float(trade.get("pnl", 0.0) or 0.0) for trade in trades]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if eq_x:
        ax.plot(eq_x, eq_y)
        ax.set_xlabel("Step")
        ax.set_ylabel("Equity")
    _save_or_placeholder(
        fig,
        ax,
        title="Equity Curve",
        empty=not bool(eq_x),
        path=charts_dir / "equity_curve.png",
    )
    plt.close(fig)
    chart_paths["equity_curve"] = str(charts_dir / "equity_curve.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if eq_x:
        ax.plot(eq_x, dd_y)
        ax.set_xlabel("Step")
        ax.set_ylabel("Drawdown")
    _save_or_placeholder(
        fig,
        ax,
        title="Drawdown",
        empty=not bool(eq_x),
        path=charts_dir / "drawdown.png",
    )
    plt.close(fig)
    chart_paths["drawdown"] = str(charts_dir / "drawdown.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if pnl_values:
        ax.bar(range(1, len(pnl_values) + 1), pnl_values)
        ax.set_xlabel("Trade")
        ax.set_ylabel("PnL")
    _save_or_placeholder(
        fig,
        ax,
        title="PnL Per Trade",
        empty=not bool(pnl_values),
        path=charts_dir / "pnl_per_trade.png",
    )
    plt.close(fig)
    chart_paths["pnl_per_trade"] = str(charts_dir / "pnl_per_trade.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if pnl_values:
        bins = min(50, max(10, int(len(pnl_values) ** 0.5)))
        ax.hist(pnl_values, bins=bins)
        ax.set_xlabel("PnL")
        ax.set_ylabel("Frequency")
    _save_or_placeholder(
        fig,
        ax,
        title="PnL Histogram",
        empty=not bool(pnl_values),
        path=charts_dir / "pnl_hist.png",
    )
    plt.close(fig)
    chart_paths["pnl_hist"] = str(charts_dir / "pnl_hist.png")

    monthly: dict[str, float] = defaultdict(float)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    for trade in trades:
        dt = _parse_ts(trade.get("exit_ts"))
        if dt is None:
            continue
        key = dt.strftime("%Y-%m")
        monthly[key] += float(trade.get("pnl", 0.0) or 0.0)
        min_ts = dt if min_ts is None or dt < min_ts else min_ts
        max_ts = dt if max_ts is None or dt > max_ts else max_ts

    long_range = bool(min_ts and max_ts and (max_ts - min_ts).days >= 45)
    if long_range and len(monthly) >= 2:
        labels = sorted(monthly.keys())
        values = [monthly[label] for label in labels]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(labels, values)
        ax.set_title("PnL By Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("PnL")
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        path = charts_dir / "pnl_by_month.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        chart_paths["pnl_by_month"] = str(path)

    return chart_paths
