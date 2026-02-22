"""Monte Carlo bootstrap simulation for backtest trade results.

Given a sequence of trade PnLs (or R-multiples) from a completed backtest,
this module:
  1. Resamples (bootstrap) N random equity paths.
  2. Computes percentile equity-end values (p5, p25, p50, p75, p95).
  3. Computes max-drawdown percentiles.
  4. Estimates probability of ruin (equity drops below a DD threshold).
  5. Computes input-trade descriptive stats (win_rate, avg_pnl, etc.).
  6. Generates a fan-style equity chart (PNG).
  7. Writes a JSON summary with all statistics + generation timestamp.

The viewer in ``tools/monte_carlo_live_viewer.py`` picks up these files
and displays them in real-time.

Public API
----------
- ``run_monte_carlo_simulation(...)`` — one-shot simulation + file output.
- ``MonteCarloResult`` — dataclass with all computed statistics.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

_SAMPLING_IID = "iid_bootstrap"
_SAMPLING_MBB = "moving_block_bootstrap"
_EQUITY_MODE_INITIAL = "initial"
_EQUITY_MODE_CURRENT = "current"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MonteCarloResult:
    """Statistics from a single Monte Carlo simulation run."""

    num_simulations: int = 0
    num_trades: int = 0
    starting_equity: float = 0.0

    # Equity-end percentiles
    equity_end_p5: float = 0.0
    equity_end_p25: float = 0.0
    equity_end_p50: float = 0.0
    equity_end_p75: float = 0.0
    equity_end_p95: float = 0.0
    equity_end_mean: float = 0.0

    # Max-drawdown percentiles (0–1 scale, fraction of peak)
    max_dd_p5: float = 0.0
    max_dd_p25: float = 0.0
    max_dd_p50: float = 0.0
    max_dd_p75: float = 0.0
    max_dd_p95: float = 0.0
    max_dd_mean: float = 0.0

    # Ruin estimation
    ruin_dd: float = 0.5  # drawdown threshold used
    prob_ruin: float = 0.0  # fraction of paths that hit ruin

    # Additional metrics
    min_equity_p5: float = 0.0   # 5th-percentile lowest equity reached
    median_return_pct: float = 0.0  # median total-return percentage
    max_consecutive_loss_p95: int = 0  # 95th-percentile longest loss streak

    # Simulation metadata
    sampling_mode: str = _SAMPLING_IID
    block_size: int = 8
    equity_mode: str = _EQUITY_MODE_INITIAL
    ruin_floor_pct: float | None = None
    ruin_floor_abs: float | None = None
    count_breakeven_as_loss: bool = False

    # Input-trade descriptive stats
    input_win_rate: float = 0.0
    input_avg_pnl: float = 0.0
    input_total_pnl: float = 0.0
    input_profit_factor: float = 0.0

    # Health score (0–1, higher = healthier)
    health_score: float = 0.0

    # Timestamp
    generated_at: str = ""

    # Per-step percentile arrays for terminal fan chart (JSON-serialisable)
    step_percentiles: dict[str, list[float]] = field(default_factory=dict)

    # Raw paths (for charting) — kept in memory only, never serialised
    equity_paths: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary (no numpy arrays)."""
        return {
            "num_simulations": self.num_simulations,
            "num_trades": self.num_trades,
            "starting_equity": round(self.starting_equity, 2),
            "equity_end_p5": round(self.equity_end_p5, 2),
            "equity_end_p25": round(self.equity_end_p25, 2),
            "equity_end_p50": round(self.equity_end_p50, 2),
            "equity_end_p75": round(self.equity_end_p75, 2),
            "equity_end_p95": round(self.equity_end_p95, 2),
            "equity_end_mean": round(self.equity_end_mean, 2),
            "max_dd_p5": round(self.max_dd_p5, 4),
            "max_dd_p25": round(self.max_dd_p25, 4),
            "max_dd_p50": round(self.max_dd_p50, 4),
            "max_dd_p75": round(self.max_dd_p75, 4),
            "max_dd_p95": round(self.max_dd_p95, 4),
            "max_dd_mean": round(self.max_dd_mean, 4),
            "ruin_dd": round(self.ruin_dd, 4),
            "prob_ruin": round(self.prob_ruin, 4),
            "min_equity_p5": round(self.min_equity_p5, 2),
            "median_return_pct": round(self.median_return_pct, 4),
            "max_consecutive_loss_p95": self.max_consecutive_loss_p95,
            "sampling_mode": self.sampling_mode,
            "block_size": int(self.block_size),
            "equity_mode": self.equity_mode,
            "ruin_floor_pct": (round(float(self.ruin_floor_pct), 6) if self.ruin_floor_pct is not None else None),
            "ruin_floor_abs": (round(float(self.ruin_floor_abs), 6) if self.ruin_floor_abs is not None else None),
            "count_breakeven_as_loss": bool(self.count_breakeven_as_loss),
            "input_win_rate": round(self.input_win_rate, 4),
            "input_avg_pnl": round(self.input_avg_pnl, 2),
            "input_total_pnl": round(self.input_total_pnl, 2),
            "input_profit_factor": round(self.input_profit_factor, 4),
            "health_score": round(self.health_score, 4),
            "generated_at": self.generated_at,
            "step_percentiles": self.step_percentiles,
        }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _compute_equity_paths(
    pnls: np.ndarray,
    starting_equity: float,
    num_simulations: int,
    rng: np.random.Generator,
    *,
    sampling_mode: str = _SAMPLING_IID,
    block_size: int = 8,
) -> np.ndarray:
    """Bootstrap-resample *pnls* to build *num_simulations* equity curves.

    Returns array of shape ``(num_simulations, len(pnls) + 1)`` where
    column 0 is the starting equity.
    """
    n_trades = len(pnls)
    indices = _sample_trade_indices(
        n_trades=n_trades,
        num_simulations=num_simulations,
        rng=rng,
        sampling_mode=sampling_mode,
        block_size=block_size,
    )
    sampled_pnls = pnls[indices]  # (N, T)

    # Build cumulative equity: col-0 = starting equity
    cum = np.cumsum(sampled_pnls, axis=1)
    equity = np.empty((num_simulations, n_trades + 1), dtype=np.float64)
    equity[:, 0] = starting_equity
    equity[:, 1:] = starting_equity + cum
    return equity


def _normalize_sampling_mode(sampling_mode: str) -> str:
    mode = str(sampling_mode or _SAMPLING_IID).strip().lower()
    if mode not in {_SAMPLING_IID, _SAMPLING_MBB}:
        raise ValueError(
            "sampling_mode must be 'iid_bootstrap' or 'moving_block_bootstrap', "
            f"got {sampling_mode!r}",
        )
    return mode


def _normalize_equity_mode(equity_mode: str) -> str:
    mode = str(equity_mode or _EQUITY_MODE_INITIAL).strip().lower()
    if mode not in {_EQUITY_MODE_INITIAL, _EQUITY_MODE_CURRENT}:
        raise ValueError("equity_mode must be 'initial' or 'current'")
    return mode


def _sample_trade_indices(
    *,
    n_trades: int,
    num_simulations: int,
    rng: np.random.Generator,
    sampling_mode: str,
    block_size: int,
) -> np.ndarray:
    mode = _normalize_sampling_mode(sampling_mode)
    if mode == _SAMPLING_IID:
        return _sample_trade_indices_iid(
            n_trades=n_trades,
            num_simulations=num_simulations,
            rng=rng,
        )
    return _sample_trade_indices_mbb(
        n_trades=n_trades,
        num_simulations=num_simulations,
        rng=rng,
        block_size=block_size,
    )


def _sample_trade_indices_iid(
    *,
    n_trades: int,
    num_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, n_trades, size=(num_simulations, n_trades))


def _sample_trade_indices_mbb(
    *,
    n_trades: int,
    num_simulations: int,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    if block_size < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    num_blocks = int(math.ceil(n_trades / float(block_size)))
    starts = rng.integers(0, n_trades, size=(num_simulations, num_blocks))
    offsets = np.arange(block_size, dtype=np.int64)
    indices = (starts[..., None] + offsets[None, None, :]) % n_trades
    return indices.reshape(num_simulations, num_blocks * block_size)[:, :n_trades]


def _max_drawdowns(equity_paths: np.ndarray) -> np.ndarray:
    """Compute per-path max drawdown as fraction of peak equity.

    Returns 1-D array of shape ``(num_simulations,)`` with values in [0, 1].
    """
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    # Avoid division by zero for degenerate paths
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdowns = (running_max - equity_paths) / safe_max
    return np.max(drawdowns, axis=1)


def _max_consecutive_losses(
    pnls: np.ndarray,
    num_simulations: int,
    rng: np.random.Generator,
    *,
    sampling_mode: str = _SAMPLING_IID,
    block_size: int = 8,
    count_breakeven_as_loss: bool = False,
) -> np.ndarray:
    """Compute the max consecutive losing-trade streak per simulation path.

    Returns 1-D array of shape ``(num_simulations,)`` with integer counts.
    """
    n_trades = len(pnls)
    indices = _sample_trade_indices(
        n_trades=n_trades,
        num_simulations=num_simulations,
        rng=rng,
        sampling_mode=sampling_mode,
        block_size=block_size,
    )
    sampled = pnls[indices]  # (N, T)
    if count_breakeven_as_loss:
        is_loss = (sampled <= 0).astype(np.int8)  # 1 = loss, 0 = win
    else:
        is_loss = (sampled < 0).astype(np.int8)  # 1 = loss, 0 = win/BE

    # Walk columns to find max run of 1s per row (vectorised row-wise)
    max_streak = np.zeros(num_simulations, dtype=np.int64)
    current = np.zeros(num_simulations, dtype=np.int64)
    for col in range(n_trades):
        current = np.where(is_loss[:, col] == 1, current + 1, 0)
        max_streak = np.maximum(max_streak, current)
    return max_streak


def _input_trade_stats(pnl_arr: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for the raw input trade PnLs."""
    n = len(pnl_arr)
    if n == 0:
        return {"win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0, "profit_factor": 0.0}
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr <= 0]
    total_wins = float(wins.sum()) if len(wins) else 0.0
    total_losses = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (total_wins / total_losses) if total_losses > 0 else (
        float("inf") if total_wins > 0 else 0.0
    )
    return {
        "win_rate": float(len(wins) / n),
        "avg_pnl": float(pnl_arr.mean()),
        "total_pnl": float(pnl_arr.sum()),
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Health score
# ---------------------------------------------------------------------------

def mc_health_score(
    result: MonteCarloResult,
    *,
    ruin_weight: float = 0.35,
    dd_weight: float = 0.25,
    pf_weight: float = 0.25,
    wr_weight: float = 0.15,
) -> float:
    """Compute a 0–1 health score from MC simulation results.

    Components
    ----------
    ruin  : ``1 - prob_ruin``        — lower ruin probability → higher score.
    dd    : ``1 - max_dd_p95``       — lower expected DD → higher score.
    pf    : ``min(PF / 2, 1)``      — PF ≥ 2 → perfect component.
    wr    : ``win_rate``             — higher win rate → higher score.

    Returns
    -------
    float in [0.0, 1.0].
    """
    ruin_c = max(0.0, 1.0 - result.prob_ruin)
    dd_c = max(0.0, 1.0 - result.max_dd_p95)
    pf_raw = result.input_profit_factor if result.input_profit_factor < 9999 else 3.0
    pf_c = min(max(pf_raw, 0.0) / 2.0, 1.0)
    wr_c = max(0.0, min(result.input_win_rate, 1.0))

    score = ruin_c * ruin_weight + dd_c * dd_weight + pf_c * pf_weight + wr_c * wr_weight
    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(
    pnls: Sequence[float],
    starting_equity: float = 10_000.0,
    num_simulations: int = 1_000,
    ruin_dd_threshold: float = 0.50,
    seed: int | None = None,
    *,
    sampling_mode: str = _SAMPLING_IID,
    block_size: int = 8,
    ruin_equity_floor_pct: float | None = None,
    ruin_equity_floor_abs: float | None = None,
    count_breakeven_as_loss: bool = False,
    equity_mode: str = _EQUITY_MODE_INITIAL,
) -> MonteCarloResult:
    """Run a bootstrap Monte Carlo simulation and return statistics.

    Parameters
    ----------
    pnls:
        Sequence of per-trade P&L values (in account currency).
    starting_equity:
        Starting equity for the simulation paths.  Must be > 0.
    num_simulations:
        How many random equity paths to generate.  Must be >= 1.
    ruin_dd_threshold:
        Drawdown fraction (0–1) that defines "ruin".  E.g. 0.5 = 50%.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    MonteCarloResult with all computed statistics + raw equity paths.

    Raises
    ------
    ValueError
        If starting_equity <= 0, num_simulations < 1, or
        ruin_dd_threshold is outside (0, 1].
    """
    # ── input validation ──────────────────────────────────────────────
    if starting_equity <= 0:
        raise ValueError(f"starting_equity must be > 0, got {starting_equity}")
    if num_simulations < 1:
        raise ValueError(f"num_simulations must be >= 1, got {num_simulations}")
    if not (0 < ruin_dd_threshold <= 1.0):
        raise ValueError(
            f"ruin_dd_threshold must be in (0, 1], got {ruin_dd_threshold}"
        )
    if ruin_equity_floor_pct is not None and not (0.0 <= float(ruin_equity_floor_pct) <= 1.0):
        raise ValueError("ruin_equity_floor_pct must be in [0, 1]")
    if ruin_equity_floor_abs is not None and float(ruin_equity_floor_abs) < 0:
        raise ValueError("ruin_equity_floor_abs must be >= 0")

    mode = _normalize_sampling_mode(sampling_mode)
    eq_mode = _normalize_equity_mode(equity_mode)
    block_size_norm = int(block_size)
    if mode == _SAMPLING_MBB and block_size_norm < 2:
        raise ValueError(f"block_size must be >= 2, got {block_size}")
    if mode == _SAMPLING_IID and block_size_norm < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    pnl_arr = np.asarray(pnls, dtype=np.float64)
    n_trades = len(pnl_arr)

    if n_trades == 0:
        return MonteCarloResult(
            num_simulations=num_simulations,
            num_trades=0,
            starting_equity=starting_equity,
            equity_end_p50=starting_equity,
            equity_end_mean=starting_equity,
            min_equity_p5=starting_equity,
            ruin_dd=ruin_dd_threshold,
            sampling_mode=mode,
            block_size=block_size_norm,
            equity_mode=eq_mode,
            ruin_floor_pct=(float(ruin_equity_floor_pct) if ruin_equity_floor_pct is not None else None),
            ruin_floor_abs=(float(ruin_equity_floor_abs) if ruin_equity_floor_abs is not None else None),
            count_breakeven_as_loss=bool(count_breakeven_as_loss),
            generated_at=now_iso,
        )

    # ── input-trade descriptive stats ─────────────────────────────────
    trade_stats = _input_trade_stats(pnl_arr)

    rng = np.random.default_rng(seed)
    equity_paths = _compute_equity_paths(
        pnl_arr,
        starting_equity,
        num_simulations,
        rng,
        sampling_mode=mode,
        block_size=block_size_norm,
    )

    # Ending equity = last column
    ends = equity_paths[:, -1]

    # Max drawdowns
    dd = _max_drawdowns(equity_paths)

    # Min equity reached per path → take 5th percentile (worst-case floor)
    min_per_path = np.min(equity_paths, axis=1)
    min_equity_p5 = float(np.percentile(min_per_path, 5))

    # Probability of ruin: DD threshold and optional equity floors
    ruin_by_dd = dd >= ruin_dd_threshold
    ruin_mask = ruin_by_dd.copy()
    if ruin_equity_floor_pct is not None:
        floor_pct_level = starting_equity * float(ruin_equity_floor_pct)
        ruin_mask = np.logical_or(ruin_mask, min_per_path <= floor_pct_level)
    if ruin_equity_floor_abs is not None:
        ruin_mask = np.logical_or(ruin_mask, min_per_path <= float(ruin_equity_floor_abs))
    prob_ruin = float(np.mean(ruin_mask))

    # Median total-return percentage
    median_return_pct = float(
        ((np.median(ends) - starting_equity) / starting_equity) * 100
    )

    # Max consecutive-loss streak (separate RNG fork for determinism)
    rng_streak = np.random.default_rng(seed)
    streaks = _max_consecutive_losses(
        pnl_arr,
        num_simulations,
        rng_streak,
        sampling_mode=mode,
        block_size=block_size_norm,
        count_breakeven_as_loss=bool(count_breakeven_as_loss),
    )
    max_consecutive_loss_p95 = int(np.percentile(streaks, 95))

    # Per-step percentiles for terminal fan chart
    step_pcts: dict[str, list[float]] = {}
    for label, q in [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]:
        arr = np.percentile(equity_paths, q, axis=0)
        step_pcts[label] = [round(float(v), 2) for v in arr]

    result = MonteCarloResult(
        num_simulations=num_simulations,
        num_trades=n_trades,
        starting_equity=starting_equity,
        equity_end_p5=float(np.percentile(ends, 5)),
        equity_end_p25=float(np.percentile(ends, 25)),
        equity_end_p50=float(np.percentile(ends, 50)),
        equity_end_p75=float(np.percentile(ends, 75)),
        equity_end_p95=float(np.percentile(ends, 95)),
        equity_end_mean=float(np.mean(ends)),
        max_dd_p5=float(np.percentile(dd, 5)),
        max_dd_p25=float(np.percentile(dd, 25)),
        max_dd_p50=float(np.percentile(dd, 50)),
        max_dd_p75=float(np.percentile(dd, 75)),
        max_dd_p95=float(np.percentile(dd, 95)),
        max_dd_mean=float(np.mean(dd)),
        ruin_dd=ruin_dd_threshold,
        prob_ruin=prob_ruin,
        min_equity_p5=min_equity_p5,
        median_return_pct=median_return_pct,
        max_consecutive_loss_p95=max_consecutive_loss_p95,
        sampling_mode=mode,
        block_size=block_size_norm,
        equity_mode=eq_mode,
        ruin_floor_pct=(float(ruin_equity_floor_pct) if ruin_equity_floor_pct is not None else None),
        ruin_floor_abs=(float(ruin_equity_floor_abs) if ruin_equity_floor_abs is not None else None),
        count_breakeven_as_loss=bool(count_breakeven_as_loss),
        input_win_rate=trade_stats["win_rate"],
        input_avg_pnl=trade_stats["avg_pnl"],
        input_total_pnl=trade_stats["total_pnl"],
        input_profit_factor=min(trade_stats["profit_factor"], 9999.99),
        generated_at=now_iso,
        step_percentiles=step_pcts,
        equity_paths=equity_paths,
    )
    # Compute health after construction so all fields are set
    result.health_score = mc_health_score(result)
    return result


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _render_chart(
    result: MonteCarloResult,
    png_path: Path,
    *,
    max_paths_plotted: int = 200,
    dpi: int = 120,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Render the Monte Carlo fan chart to *png_path*.

    Uses ``matplotlib.collections.LineCollection`` for individual paths
    instead of a Python loop — significantly faster for large path counts.
    """
    import matplotlib
    # _render_chart always saves to a file — use the non-interactive Agg
    # backend unconditionally.  The live viewer runs in a separate process
    # with its own TkAgg backend, so there is no conflict.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.collections import LineCollection

    equity = result.equity_paths
    if equity is None or equity.size == 0:
        return

    n_sims, n_steps = equity.shape
    x = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    # ── individual paths via LineCollection (vectorised) ──────────────
    plot_count = min(max_paths_plotted, n_sims)
    idx = np.linspace(0, n_sims - 1, plot_count, dtype=int)
    segments = np.empty((plot_count, n_steps, 2), dtype=np.float64)
    segments[:, :, 0] = x[np.newaxis, :]
    segments[:, :, 1] = equity[idx]
    lc = LineCollection(segments, colors="#6c7086", alpha=0.08, linewidths=0.5)
    ax.add_collection(lc)
    # Set data limits so autoscale works with LineCollection
    ax.set_xlim(0, n_steps - 1)

    # ── percentile bands (filled) ────────────────────────────────────
    p5 = np.percentile(equity, 5, axis=0)
    p25 = np.percentile(equity, 25, axis=0)
    p50 = np.percentile(equity, 50, axis=0)
    p75 = np.percentile(equity, 75, axis=0)
    p95 = np.percentile(equity, 95, axis=0)

    ax.fill_between(x, p5, p95, color="#89b4fa", alpha=0.15, label="5\u201395 %ile")
    ax.fill_between(x, p25, p75, color="#89b4fa", alpha=0.30, label="25\u201375 %ile")
    ax.plot(x, p50, color="#f5c2e7", linewidth=2.0, label="Median (p50)")
    ax.plot(x, p5, color="#f38ba8", linewidth=1.0, linestyle="--", alpha=0.7, label="p5")
    ax.plot(x, p95, color="#a6e3a1", linewidth=1.0, linestyle="--", alpha=0.7, label="p95")

    # Ruin threshold line
    ruin_equity = result.starting_equity * (1 - result.ruin_dd)
    ax.axhline(y=ruin_equity, color="#f38ba8", linewidth=1.2, linestyle=":", alpha=0.8,
               label=f"Ruin ({result.ruin_dd:.0%} DD)")
    if result.ruin_floor_pct is not None:
        floor_pct_equity = result.starting_equity * float(result.ruin_floor_pct)
        ax.axhline(
            y=floor_pct_equity,
            color="#fab387",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label=f"Ruin floor {float(result.ruin_floor_pct):.0%}",
        )
    if result.ruin_floor_abs is not None:
        ax.axhline(
            y=float(result.ruin_floor_abs),
            color="#f9e2af",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label=f"Ruin floor abs {float(result.ruin_floor_abs):,.0f}",
        )

    # Starting equity
    ax.axhline(y=result.starting_equity, color="#cdd6f4", linewidth=0.8, linestyle="-", alpha=0.3)

    # ── labels ───────────────────────────────────────────────────────
    ax.set_title(
        f"Monte Carlo Simulation \u2014 {result.num_simulations:,} paths \u00d7 {result.num_trades} trades",
        color="#cdd6f4", fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Trade #", color="#a6adc8", fontsize=10)
    ax.set_ylabel("Equity", color="#a6adc8", fontsize=10)

    # Comma formatting for Y-axis (e.g. 10,000 instead of 10000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # ── stats text box (two-column layout) ───────────────────────────
    pf_str = (
        f"{result.input_profit_factor:.2f}"
        if result.input_profit_factor < 9999
        else "\u221e"
    )
    stats_text = (
        f"P(ruin \u2265 {result.ruin_dd:.0%}) = {result.prob_ruin:.1%}  "
        f"Health = {result.health_score:.0%}\n"
        f"Equity p50 = {result.equity_end_p50:,.0f}  "
        f"p5 = {result.equity_end_p5:,.0f}  "
        f"p95 = {result.equity_end_p95:,.0f}\n"
        f"Max DD p50 = {result.max_dd_p50:.1%}  "
        f"p95 = {result.max_dd_p95:.1%}  "
        f"Min Eq p5 = {result.min_equity_p5:,.0f}\n"
        f"Return = {result.median_return_pct:+.1f}%  "
        f"WR = {result.input_win_rate:.0%}  "
        f"PF = {pf_str}  "
        f"Consec.Loss p95 = {result.max_consecutive_loss_p95}"
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        color="#cdd6f4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#313244", edgecolor="#45475a", alpha=0.9),
    )

    ax.legend(loc="lower right", fontsize=8, facecolor="#313244", edgecolor="#45475a",
              labelcolor="#cdd6f4", framealpha=0.9)
    ax.tick_params(colors="#6c7086")
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.grid(True, alpha=0.1, color="#6c7086")

    fig.tight_layout()

    # ── write atomically via temp file ───────────────────────────────
    # Close the fd immediately — savefig works with the path string,
    # so we don't need the descriptor open.  This avoids the previous
    # double-close bug if os.replace raised after os.close in the
    # success path.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=str(png_path.parent))
    os.close(tmp_fd)
    try:
        fig.savefig(tmp_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        os.replace(tmp_path, str(png_path))
    except BaseException:
        plt.close(fig)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _write_json(result: MonteCarloResult, json_path: Path) -> None:
    """Write the summary JSON atomically."""
    payload = result.to_dict()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(json_path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=True)
        os.replace(tmp_path, str(json_path))
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def run_monte_carlo_simulation(
    trade_pnls: Sequence[float],
    starting_equity: float,
    png_path: str | Path,
    json_path: str | Path,
    num_simulations: int = 1_000,
    ruin_dd_threshold: float = 0.50,
    seed: int | None = None,
    max_paths_plotted: int = 200,
    *,
    sampling_mode: str = _SAMPLING_IID,
    block_size: int = 8,
    equity_mode: str = _EQUITY_MODE_INITIAL,
    ruin_equity_floor_pct: float | None = None,
    ruin_equity_floor_abs: float | None = None,
    count_breakeven_as_loss: bool = False,
) -> MonteCarloResult:
    """Run full Monte Carlo pipeline: simulate → chart → JSON.

    Parameters
    ----------
    trade_pnls:
        Per-trade P&L values from the backtest ``trade_log``.
    starting_equity:
        Account equity at start of backtest.  Must be > 0.
    png_path / json_path:
        Output file locations (directories created automatically).
    num_simulations:
        Number of bootstrap equity paths.  Must be >= 1.
    ruin_dd_threshold:
        Drawdown fraction that counts as ruin (0–1).
    seed:
        Optional RNG seed.
    max_paths_plotted:
        Cap on how many individual paths to render (keeps chart fast).

    Returns
    -------
    MonteCarloResult
        The ``equity_paths`` array is released (set to ``None``) after the
        chart is rendered to free memory.
    """
    png_path = Path(png_path)
    json_path = Path(json_path)

    # Ensure output directories exist
    png_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    n_trades = len(trade_pnls) if hasattr(trade_pnls, "__len__") else 0
    sim_started = time.perf_counter()
    LOGGER.info(
        "Monte Carlo simulation: trades=%d paths=%d mode=%s block=%d equity_mode=%s "
        "ruin_dd=%.0f%% floor_pct=%s floor_abs=%s be_as_loss=%s equity=%.2f",
        n_trades,
        num_simulations,
        sampling_mode,
        int(block_size),
        equity_mode,
        ruin_dd_threshold * 100,
        ("None" if ruin_equity_floor_pct is None else f"{float(ruin_equity_floor_pct):.4f}"),
        ("None" if ruin_equity_floor_abs is None else f"{float(ruin_equity_floor_abs):.2f}"),
        bool(count_breakeven_as_loss),
        starting_equity,
    )

    result = simulate(
        pnls=trade_pnls,
        starting_equity=starting_equity,
        num_simulations=num_simulations,
        ruin_dd_threshold=ruin_dd_threshold,
        seed=seed,
        sampling_mode=sampling_mode,
        block_size=block_size,
        ruin_equity_floor_pct=ruin_equity_floor_pct,
        ruin_equity_floor_abs=ruin_equity_floor_abs,
        count_breakeven_as_loss=count_breakeven_as_loss,
        equity_mode=equity_mode,
    )

    if result.num_trades == 0:
        LOGGER.warning("Monte Carlo skipped — no trades to simulate.")
        return result

    _render_chart(result, png_path, max_paths_plotted=max_paths_plotted)

    # Free the potentially large equity_paths array now that the chart
    # has been rendered — avoids keeping megabytes alive for nothing.
    result.equity_paths = None

    _write_json(result, json_path)

    elapsed_ms = (time.perf_counter() - sim_started) * 1000.0

    LOGGER.info(
        "Monte Carlo done: elapsed=%.1fms P(ruin)=%.1f%% Eq-p50=%.0f maxDD-p95=%.1f%% "
        "Return=%.1f%% WR=%.0f%% ConsecLoss-p95=%d | %s / %s",
        elapsed_ms,
        result.prob_ruin * 100,
        result.equity_end_p50,
        result.max_dd_p95 * 100,
        result.median_return_pct,
        result.input_win_rate * 100,
        result.max_consecutive_loss_p95,
        png_path,
        json_path,
    )
    return result


# ---------------------------------------------------------------------------
# Adaptive Monte-Carlo model
# ---------------------------------------------------------------------------

class MCAdaptiveModel:
    """Real-time MC-based risk scaling model.

    Feed it trade PnLs as they occur; periodically it re-runs MC
    simulation, computes a health score, and derives a ``risk_multiplier``
    that can be used to scale ``risk_per_trade``.

    The model also writes updated MC chart (PNG) and JSON files so the
    Monte Carlo live viewer can display progress in real time.
    """

    def __init__(
        self,
        *,
        min_trades: int = 15,
        resim_interval: int = 5,
        num_simulations: int = 500,
        num_simulations_online: int | None = None,
        ruin_dd_threshold: float = 0.50,
        chart_interval: int = 10,
        max_paths_plotted: int = 200,
        seed: int | None = None,
        ruin_weight: float = 0.35,
        dd_weight: float = 0.25,
        pf_weight: float = 0.25,
        wr_weight: float = 0.15,
        full_risk_health: float = 0.70,
        min_risk_health: float = 0.35,
        floor_multiplier: float = 0.25,
        health_ema_alpha: float = 0.25,
        max_step_up: float = 0.05,
        max_step_down: float = 0.10,
        sampling_mode: str = _SAMPLING_IID,
        block_size: int = 8,
        ruin_equity_floor_pct: float | None = None,
        ruin_equity_floor_abs: float | None = None,
        count_breakeven_as_loss: bool = False,
        equity_mode: str = _EQUITY_MODE_CURRENT,
        png_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> None:
        self._min_trades = max(5, min_trades)
        self._resim_interval = max(1, resim_interval)
        _online = num_simulations_online if num_simulations_online is not None else num_simulations
        self._num_simulations = max(1, int(_online))
        self._ruin_dd = ruin_dd_threshold
        self._chart_interval = max(1, chart_interval)
        self._max_paths_plotted = max_paths_plotted
        self._seed = seed

        # Health score weights
        self._ruin_w = ruin_weight
        self._dd_w = dd_weight
        self._pf_w = pf_weight
        self._wr_w = wr_weight

        # Risk multiplier mapping
        self._full_health = full_risk_health
        self._min_health = min_risk_health
        self._floor_mult = floor_multiplier
        self._health_alpha = max(0.0, min(1.0, float(health_ema_alpha)))
        self._max_step_up = max(0.0, float(max_step_up))
        self._max_step_down = max(0.0, float(max_step_down))
        self._sampling_mode = _normalize_sampling_mode(sampling_mode)
        self._block_size = max(2, int(block_size))
        self._ruin_floor_pct = (float(ruin_equity_floor_pct) if ruin_equity_floor_pct is not None else None)
        self._ruin_floor_abs = (float(ruin_equity_floor_abs) if ruin_equity_floor_abs is not None else None)
        self._count_be_as_loss = bool(count_breakeven_as_loss)
        self._equity_mode = _normalize_equity_mode(equity_mode)

        # Output paths (None == no file output)
        self._png_path = Path(png_path) if png_path else None
        self._json_path = Path(json_path) if json_path else None

        # State
        self._pnls: list[float] = []
        self._trades_since_sim = 0
        self._trades_since_chart = 0
        self._health_ema: float | None = None
        self._initial_equity_anchor: float | None = None

        # Public read-only state
        self.health_score: float = 1.0
        self.risk_multiplier: float = 1.0
        self.last_result: MonteCarloResult | None = None

    # -- public API --------------------------------------------------------

    def add_trade(self, pnl: float) -> None:
        """Record a completed trade PnL."""
        self._pnls.append(float(pnl))
        self._trades_since_sim += 1
        self._trades_since_chart += 1

    def update(self, equity: float) -> float:
        """Re-evaluate model; return current ``risk_multiplier``.

        Triggers a lightweight MC re-simulation every ``resim_interval``
        trades (no chart) and a full simulation with chart render every
        ``chart_interval`` trades.  Both only fire once ``min_trades``
        are accumulated.
        """
        n = len(self._pnls)
        if n < self._min_trades:
            return self.risk_multiplier

        render_chart = self._trades_since_chart >= self._chart_interval
        run_sim = self._trades_since_sim >= self._resim_interval or render_chart

        if not run_sim:
            return self.risk_multiplier

        sim_equity = max(equity, 1.0)
        if self._equity_mode == _EQUITY_MODE_INITIAL:
            if self._initial_equity_anchor is None:
                self._initial_equity_anchor = sim_equity
            sim_equity = self._initial_equity_anchor

        # Run the simulation
        result = simulate(
            pnls=self._pnls,
            starting_equity=sim_equity,
            num_simulations=self._num_simulations,
            ruin_dd_threshold=self._ruin_dd,
            seed=self._seed,
            sampling_mode=self._sampling_mode,
            block_size=self._block_size,
            ruin_equity_floor_pct=self._ruin_floor_pct,
            ruin_equity_floor_abs=self._ruin_floor_abs,
            count_breakeven_as_loss=self._count_be_as_loss,
            equity_mode=self._equity_mode,
        )
        self._trades_since_sim = 0

        # Compute health score
        raw_health = mc_health_score(
            result,
            ruin_weight=self._ruin_w,
            dd_weight=self._dd_w,
            pf_weight=self._pf_w,
            wr_weight=self._wr_w,
        )
        if self._health_ema is None:
            self._health_ema = raw_health
        else:
            self._health_ema = (self._health_alpha * raw_health) + ((1.0 - self._health_alpha) * self._health_ema)
        self.health_score = max(0.0, min(self._health_ema, 1.0))
        result.health_score = self.health_score

        # Derive and smooth risk multiplier from health score.
        target_mult = self._score_to_multiplier(self.health_score)
        prev_mult = self.risk_multiplier
        delta = target_mult - prev_mult
        if delta > 0:
            delta = min(delta, self._max_step_up)
        elif delta < 0:
            delta = max(delta, -self._max_step_down)
        stepped_mult = prev_mult + delta
        self.risk_multiplier = max(self._floor_mult, min(1.0, stepped_mult))
        self.last_result = result

        # Write JSON on every re-sim so terminal viewer gets frequent updates.
        # Render the heavier PNG chart only every chart_interval trades.
        if self._json_path:
            try:
                self._json_path.parent.mkdir(parents=True, exist_ok=True)
                _write_json(result, self._json_path)
            except Exception:
                LOGGER.warning("MCAdaptiveModel: JSON write failed", exc_info=True)

        if render_chart and self._png_path:
            self._trades_since_chart = 0
            try:
                self._png_path.parent.mkdir(parents=True, exist_ok=True)
                _render_chart(result, self._png_path, max_paths_plotted=self._max_paths_plotted)
            except Exception:
                LOGGER.warning("MCAdaptiveModel: chart render failed", exc_info=True)

        result.equity_paths = None  # free memory

        LOGGER.debug(
            "MCAdaptiveModel: trades=%d health_raw=%.3f health_ema=%.3f mult_target=%.3f mult=%.3f "
            "ruin=%.2f%% dd95=%.1f%% pf=%.2f wr=%.0f%%",
            n,
            raw_health,
            self.health_score,
            target_mult,
            self.risk_multiplier,
            result.prob_ruin * 100, result.max_dd_p95 * 100,
            result.input_profit_factor, result.input_win_rate * 100,
        )
        return self.risk_multiplier

    # -- internals ---------------------------------------------------------

    def _score_to_multiplier(self, score: float) -> float:
        """Map health score to risk_multiplier.

        score >= full_risk_health  →  1.0
        score <= min_risk_health   →  floor_multiplier
        in between                 →  linear interpolation
        """
        if score >= self._full_health:
            return 1.0
        if score <= self._min_health:
            return self._floor_mult
        # Linear interpolation between [min_health, full_health] → [floor, 1.0]
        t = (score - self._min_health) / (self._full_health - self._min_health)
        return self._floor_mult + t * (1.0 - self._floor_mult)

    @classmethod
    def from_config(
        cls,
        mc_config: "MonteCarloConfig",
        *,
        png_path: str | Path | None = None,
        json_path: str | Path | None = None,
    ) -> "MCAdaptiveModel":
        """Create from a ``MonteCarloConfig`` (reads the ``adaptive`` sub-section)."""
        # Avoid circular import at module level
        from bot.config import MonteCarloConfig as _MC  # noqa: F811
        a = mc_config.adaptive
        return cls(
            min_trades=a.min_trades,
            resim_interval=a.resim_interval,
            num_simulations=a.num_simulations,
            num_simulations_online=a.num_simulations_online,
            ruin_dd_threshold=mc_config.ruin_dd_threshold,
            chart_interval=a.chart_interval,
            max_paths_plotted=mc_config.max_paths_plotted,
            seed=mc_config.seed,
            ruin_weight=a.ruin_weight,
            dd_weight=a.dd_weight,
            pf_weight=a.pf_weight,
            wr_weight=a.wr_weight,
            full_risk_health=a.full_risk_health,
            min_risk_health=a.min_risk_health,
            floor_multiplier=a.floor_multiplier,
            health_ema_alpha=a.health_ema_alpha,
            max_step_up=a.max_step_up,
            max_step_down=a.max_step_down,
            sampling_mode=mc_config.sampling_mode,
            block_size=mc_config.block_size,
            ruin_equity_floor_pct=mc_config.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc_config.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc_config.count_breakeven_as_loss,
            equity_mode=mc_config.equity_mode_adaptive,
            png_path=png_path,
            json_path=json_path,
        )
