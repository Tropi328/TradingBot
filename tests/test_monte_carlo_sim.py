"""Tests for bot.backtest.monte_carlo simulation module."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bot.backtest.monte_carlo import (
    MonteCarloResult,
    _compute_equity_paths,
    _input_trade_stats,
    _max_consecutive_losses,
    _max_drawdowns,
    _render_chart,
    _write_json,
    run_monte_carlo_simulation,
    simulate,
)


# ---------------------------------------------------------------------------
# simulate() — core logic
# ---------------------------------------------------------------------------

class TestSimulate:
    """Tests for the core simulate function."""

    def test_empty_pnls_returns_defaults(self):
        result = simulate(pnls=[], starting_equity=10_000)
        assert result.num_trades == 0
        assert result.equity_end_p50 == 10_000
        assert result.equity_end_mean == 10_000
        assert result.prob_ruin == 0.0
        assert result.generated_at != ""

    def test_single_trade_positive(self):
        result = simulate(pnls=[100.0], starting_equity=10_000, num_simulations=500, seed=42)
        assert result.num_trades == 1
        assert result.num_simulations == 500
        assert result.equity_end_p50 == pytest.approx(10_100.0, abs=0.01)
        assert result.equity_end_mean == pytest.approx(10_100.0, abs=0.01)
        assert result.prob_ruin == 0.0

    def test_mixed_trades_stats(self):
        pnls = [50.0, -30.0, 80.0, -20.0, 60.0, -10.0, 40.0, -50.0, 70.0, -15.0]
        result = simulate(pnls=pnls, starting_equity=1_000, num_simulations=2_000, seed=123)
        assert result.num_trades == 10
        assert result.num_simulations == 2_000
        assert result.starting_equity == 1_000
        # Net PnL of input is +175, so median should be around 1175
        assert 900 < result.equity_end_p50 < 1400
        assert result.equity_end_p5 < result.equity_end_p50 < result.equity_end_p95
        assert 0 <= result.prob_ruin <= 1.0

    def test_large_losses_increase_ruin(self):
        """With heavy losses, probability of ruin should be non-trivial."""
        pnls = [-500.0, -400.0, 100.0, -300.0, 50.0]
        result = simulate(pnls=pnls, starting_equity=1_000, num_simulations=5_000, seed=7, ruin_dd_threshold=0.5)
        # Most paths should hit 50% DD given the heavy losses
        assert result.prob_ruin > 0.3

    def test_all_winning_zero_ruin(self):
        pnls = [100.0] * 20
        result = simulate(pnls=pnls, starting_equity=10_000, num_simulations=1_000, seed=99)
        assert result.prob_ruin == 0.0
        assert result.equity_end_p5 == pytest.approx(12_000.0, abs=0.01)
        assert result.max_dd_p95 == pytest.approx(0.0, abs=0.001)

    def test_seed_reproducibility(self):
        pnls = [10, -5, 20, -15, 30]
        r1 = simulate(pnls=pnls, starting_equity=1000, num_simulations=100, seed=42)
        r2 = simulate(pnls=pnls, starting_equity=1000, num_simulations=100, seed=42)
        assert r1.equity_end_p50 == r2.equity_end_p50
        assert r1.prob_ruin == r2.prob_ruin

    def test_iid_mode_regression_with_seed(self):
        pnls = [10, -5, 20, -15, 30]
        baseline = simulate(pnls=pnls, starting_equity=1000, num_simulations=500, seed=123)
        explicit_iid = simulate(
            pnls=pnls,
            starting_equity=1000,
            num_simulations=500,
            seed=123,
            sampling_mode="iid_bootstrap",
        )
        assert explicit_iid.equity_end_p50 == pytest.approx(baseline.equity_end_p50, abs=1e-9)
        assert explicit_iid.max_dd_p95 == pytest.approx(baseline.max_dd_p95, abs=1e-12)
        assert explicit_iid.prob_ruin == pytest.approx(baseline.prob_ruin, abs=1e-12)

    def test_block_bootstrap_produces_heavier_drawdown_tail(self):
        pnls = ([8.0] * 40) + ([-25.0] * 12) + ([8.0] * 40) + ([-20.0] * 10) + ([8.0] * 40)
        iid = simulate(
            pnls=pnls,
            starting_equity=1_000,
            num_simulations=3_000,
            seed=7,
            sampling_mode="iid_bootstrap",
        )
        mbb = simulate(
            pnls=pnls,
            starting_equity=1_000,
            num_simulations=3_000,
            seed=7,
            sampling_mode="moving_block_bootstrap",
            block_size=8,
        )
        assert mbb.max_dd_p95 > iid.max_dd_p95

    def test_ruin_floor_abs_counts_even_when_dd_threshold_not_hit(self):
        baseline = simulate(
            pnls=[-600.0],
            starting_equity=1_000,
            num_simulations=100,
            ruin_dd_threshold=0.80,
            seed=1,
        )
        with_floor = simulate(
            pnls=[-600.0],
            starting_equity=1_000,
            num_simulations=100,
            ruin_dd_threshold=0.80,
            ruin_equity_floor_abs=500.0,
            seed=1,
        )
        assert baseline.prob_ruin == pytest.approx(0.0, abs=1e-12)
        assert with_floor.prob_ruin == pytest.approx(1.0, abs=1e-12)

    def test_breakeven_policy_not_counted_as_loss_by_default(self):
        pnls = [0.0, 0.0, 0.0, 5.0]
        be_not_loss = simulate(
            pnls=pnls,
            starting_equity=1_000,
            num_simulations=1_000,
            seed=42,
            count_breakeven_as_loss=False,
        )
        be_is_loss = simulate(
            pnls=pnls,
            starting_equity=1_000,
            num_simulations=1_000,
            seed=42,
            count_breakeven_as_loss=True,
        )
        assert be_not_loss.max_consecutive_loss_p95 == 0
        assert be_is_loss.max_consecutive_loss_p95 > 0

    def test_equity_paths_shape(self):
        pnls = [10, -5, 20]
        result = simulate(pnls=pnls, starting_equity=100, num_simulations=50, seed=1)
        assert result.equity_paths is not None
        assert result.equity_paths.shape == (50, 4)  # 50 sims × (3 trades + 1 starting)
        # First column is always starting equity
        assert np.all(result.equity_paths[:, 0] == 100.0)

    def test_drawdown_percentile_ordering(self):
        pnls = [50, -100, 30, -80, 40, -60, 20, -40]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=2000, seed=55)
        assert result.max_dd_p5 <= result.max_dd_p25
        assert result.max_dd_p25 <= result.max_dd_p50
        assert result.max_dd_p50 <= result.max_dd_p75
        assert result.max_dd_p75 <= result.max_dd_p95

    # ── input validation ─────────────────────────────────────────────
    def test_negative_starting_equity_raises(self):
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            simulate(pnls=[10], starting_equity=-1)

    def test_zero_starting_equity_raises(self):
        with pytest.raises(ValueError, match="starting_equity must be > 0"):
            simulate(pnls=[10], starting_equity=0)

    def test_zero_simulations_raises(self):
        with pytest.raises(ValueError, match="num_simulations must be >= 1"):
            simulate(pnls=[10], starting_equity=1000, num_simulations=0)

    def test_ruin_dd_zero_raises(self):
        with pytest.raises(ValueError, match="ruin_dd_threshold must be in"):
            simulate(pnls=[10], starting_equity=1000, ruin_dd_threshold=0.0)

    def test_ruin_dd_above_one_raises(self):
        with pytest.raises(ValueError, match="ruin_dd_threshold must be in"):
            simulate(pnls=[10], starting_equity=1000, ruin_dd_threshold=1.5)

    # ── new metrics ──────────────────────────────────────────────────
    def test_min_equity_p5(self):
        pnls = [50, -100, 30, -80, 40]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=1000, seed=42)
        # Min equity should be below starting equity
        assert result.min_equity_p5 < result.starting_equity

    def test_median_return_pct(self):
        pnls = [100.0] * 10
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=500, seed=1)
        # All wins → return should be exactly 100%
        assert result.median_return_pct == pytest.approx(100.0, abs=0.1)

    def test_max_consecutive_loss_p95(self):
        pnls = [-10, -10, -10, -10, -10, -10, -10, -10, 100, 100]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=2000, seed=42)
        # With 80% losses we should see significant consecutive streaks
        assert result.max_consecutive_loss_p95 >= 3

    def test_input_trade_stats_populated(self):
        pnls = [100.0, -50.0, 80.0]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=100, seed=1)
        assert result.input_win_rate == pytest.approx(2 / 3, abs=0.01)
        assert result.input_avg_pnl == pytest.approx(130 / 3, abs=0.1)
        assert result.input_total_pnl == pytest.approx(130.0, abs=0.01)
        assert result.input_profit_factor > 0

    def test_generated_at_iso_format(self):
        result = simulate(pnls=[10], starting_equity=1000, num_simulations=10, seed=1)
        # Should be parseable ISO format
        from datetime import datetime
        dt = datetime.fromisoformat(result.generated_at)
        assert dt.year >= 2024


# ---------------------------------------------------------------------------
# _compute_equity_paths
# ---------------------------------------------------------------------------

class TestComputeEquityPaths:
    def test_shape(self):
        pnls = np.array([10.0, -5.0, 20.0])
        rng = np.random.default_rng(42)
        paths = _compute_equity_paths(pnls, starting_equity=100, num_simulations=10, rng=rng)
        assert paths.shape == (10, 4)

    def test_starting_equity(self):
        pnls = np.array([1.0, 2.0])
        rng = np.random.default_rng(1)
        paths = _compute_equity_paths(pnls, starting_equity=500, num_simulations=5, rng=rng)
        assert np.all(paths[:, 0] == 500.0)


# ---------------------------------------------------------------------------
# _max_drawdowns
# ---------------------------------------------------------------------------

class TestMaxDrawdowns:
    def test_no_drawdown(self):
        """Monotonically increasing equity → zero drawdown."""
        paths = np.array([[100, 110, 120, 130]])
        dd = _max_drawdowns(paths)
        assert dd[0] == pytest.approx(0.0, abs=1e-10)

    def test_full_drawdown(self):
        """Equity drops to zero → 100% drawdown."""
        paths = np.array([[100, 50, 0]])
        dd = _max_drawdowns(paths)
        assert dd[0] == pytest.approx(1.0, abs=1e-10)

    def test_partial_drawdown(self):
        paths = np.array([[1000, 800, 900, 700, 950]])
        dd = _max_drawdowns(paths)
        # Peak at 1000, trough at 700 → 30% DD
        assert dd[0] == pytest.approx(0.3, abs=1e-10)


# ---------------------------------------------------------------------------
# _max_consecutive_losses
# ---------------------------------------------------------------------------

class TestMaxConsecutiveLosses:
    def test_all_winners(self):
        pnls = np.array([10.0, 20.0, 30.0])
        rng = np.random.default_rng(1)
        streaks = _max_consecutive_losses(pnls, num_simulations=100, rng=rng)
        assert streaks.shape == (100,)
        assert np.all(streaks == 0)

    def test_all_losers(self):
        pnls = np.array([-10.0, -20.0, -30.0, -5.0])
        rng = np.random.default_rng(42)
        streaks = _max_consecutive_losses(pnls, num_simulations=100, rng=rng)
        # Every trade is a loss → max streak == num_trades
        assert np.all(streaks == 4)

    def test_mixed(self):
        pnls = np.array([10.0, -5.0, -5.0, 20.0, -5.0])
        rng = np.random.default_rng(7)
        streaks = _max_consecutive_losses(pnls, num_simulations=500, rng=rng)
        # Most paths should have at least 1 loss streak, but some may
        # randomly pick only winners → streaks >= 0 always holds
        assert np.all(streaks >= 0)
        assert np.all(streaks <= 5)  # can't exceed num trades
        # With 60% loss trades, most paths should have at least 1 streak
        assert np.mean(streaks >= 1) > 0.9

    def test_breakeven_loss_policy_toggle(self):
        pnls = np.array([0.0, 0.0, 5.0, -1.0])
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        streaks_without_be = _max_consecutive_losses(
            pnls,
            num_simulations=500,
            rng=rng_a,
            count_breakeven_as_loss=False,
        )
        streaks_with_be = _max_consecutive_losses(
            pnls,
            num_simulations=500,
            rng=rng_b,
            count_breakeven_as_loss=True,
        )
        assert np.percentile(streaks_with_be, 95) >= np.percentile(streaks_without_be, 95)


# ---------------------------------------------------------------------------
# _input_trade_stats
# ---------------------------------------------------------------------------

class TestInputTradeStats:
    def test_empty(self):
        stats = _input_trade_stats(np.array([]))
        assert stats["win_rate"] == 0.0
        assert stats["avg_pnl"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_all_winners(self):
        stats = _input_trade_stats(np.array([100.0, 200.0, 50.0]))
        assert stats["win_rate"] == 1.0
        assert stats["total_pnl"] == 350.0
        assert stats["profit_factor"] == float("inf")

    def test_all_losers(self):
        stats = _input_trade_stats(np.array([-100.0, -50.0]))
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == -150.0
        assert stats["profit_factor"] == 0.0

    def test_mixed(self):
        stats = _input_trade_stats(np.array([100.0, -50.0, 80.0, -30.0]))
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["total_pnl"] == pytest.approx(100.0)
        assert stats["avg_pnl"] == pytest.approx(25.0)
        # PF = 180 / 80 = 2.25
        assert stats["profit_factor"] == pytest.approx(2.25)


# ---------------------------------------------------------------------------
# MonteCarloResult.to_dict()
# ---------------------------------------------------------------------------

class TestMonteCarloResultToDict:
    def test_all_keys_present(self):
        result = MonteCarloResult(num_simulations=100, num_trades=10, starting_equity=1000)
        d = result.to_dict()
        expected_keys = {
            "num_simulations", "num_trades", "starting_equity",
            "equity_end_p5", "equity_end_p25", "equity_end_p50", "equity_end_p75", "equity_end_p95",
            "equity_end_mean",
            "max_dd_p5", "max_dd_p25", "max_dd_p50", "max_dd_p75", "max_dd_p95", "max_dd_mean",
            "ruin_dd", "prob_ruin",
            "min_equity_p5", "median_return_pct", "max_consecutive_loss_p95",
            "sampling_mode", "block_size", "equity_mode", "ruin_floor_pct", "ruin_floor_abs",
            "count_breakeven_as_loss",
            "input_win_rate", "input_avg_pnl", "input_total_pnl", "input_profit_factor",
            "generated_at", "health_score", "step_percentiles",
        }
        assert set(d.keys()) == expected_keys

    def test_no_numpy_arrays_in_dict(self):
        pnls = [10, -5, 20]
        result = simulate(pnls=pnls, starting_equity=100, num_simulations=10, seed=1)
        d = result.to_dict()
        serialised = json.dumps(d)  # must not raise
        assert isinstance(serialised, str)

    def test_new_metrics_in_dict(self):
        result = simulate(pnls=[100, -50, 80], starting_equity=1000, num_simulations=100, seed=1)
        d = result.to_dict()
        assert "min_equity_p5" in d
        assert "median_return_pct" in d
        assert "max_consecutive_loss_p95" in d
        assert "input_win_rate" in d
        assert "input_profit_factor" in d
        assert "generated_at" in d
        assert d["generated_at"] != ""


# ---------------------------------------------------------------------------
# File output — _render_chart and _write_json
# ---------------------------------------------------------------------------

class TestFileOutput:
    def test_write_json(self, tmp_path: Path):
        result = MonteCarloResult(
            num_simulations=100, num_trades=5, starting_equity=1000,
            equity_end_p50=1100, prob_ruin=0.05,
        )
        json_path = tmp_path / "mc.json"
        _write_json(result, json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["num_simulations"] == 100
        assert data["prob_ruin"] == 0.05

    def test_write_json_has_new_keys(self, tmp_path: Path):
        result = simulate(pnls=[100, -50, 80], starting_equity=1000, num_simulations=100, seed=1)
        json_path = tmp_path / "mc.json"
        _write_json(result, json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "min_equity_p5" in data
        assert "generated_at" in data
        assert "input_win_rate" in data

    def test_render_chart_creates_png(self, tmp_path: Path):
        pnls = [50, -30, 80, -20, 60, -10]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=50, seed=42)
        png_path = tmp_path / "mc.png"
        _render_chart(result, png_path, max_paths_plotted=10)
        assert png_path.exists()
        assert png_path.stat().st_size > 1000  # non-trivial PNG

    def test_render_chart_skips_empty(self, tmp_path: Path):
        result = MonteCarloResult()
        png_path = tmp_path / "mc.png"
        _render_chart(result, png_path)
        assert not png_path.exists()


# ---------------------------------------------------------------------------
# run_monte_carlo_simulation — high-level API
# ---------------------------------------------------------------------------

class TestRunMonteCarloSimulation:
    def test_creates_both_files(self, tmp_path: Path):
        pnls = [100, -50, 80, -30, 60]
        result = run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=5000,
            png_path=tmp_path / "live" / "mc.png",
            json_path=tmp_path / "live" / "mc.json",
            num_simulations=100,
            seed=42,
        )
        assert (tmp_path / "live" / "mc.png").exists()
        assert (tmp_path / "live" / "mc.json").exists()
        data = json.loads((tmp_path / "live" / "mc.json").read_text(encoding="utf-8"))
        assert data["num_trades"] == 5
        assert data["starting_equity"] == 5000.0

    def test_empty_pnls_no_output(self, tmp_path: Path):
        result = run_monte_carlo_simulation(
            trade_pnls=[],
            starting_equity=1000,
            png_path=tmp_path / "mc.png",
            json_path=tmp_path / "mc.json",
        )
        assert result.num_trades == 0
        assert not (tmp_path / "mc.png").exists()
        assert not (tmp_path / "mc.json").exists()

    def test_result_matches_json(self, tmp_path: Path):
        pnls = [20, -10, 30, -5, 15]
        result = run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=1000,
            png_path=tmp_path / "mc.png",
            json_path=tmp_path / "mc.json",
            num_simulations=200,
            seed=77,
        )
        data = json.loads((tmp_path / "mc.json").read_text(encoding="utf-8"))
        assert data["equity_end_p50"] == pytest.approx(result.equity_end_p50, abs=0.01)
        assert data["prob_ruin"] == pytest.approx(result.prob_ruin, abs=0.0001)

    def test_custom_ruin_threshold(self, tmp_path: Path):
        pnls = [-200, 50, -200, 50, -200]
        result = run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=1000,
            png_path=tmp_path / "mc.png",
            json_path=tmp_path / "mc.json",
            num_simulations=500,
            ruin_dd_threshold=0.30,
            seed=42,
        )
        # With 30% threshold and heavy losses, ruin prob should be high
        assert result.ruin_dd == 0.30
        assert result.prob_ruin > 0.5

    def test_equity_paths_freed_after_run(self, tmp_path: Path):
        """After run_monte_carlo_simulation, equity_paths should be None
        to release memory."""
        pnls = [100, -50, 80, -30, 60]
        result = run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=1000,
            png_path=tmp_path / "mc.png",
            json_path=tmp_path / "mc.json",
            num_simulations=100,
            seed=42,
        )
        assert result.equity_paths is None

    def test_json_contains_new_metrics(self, tmp_path: Path):
        pnls = [100, -50, 80, -30, 60]
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=1000,
            png_path=tmp_path / "mc.png",
            json_path=tmp_path / "mc.json",
            num_simulations=200,
            seed=42,
            sampling_mode="moving_block_bootstrap",
            block_size=8,
            equity_mode="initial",
            ruin_equity_floor_pct=0.2,
            count_breakeven_as_loss=False,
        )
        data = json.loads((tmp_path / "mc.json").read_text(encoding="utf-8"))
        assert "min_equity_p5" in data
        assert "median_return_pct" in data
        assert "max_consecutive_loss_p95" in data
        assert "input_win_rate" in data
        assert "input_profit_factor" in data
        assert "generated_at" in data
        assert data["sampling_mode"] == "moving_block_bootstrap"
        assert data["block_size"] == 8
        assert data["equity_mode"] == "initial"
        assert data["ruin_floor_pct"] == pytest.approx(0.2, abs=1e-9)
