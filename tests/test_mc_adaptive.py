"""Tests for MC health score, MCAdaptiveModel, and MCAdaptiveConfig."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bot.backtest.monte_carlo import (
    MCAdaptiveModel,
    MonteCarloResult,
    mc_health_score,
    simulate,
)
from bot.config import MCAdaptiveConfig, MonteCarloConfig


# ---------------------------------------------------------------------------
# mc_health_score()
# ---------------------------------------------------------------------------

class TestMcHealthScore:
    """Unit tests for mc_health_score()."""

    def test_perfect_result_returns_one(self):
        result = MonteCarloResult(
            prob_ruin=0.0,
            max_dd_p95=0.0,
            input_profit_factor=3.0,
            input_win_rate=1.0,
        )
        score = mc_health_score(result)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_terrible_result_returns_near_zero(self):
        result = MonteCarloResult(
            prob_ruin=1.0,
            max_dd_p95=1.0,
            input_profit_factor=0.0,
            input_win_rate=0.0,
        )
        score = mc_health_score(result)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_moderate_result_between_zero_and_one(self):
        result = MonteCarloResult(
            prob_ruin=0.10,
            max_dd_p95=0.30,
            input_profit_factor=1.5,
            input_win_rate=0.55,
        )
        score = mc_health_score(result)
        assert 0.3 < score < 0.9

    def test_clamps_to_zero_one_range(self):
        # Even with extreme values, result should be in [0, 1]
        result = MonteCarloResult(
            prob_ruin=2.0,  # invalid but shouldn't crash
            max_dd_p95=2.0,
            input_profit_factor=-5.0,
            input_win_rate=-1.0,
        )
        score = mc_health_score(result)
        assert 0.0 <= score <= 1.0

    def test_custom_weights(self):
        result = MonteCarloResult(
            prob_ruin=0.0,
            max_dd_p95=0.0,
            input_profit_factor=0.0,
            input_win_rate=0.0,
        )
        # With ruin=0 and all weight on ruin, score = 1.0 * 1.0 = 1.0
        score = mc_health_score(result, ruin_weight=1.0, dd_weight=0.0, pf_weight=0.0, wr_weight=0.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_default_weights_sum_to_one(self):
        """Verify default weights sum to 1.0 via a known calculation."""
        result = MonteCarloResult(
            prob_ruin=0.0,
            max_dd_p95=0.0,
            input_profit_factor=2.0,
            input_win_rate=1.0,
        )
        # ruin_c=1, dd_c=1, pf_c=1, wr_c=1 → score = 0.35+0.25+0.25+0.15 = 1.0
        score = mc_health_score(result)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_inf_profit_factor_capped(self):
        """Infinite PF (all wins) shouldn't break scoring."""
        result = MonteCarloResult(
            prob_ruin=0.0,
            max_dd_p95=0.0,
            input_profit_factor=float("inf"),
            input_win_rate=1.0,
        )
        score = mc_health_score(result)
        assert 0.0 <= score <= 1.0

    def test_health_score_in_simulate_result(self):
        """simulate() populates health_score on the result."""
        pnls = [50.0, -20.0, 80.0, -10.0, 30.0]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=200, seed=42)
        assert 0.0 <= result.health_score <= 1.0
        # Health score should match standalone calculation
        expected = mc_health_score(result)
        assert result.health_score == pytest.approx(expected, abs=0.01)

    def test_health_score_in_to_dict(self):
        """health_score appears in the serialised dict."""
        pnls = [50.0, -20.0, 30.0]
        result = simulate(pnls=pnls, starting_equity=1000, num_simulations=100, seed=1)
        d = result.to_dict()
        assert "health_score" in d
        assert isinstance(d["health_score"], float)
        assert 0.0 <= d["health_score"] <= 1.0


# ---------------------------------------------------------------------------
# MCAdaptiveConfig validation
# ---------------------------------------------------------------------------

class TestMCAdaptiveConfig:
    def test_defaults(self):
        cfg = MCAdaptiveConfig()
        assert cfg.enabled is False
        assert cfg.min_trades == 15
        assert cfg.resim_interval == 5
        assert cfg.floor_multiplier == 0.25
        assert cfg.num_simulations_online == 250
        assert cfg.health_ema_alpha == pytest.approx(0.25)
        assert cfg.max_step_up == pytest.approx(0.05)
        assert cfg.max_step_down == pytest.approx(0.10)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(ruin_weight=0.5, dd_weight=0.5, pf_weight=0.5, wr_weight=0.5)

    def test_min_trades_at_least_five(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(min_trades=2)

    def test_resim_interval_at_least_one(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(resim_interval=0)

    def test_floor_multiplier_positive(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(floor_multiplier=0.0)

    def test_health_thresholds_ordering(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(min_risk_health=0.8, full_risk_health=0.5)

    def test_health_ema_alpha_range(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(health_ema_alpha=0.0)

    def test_step_limits_range(self):
        with pytest.raises(Exception):
            MCAdaptiveConfig(max_step_up=0.0)
        with pytest.raises(Exception):
            MCAdaptiveConfig(max_step_down=0.0)

    def test_valid_custom_config(self):
        cfg = MCAdaptiveConfig(
            enabled=True,
            min_trades=20,
            resim_interval=3,
            num_simulations=1000,
            chart_interval=15,
            ruin_weight=0.4,
            dd_weight=0.2,
            pf_weight=0.3,
            wr_weight=0.1,
            full_risk_health=0.80,
            min_risk_health=0.30,
            floor_multiplier=0.10,
        )
        assert cfg.enabled is True
        assert cfg.num_simulations == 1000


class TestMonteCarloConfigValidation:
    def test_sampling_mode_validation(self):
        with pytest.raises(Exception):
            MonteCarloConfig(sampling_mode="bad_mode")

    def test_equity_mode_validation(self):
        with pytest.raises(Exception):
            MonteCarloConfig(equity_mode_backtest="bad")

    def test_ruin_floor_validation(self):
        with pytest.raises(Exception):
            MonteCarloConfig(ruin_equity_floor_pct=1.5)
        with pytest.raises(Exception):
            MonteCarloConfig(ruin_equity_floor_abs=-1.0)


# ---------------------------------------------------------------------------
# MCAdaptiveModel
# ---------------------------------------------------------------------------

class TestMCAdaptiveModel:
    """Tests for the MCAdaptiveModel risk-scaling class."""

    def _make_model(self, **kwargs) -> MCAdaptiveModel:
        defaults = dict(
            min_trades=5,
            resim_interval=3,
            num_simulations=100,
            chart_interval=100,  # large to avoid chart renders in tests
            seed=42,
        )
        defaults.update(kwargs)
        return MCAdaptiveModel(**defaults)

    def test_initial_state(self):
        model = self._make_model()
        assert model.health_score == 1.0
        assert model.risk_multiplier == 1.0
        assert model.last_result is None

    def test_add_trade_accumulates(self):
        model = self._make_model()
        model.add_trade(100.0)
        model.add_trade(-50.0)
        assert len(model._pnls) == 2

    def test_update_before_min_trades_noop(self):
        model = self._make_model(min_trades=10)
        for _ in range(5):
            model.add_trade(10.0)
        mult = model.update(1000.0)
        assert mult == 1.0
        assert model.last_result is None

    def test_update_triggers_simulation(self):
        model = self._make_model(min_trades=5, resim_interval=1)
        pnls = [50.0, -20.0, 80.0, -10.0, 30.0]
        for p in pnls:
            model.add_trade(p)
        mult = model.update(1000.0)
        assert model.last_result is not None
        assert 0.0 <= model.health_score <= 1.0
        assert 0.0 < mult <= 1.0

    def test_risk_multiplier_all_wins_near_one(self):
        model = self._make_model(min_trades=5, resim_interval=1)
        for _ in range(10):
            model.add_trade(100.0)
        model.update(10_000.0)
        assert model.risk_multiplier >= 0.9

    def test_risk_multiplier_heavy_losses_reduced(self):
        model = self._make_model(min_trades=5, resim_interval=1, max_step_down=1.0, health_ema_alpha=1.0)
        for _ in range(10):
            model.add_trade(-500.0)
        model.update(1000.0)
        assert model.risk_multiplier < 0.5

    def test_score_to_multiplier_above_full(self):
        model = self._make_model(full_risk_health=0.70)
        assert model._score_to_multiplier(0.80) == 1.0
        assert model._score_to_multiplier(0.70) == 1.0

    def test_score_to_multiplier_below_min(self):
        model = self._make_model(min_risk_health=0.35, floor_multiplier=0.25)
        assert model._score_to_multiplier(0.20) == 0.25
        assert model._score_to_multiplier(0.00) == 0.25
        assert model._score_to_multiplier(0.35) == 0.25

    def test_score_to_multiplier_midpoint(self):
        model = self._make_model(
            full_risk_health=0.70,
            min_risk_health=0.30,
            floor_multiplier=0.20,
        )
        # midpoint of [0.30, 0.70] = 0.50
        # t = (0.50 - 0.30) / (0.70 - 0.30) = 0.50
        # result = 0.20 + 0.50 * (1.0 - 0.20) = 0.20 + 0.40 = 0.60
        assert model._score_to_multiplier(0.50) == pytest.approx(0.60, abs=0.01)

    def test_resim_interval_respected(self):
        model = self._make_model(min_trades=5, resim_interval=3)
        for p in [10, -5, 20, -3, 15]:
            model.add_trade(p)
        # Only 5 trades, resim_interval=3, so first update should trigger
        model.update(1000.0)
        r1 = model.last_result
        assert r1 is not None

        # Add 1 more trade — not enough for resim (need 3)
        model.add_trade(10.0)
        model.update(1000.0)
        assert model.last_result is r1  # same object, no new sim

        # Add 2 more (total 3 since last sim) — should trigger
        model.add_trade(10.0)
        model.add_trade(10.0)
        model.update(1000.0)
        assert model.last_result is not r1  # new simulation

    def test_risk_multiplier_step_clamp(self):
        model = self._make_model(
            min_trades=5,
            resim_interval=1,
            max_step_up=0.03,
            max_step_down=0.04,
            health_ema_alpha=1.0,
        )
        for _ in range(10):
            model.add_trade(-400.0)
        start_mult = model.risk_multiplier
        model.update(1000.0)
        assert (start_mult - model.risk_multiplier) <= (0.04 + 1e-9)

    def test_online_update_runtime_soft_limit(self):
        model = self._make_model(
            min_trades=5,
            resim_interval=1,
            num_simulations_online=150,
            chart_interval=9999,
            health_ema_alpha=1.0,
        )
        for p in [20.0, -10.0, 15.0, -8.0, 25.0, -5.0, 10.0, -7.0]:
            model.add_trade(p)
        started = time.perf_counter()
        model.update(1000.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        assert elapsed_ms < 500.0

    def test_equity_mode_initial_uses_anchor_equity(self):
        model = self._make_model(
            min_trades=5,
            resim_interval=1,
            equity_mode="initial",
            health_ema_alpha=1.0,
            max_step_down=1.0,
        )
        for p in [50.0, -20.0, 10.0, -5.0, 8.0]:
            model.add_trade(p)
        model.update(1500.0)
        assert model.last_result is not None
        first_start = model.last_result.starting_equity
        model.add_trade(-15.0)
        model.update(800.0)
        assert model.last_result is not None
        assert model.last_result.starting_equity == pytest.approx(first_start, abs=1e-9)

    def test_chart_output(self, tmp_path: Path):
        png = tmp_path / "mc.png"
        jsn = tmp_path / "mc.json"
        model = self._make_model(
            min_trades=5,
            resim_interval=1,
            chart_interval=1,
            png_path=png,
            json_path=jsn,
        )
        for p in [50.0, -20.0, 80.0, -10.0, 30.0]:
            model.add_trade(p)
        model.update(1000.0)
        assert png.exists()
        assert jsn.exists()
        data = json.loads(jsn.read_text(encoding="utf-8"))
        assert "health_score" in data
        assert data["health_score"] > 0

    def test_no_chart_without_paths(self, tmp_path: Path):
        model = self._make_model(min_trades=5, resim_interval=1, chart_interval=1)
        for p in [50.0, -20.0, 80.0, -10.0, 30.0]:
            model.add_trade(p)
        model.update(1000.0)
        # Should not crash; no files created
        assert model.last_result is not None

    def test_from_config(self):
        mc_cfg = MonteCarloConfig(
            enabled=True,
            num_simulations=500,
            ruin_dd_threshold=0.40,
            seed=99,
            adaptive=MCAdaptiveConfig(
                enabled=True,
                min_trades=10,
                resim_interval=4,
                num_simulations_online=180,
                floor_multiplier=0.30,
            ),
        )
        model = MCAdaptiveModel.from_config(mc_cfg, png_path="/tmp/mc.png", json_path="/tmp/mc.json")
        assert model._min_trades == 10
        assert model._resim_interval == 4
        assert model._floor_mult == 0.30
        assert model._ruin_dd == 0.40
        assert model._seed == 99
        assert model._num_simulations == 180


# ---------------------------------------------------------------------------
# Viewer health_score integration
# ---------------------------------------------------------------------------

class TestViewerHealthScore:
    def test_health_score_shown_in_summary(self):
        from tools.monte_carlo_live_viewer import parse_mc_summary
        data = {
            "prob_ruin": 0.10,
            "ruin_dd": 0.50,
            "equity_end_p5": 800,
            "equity_end_p50": 1200,
            "equity_end_p95": 1800,
            "max_dd_p95": 0.25,
            "health_score": 0.72,
        }
        result = parse_mc_summary(data)
        assert "Health=72%" in result

    def test_no_health_score_no_crash(self):
        from tools.monte_carlo_live_viewer import parse_mc_summary
        data = {"prob_ruin": 0.10, "ruin_dd": 0.50}
        result = parse_mc_summary(data)
        assert "Health" not in result

    def test_health_score_bad_value(self):
        from tools.monte_carlo_live_viewer import parse_mc_summary
        data = {"health_score": "bad", "prob_ruin": 0.1, "ruin_dd": 0.5}
        result = parse_mc_summary(data)
        # Should degrade gracefully — no Health shown
        assert "Health" not in result
