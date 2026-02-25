"""Tests for signal_candidates, paper_costs and micro_loss_defense modules."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot.execution.micro_loss_defense import (
    MicroLossCheckResult,
    MicroLossDefenseConfig,
    MicroLossMetrics,
    check_edge_filter,
    check_min_sl_distance,
    is_micro_loss,
    run_micro_loss_checks,
)
from bot.execution.paper_costs import (
    FillPrices,
    PaperCostConfig,
    RoundtripCost,
    SlippageModelConfig,
    compute_be_offset,
    compute_fill_prices,
    compute_slippage,
    estimate_roundtrip_cost,
    estimate_roundtrip_cost_points,
)
from bot.monitoring.signal_candidates import (
    CandidateAggregation,
    SignalCandidate,
    SignalCandidateAggregator,
    SignalCandidateLogger,
    export_diagnostics,
    init_signal_candidates_table,
)


# =========================================================================
# paper_costs tests
# =========================================================================
class TestSlippage:
    def test_compute_slippage_basic(self):
        model = SlippageModelConfig(base_ticks=0.02, beta_spread=0.1, beta_atr=0.005)
        result = compute_slippage(model, spread=0.3, atr=2.0)
        expected = 0.02 + 0.1 * 0.3 + 0.005 * 2.0
        assert abs(result - expected) < 1e-9

    def test_compute_slippage_zero_spread(self):
        model = SlippageModelConfig(base_ticks=0.05, beta_spread=0.2, beta_atr=0.0)
        result = compute_slippage(model, spread=0.0, atr=None)
        assert abs(result - 0.05) < 1e-9

    def test_compute_slippage_no_atr(self):
        model = SlippageModelConfig(base_ticks=0.01, beta_spread=0.15, beta_atr=0.01)
        result = compute_slippage(model, spread=0.5, atr=None)
        expected = 0.01 + 0.15 * 0.5
        assert abs(result - expected) < 1e-9


class TestFillPrices:
    def test_long_fill_prices(self):
        cfg = PaperCostConfig()
        fills = compute_fill_prices(
            side="LONG",
            entry_price=2000.0,
            stop_price=1998.0,
            take_profit=2004.0,
            spread=0.3,
            atr=2.0,
            config=cfg,
        )
        assert fills.entry_fill > 2000.0  # ASK + slippage
        assert fills.exit_sl_fill < 1998.0  # BID - slippage
        assert fills.exit_tp_fill < 2004.0  # BID - slippage

    def test_short_fill_prices(self):
        cfg = PaperCostConfig()
        fills = compute_fill_prices(
            side="SHORT",
            entry_price=2000.0,
            stop_price=2002.0,
            take_profit=1996.0,
            spread=0.3,
            atr=2.0,
            config=cfg,
        )
        assert fills.entry_fill < 2000.0  # BID - slippage
        assert fills.exit_sl_fill > 2002.0  # ASK + slippage
        assert fills.exit_tp_fill > 1996.0  # ASK + slippage


class TestRoundtripCost:
    def test_estimate_roundtrip_cost(self):
        cfg = PaperCostConfig()
        cost = estimate_roundtrip_cost(spread=0.3, atr=2.0, size=0.1, config=cfg)
        assert cost.total > 0
        assert cost.spread_cost > 0
        assert cost.slippage_entry >= 0
        assert cost.slippage_exit >= 0

    def test_roundtrip_cost_points(self):
        cfg = PaperCostConfig()
        points = estimate_roundtrip_cost_points(spread=0.3, atr=2.0, config=cfg)
        assert points > 0.3  # at least the spread

    def test_roundtrip_cost_zero_spread(self):
        cfg = PaperCostConfig()
        cost = estimate_roundtrip_cost(spread=0.0, atr=None, size=1.0, config=cfg)
        assert cost.total >= 0


class TestBEOffset:
    def test_long_be_above_entry(self):
        cfg = PaperCostConfig()
        be = compute_be_offset(
            side="LONG", entry_price=2000.0, spread=0.3, atr=2.0,
            config=cfg, buffer_ticks=0.05,
        )
        assert be > 2000.0

    def test_short_be_below_entry(self):
        cfg = PaperCostConfig()
        be = compute_be_offset(
            side="SHORT", entry_price=2000.0, spread=0.3, atr=2.0,
            config=cfg, buffer_ticks=0.05,
        )
        assert be < 2000.0


# =========================================================================
# micro_loss_defense tests
# =========================================================================
class TestMinSLDistance:
    def test_sl_too_tight(self):
        cfg = MicroLossDefenseConfig(min_stop_spread_mult=5.0, min_stop_atr_mult=0.15)
        ok, min_req, reasons = check_min_sl_distance(
            sl_distance=0.5, spread=0.3, atr=10.0, config=cfg,
        )
        assert not ok
        assert "SL_TOO_TIGHT" in reasons
        assert min_req == max(5.0 * 0.3, 0.15 * 10.0)

    def test_sl_passes(self):
        cfg = MicroLossDefenseConfig(min_stop_spread_mult=5.0, min_stop_atr_mult=0.15)
        ok, min_req, reasons = check_min_sl_distance(
            sl_distance=2.0, spread=0.3, atr=5.0, config=cfg,
        )
        assert ok
        assert len(reasons) == 0


class TestEdgeFilter:
    def test_edge_too_low(self):
        cfg = MicroLossDefenseConfig(edge_mult=3.0)
        ok, ratio, reasons = check_edge_filter(
            expected_move=0.5, roundtrip_cost_points=0.4, config=cfg,
        )
        assert not ok
        assert "EDGE_TOO_LOW" in reasons
        assert abs(ratio - 0.5 / 0.4) < 1e-9

    def test_edge_passes(self):
        cfg = MicroLossDefenseConfig(edge_mult=3.0)
        ok, ratio, reasons = check_edge_filter(
            expected_move=2.0, roundtrip_cost_points=0.4, config=cfg,
        )
        assert ok
        assert ratio >= 3.0

    def test_zero_cost_passes(self):
        cfg = MicroLossDefenseConfig(edge_mult=3.0)
        ok, ratio, reasons = check_edge_filter(
            expected_move=1.0, roundtrip_cost_points=0.0, config=cfg,
        )
        assert ok


class TestMicroLossChecks:
    def test_disabled_always_passes(self):
        cfg = MicroLossDefenseConfig(enabled=False)
        result = run_micro_loss_checks(
            sl_distance=0.01, tp_distance=0.02, spread=1.0,
            atr=10.0, roundtrip_cost_points=5.0, config=cfg,
        )
        assert result.passed

    def test_both_fail(self):
        cfg = MicroLossDefenseConfig(
            enabled=True, min_stop_spread_mult=10.0,
            min_stop_atr_mult=0.5, edge_mult=10.0,
        )
        result = run_micro_loss_checks(
            sl_distance=0.5, tp_distance=1.0, spread=0.3,
            atr=10.0, roundtrip_cost_points=0.5, config=cfg,
        )
        assert not result.passed
        assert "SL_TOO_TIGHT" in result.rejection_reasons


class TestIsMicroLoss:
    def test_profitable_not_micro(self):
        assert not is_micro_loss(0.5, 0.3, k=1.5)

    def test_big_loss_not_micro(self):
        assert not is_micro_loss(-5.0, 0.3, k=1.5)

    def test_micro_loss(self):
        assert is_micro_loss(-0.3, 0.3, k=1.5)

    def test_boundary(self):
        # |pnl| just inside K * cost
        assert is_micro_loss(-0.44, 0.3, k=1.5)


class TestMicroLossMetrics:
    def test_record_close(self):
        m = MicroLossMetrics()
        is_ml = m.record_close(-0.1, 0.2, k=1.5, cause="BE", setup_name="SCALP")
        assert is_ml
        assert m.micro_loss_count == 1
        assert m.micro_loss_rate == 1.0
        assert m.causes["BE"] == 1
        is_ml2 = m.record_close(1.0, 0.2, k=1.5, cause="TP", setup_name="SCALP")
        assert not is_ml2
        assert m.micro_loss_rate == 0.5


# =========================================================================
# signal_candidates tests
# =========================================================================
@contextmanager
def _test_db():
    conn = sqlite3.connect(":memory:")
    try:
        conn.row_factory = sqlite3.Row
        init_signal_candidates_table(conn)
        yield conn
    finally:
        conn.close()


def _make_candidate(**overrides) -> SignalCandidate:
    defaults = dict(
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        symbol="XAUUSD",
        timeframe="M5",
        strategy_name="SCALP_ICT_PA",
        setup_name="fvg_retest",
        side="LONG",
        score=65.0,
        spread=0.3,
        atr=2.5,
        trend_regime="TRENDING",
        volatility_regime="NORMAL",
        session_name="LONDON",
        bias_direction="LONG",
        sl_distance=2.0,
        tp_distance=4.0,
        expected_rr=2.0,
        expected_move=4.0,
        estimated_roundtrip_cost=0.5,
        action="TRADE",
        accepted=True,
        rejection_reasons=[],
        score_breakdown={"edge": 40.0, "trigger": 20.0, "execution": 5.0},
    )
    defaults.update(overrides)
    return SignalCandidate(**defaults)


class TestSignalCandidateLogger:
    def test_log_single(self):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            candidate = _make_candidate()
            logger.log(candidate)
            rows = conn.execute("SELECT * FROM signal_candidates").fetchall()
            assert len(rows) == 1
            assert rows[0]["symbol"] == "XAUUSD"
            assert rows[0]["score"] == 65.0

    def test_log_many(self):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            candidates = [_make_candidate(score=s) for s in [50.0, 60.0, 70.0]]
            logger.log_many(candidates)
            count = conn.execute("SELECT COUNT(*) FROM signal_candidates").fetchone()[0]
            assert count == 3

    def test_rejection_reasons_stored(self):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            candidate = _make_candidate(
                action="OBSERVE",
                accepted=False,
                rejection_reasons=["SCORE_BELOW_MIN", "SL_TOO_TIGHT"],
            )
            logger.log(candidate)
            row = conn.execute("SELECT * FROM signal_candidates").fetchone()
            reasons = json.loads(row["rejection_reasons"])
            assert "SCORE_BELOW_MIN" in reasons
            assert "SL_TOO_TIGHT" in reasons


class TestSignalCandidateAggregator:
    def test_aggregate_empty(self):
        with _test_db() as conn:
            agg = SignalCandidateAggregator(conn)
            start = datetime(2025, 1, 1, tzinfo=timezone.utc)
            end = datetime(2025, 12, 31, tzinfo=timezone.utc)
            result = agg.aggregate_window(start, end)
            assert result.candidates_count == 0
            assert result.score_p50 is None

    def test_aggregate_with_data(self):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            candidates = [
                _make_candidate(score=50.0, action="OBSERVE", accepted=False, rejection_reasons=["LOW_SCORE"]),
                _make_candidate(score=60.0, action="SMALL", accepted=False, rejection_reasons=["COOLDOWN"]),
                _make_candidate(score=70.0, action="TRADE", accepted=True),
            ]
            logger.log_many(candidates)
            agg = SignalCandidateAggregator(conn)
            result = agg.aggregate_window(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            assert result.candidates_count == 3
            assert result.accepted_trades_count == 1
            assert result.score_p50 == 60.0
            assert result.action_distribution["TRADE"] == 1
            assert result.action_distribution["OBSERVE"] == 1

    def test_to_dict(self):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            logger.log(_make_candidate())
            agg = SignalCandidateAggregator(conn)
            result = agg.aggregate_window(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            d = result.to_dict()
            assert "candidates_count" in d
            assert "rejection_reasons_top10" in d


class TestExportDiagnostics:
    def test_export_json(self, tmp_path):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            logger.log(_make_candidate())
            output = tmp_path / "diagnostics.json"
            result = export_diagnostics(conn, output, fmt="json")
            assert result.exists()
            data = json.loads(result.read_text())
            assert "summary" in data
            assert "candidates" in data
            assert len(data["candidates"]) == 1

    def test_export_csv(self, tmp_path):
        with _test_db() as conn:
            logger = SignalCandidateLogger(conn)
            logger.log(_make_candidate())
            output = tmp_path / "diagnostics"
            result = export_diagnostics(conn, output, fmt="csv")
            assert result.suffix == ".csv"
            assert result.exists()
            summary_path = tmp_path / "diagnostics_summary.json"
            assert summary_path.exists()


class TestSignalCandidateModel:
    def test_to_dict(self):
        c = _make_candidate()
        d = c.to_dict()
        assert d["symbol"] == "XAUUSD"
        assert isinstance(d["timestamp"], str)
        assert isinstance(d["rejection_reasons"], list)


# =========================================================================
# Config integration tests
# =========================================================================
class TestConfigIntegration:
    def test_paper_cost_config_in_app_config(self):
        from bot.config import AppConfig
        cfg = AppConfig()
        assert hasattr(cfg, "paper_costs")
        assert cfg.paper_costs.enabled is True
        assert cfg.paper_costs.slippage_limit.base_ticks == 0.01

    def test_micro_loss_defense_config_in_app_config(self):
        from bot.config import AppConfig
        cfg = AppConfig()
        assert hasattr(cfg, "micro_loss_defense")
        assert cfg.micro_loss_defense.enabled is True
        assert cfg.micro_loss_defense.min_stop_spread_mult == 5.0

    def test_adaptive_threshold_config_in_app_config(self):
        from bot.config import AppConfig
        cfg = AppConfig()
        assert hasattr(cfg, "adaptive_threshold")
        assert cfg.adaptive_threshold.enabled is False
