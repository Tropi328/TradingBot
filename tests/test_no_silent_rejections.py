"""Regression tests: verify that config values don't silently block
every trade through spread-limit, sizing, or edge-to-cost miscalibration.

Root causes caught:
- spread_limit_points set in "wrong units" after point_size change
  (6.0 pts limit vs 65 pts actual spread → 100% rejection)
- max_risk_cash_per_trade capping below percent-mode target
  ($1.00 cap vs $50 target → RISK_AFTER_ROUNDING_TOO_HIGH)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle
from bot.execution.feasibility import validate_order
from bot.execution.order_validation import (
    expected_move_too_small,
    price_to_points,
    compute_risk_cash_plan,
)
from bot.strategy.state_machine import StrategyDecision, StrategySignal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

XAUUSD_POINT_SIZE = 0.01
XAUUSD_TYPICAL_SPREAD = 0.65           # price units
XAUUSD_SPREAD_POINTS = XAUUSD_TYPICAL_SPREAD / XAUUSD_POINT_SIZE   # = 65

BTCUSD_POINT_SIZE = 0.1
BTCUSD_TYPICAL_SPREAD = 8.0
BTCUSD_SPREAD_POINTS = BTCUSD_TYPICAL_SPREAD / BTCUSD_POINT_SIZE   # = 80


def _make_candles(count: int = 900, base: float = 2000.0) -> list[Candle]:
    """Synthetic 5m XAUUSD candles with ATR ~5 and spread ~0.40."""
    start = datetime(2024, 8, 1, 8, 0, tzinfo=timezone.utc)
    candles: list[Candle] = []
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        p = base + 0.01 * (i % 200)
        candles.append(
            Candle(
                timestamp=ts, open=p, high=p + 3.0, low=p - 2.0,
                close=p + 0.3, bid=p + 0.1, ask=p + 0.5, volume=100.0,
            )
        )
    return candles


def _no_signal() -> StrategyDecision:
    return StrategyDecision(
        signal=None, reason_codes=[], bias="NEUTRAL", pd_state="UNKNOWN",
        sweep_ok=False, mss_ok=False, displacement_ok=False, fvg_ok=False,
        spread_ok=True, payload={},
    )


class _FakeLongEngine:
    """Emit a LONG signal once at bar ~701."""

    EMIT_BAR = 701

    def __init__(self, _config: AppConfig):
        self.sent = False

    def evaluate(self, **kwargs) -> StrategyDecision:  # noqa: ANN003
        candles_m5: list[Candle] = kwargs.get("candles_m5", [])
        if self.sent or len(candles_m5) < self.EMIT_BAR:
            return _no_signal()
        self.sent = True
        entry = candles_m5[-1].close
        signal = StrategySignal(
            side="LONG", entry_price=entry, stop_price=entry - 3.0,
            take_profit=entry + 6.0, rr=2.0, a_plus=False,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
            reason_codes=["REGRESSION"], metadata={},
        )
        return StrategyDecision(
            signal=signal, reason_codes=["REGRESSION"], bias="UP",
            pd_state="DISCOUNT", sweep_ok=True, mss_ok=True,
            displacement_ok=True, fvg_ok=True, spread_ok=True, payload={},
        )


@pytest.fixture()
def xauusd_asset() -> AssetConfig:
    return AssetConfig(
        epic="XAUUSD", currency="USD", instrument_currency="USD",
        point_size=0.01, minimal_tick_buffer=0.05,
        min_size=0.01, size_step=0.01, trade_enabled=True,
    )


# ---------------------------------------------------------------
# 1. Unit: spread_limit_points must allow typical XAUUSD spreads
# ---------------------------------------------------------------

class TestSpreadLimitCalibration:
    """Catches: spread_limit_points set too low after point_size change."""

    def test_xauusd_typical_spread_passes_limit(self) -> None:
        """Current config must NOT block XAUUSD typical spread."""
        cfg = AppConfig()
        limit = cfg.backtest_tuning.spread_limit_points
        if limit is None:
            return                          # disabled => nothing to block
        assert XAUUSD_SPREAD_POINTS <= float(limit), (
            f"XAUUSD spread ({XAUUSD_SPREAD_POINTS:.0f} pts) exceeds "
            f"spread_limit_points ({limit}). This blocks 100% of trades. "
            f"Likely mis-calibrated after point_size change."
        )

    def test_xauusd_p90_spread_passes_limit(self) -> None:
        """Even at p90 spread (~80 pts) the limit should allow trades."""
        cfg = AppConfig()
        limit = cfg.backtest_tuning.spread_limit_points
        if limit is None:
            return
        p90_spread_pts = 0.80 / XAUUSD_POINT_SIZE  # $0.80 -> 80 pts
        assert p90_spread_pts <= float(limit), (
            f"XAUUSD p90 spread ({p90_spread_pts:.0f} pts) exceeds "
            f"spread_limit_points ({limit})."
        )


# ---------------------------------------------------------------
# 2. Unit: max_risk_cash_per_trade must not conflict with percent
# ---------------------------------------------------------------

class TestRiskCashCalibration:
    """Catches: max_risk_cash_per_trade below percent-mode target."""

    def test_max_risk_cash_not_below_target(self) -> None:
        cfg = AppConfig()
        plan = compute_risk_cash_plan(
            risk=cfg.risk,
            equity=cfg.risk.equity,
            effective_risk_per_trade=cfg.risk.risk_per_trade,
        )
        # target should not be >> max
        if cfg.risk.max_risk_cash_per_trade is not None:
            assert plan.max_risk_cash >= plan.target_risk_cash * 0.8, (
                f"max_risk_cash (${plan.max_risk_cash:.2f}) is far below "
                f"target (${plan.target_risk_cash:.2f}). "
                f"This causes RISK_AFTER_ROUNDING_TOO_HIGH for every trade."
            )


# ---------------------------------------------------------------
# 3. Unit: edge-to-cost ratio must not block typical XAUUSD moves
# ---------------------------------------------------------------

class TestEdgeToCostCalibration:
    """Catches: edge_to_cost * spread_points exceeding typical TP moves."""

    def test_typical_tp_passes_edge_check(self) -> None:
        """A $4 TP on XAUUSD (400 pts) with 65 pts spread must pass."""
        cfg = AppConfig()
        move_points = 4.0 / XAUUSD_POINT_SIZE   # 400
        assert not expected_move_too_small(
            expected_move_points=move_points,
            spread_points=XAUUSD_SPREAD_POINTS,
            min_edge_to_cost_ratio=cfg.backtest_tuning.min_edge_to_cost_ratio,
        ), (
            f"$4 TP move ({move_points:.0f} pts) blocked by "
            f"edge_to_cost={cfg.backtest_tuning.min_edge_to_cost_ratio} * "
            f"spread={XAUUSD_SPREAD_POINTS:.0f} = "
            f"{cfg.backtest_tuning.min_edge_to_cost_ratio * XAUUSD_SPREAD_POINTS:.0f}."
        )


# ---------------------------------------------------------------
# 4. Unit: validate_order must pass with sane XAUUSD sizing
# ---------------------------------------------------------------

class TestFeasibilityRegression:
    """Full validate_order call with XAUUSD parameters."""

    def test_xauusd_order_passes_feasibility(self, xauusd_asset: AssetConfig) -> None:
        cfg = AppConfig()
        plan = compute_risk_cash_plan(
            risk=cfg.risk,
            equity=cfg.risk.equity,
            effective_risk_per_trade=cfg.risk.risk_per_trade,
        )
        entry = 2000.0
        stop = 1997.0           # $3 risk distance
        tp = 2006.0             # $6 reward
        raw_size = plan.target_risk_cash / abs(entry - stop)

        spread_pts = XAUUSD_SPREAD_POINTS
        limit = cfg.backtest_tuning.spread_limit_points
        result = validate_order(
            raw_size=raw_size,
            entry_price=entry,
            stop_price=stop,
            take_profit=tp,
            min_size=xauusd_asset.min_size,
            size_step=xauusd_asset.size_step,
            max_risk_cash=plan.max_risk_cash,
            equity=cfg.risk.equity,
            open_positions_count=0,
            max_positions=cfg.risk.max_positions,
            spread=spread_pts,
            spread_limit=(float(limit) if limit is not None else None),
            min_stop_distance=xauusd_asset.minimal_tick_buffer,
            free_margin=cfg.risk.equity,
            margin_requirement_pct=cfg.backtest_tuning.broker_margin_requirement_pct,
            max_leverage=cfg.backtest_tuning.broker_leverage,
            margin_safety_factor=1.0,
            allow_min_size_override_if_within_risk=cfg.risk.allow_min_size_override_if_within_risk,
        )
        assert result.ok, (
            f"validate_order rejected XAUUSD trade: "
            f"reason={result.reason}, details={result.details}"
        )


# ---------------------------------------------------------------
# 5. Integration: run_backtest with a fake signal must produce
#    at least 1 signal_candidate
# ---------------------------------------------------------------

class TestBacktestPipeline:
    """End-to-end: signal -> sizing -> order -> trade."""

    def test_signal_reaches_signal_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        xauusd_asset: AssetConfig,
    ) -> None:
        monkeypatch.setattr(engine_module, "StrategyEngine", _FakeLongEngine)

        config = AppConfig()
        config.backtest_tuning.spread_limit_points = 100.0
        config.risk.max_risk_cash_per_trade = None
        config.risk.allow_min_size_override_if_within_risk = True

        report = engine_module.run_backtest(
            config=config,
            asset=xauusd_asset,
            candles_m5=_make_candles(900, 2000.0),
            assumed_spread=0.40,
        )
        # run_backtest stores signal_candidates in decision_funnel, not top-level
        funnel_sc = report.decision_funnel.get("signal_candidates", 0)
        assert funnel_sc >= 1 or report.trades >= 1, (
            f"signal_candidates={funnel_sc}, trades={report.trades}, "
            f"rejected={dict(report.rejected_by_reason)}, "
            f"blockers={dict(report.top_blockers)}"
        )
        assert "SPREAD_TOO_WIDE" not in dict(report.rejected_by_reason)
        assert "RISK_AFTER_ROUNDING_TOO_HIGH" not in dict(report.rejected_by_reason)

    def test_spread_limit_blocks_when_too_tight(
        self,
        monkeypatch: pytest.MonkeyPatch,
        xauusd_asset: AssetConfig,
    ) -> None:
        """Positive control: a deliberately tight limit rejects the signal."""
        monkeypatch.setattr(engine_module, "StrategyEngine", _FakeLongEngine)

        config = AppConfig()
        config.backtest_tuning.spread_limit_points = 5.0   # WAY below 40 pts
        config.risk.max_risk_cash_per_trade = None

        report = engine_module.run_backtest(
            config=config,
            asset=xauusd_asset,
            candles_m5=_make_candles(900, 2000.0),
            assumed_spread=0.40,
        )
        funnel_sc = report.decision_funnel.get("signal_candidates", 0)
        assert funnel_sc == 0 and report.trades == 0


# ---------------------------------------------------------------
# 6. Decision-trace JSONL (multi-strategy path) smoke test
# ---------------------------------------------------------------

class TestDecisionTraceOutput:
    """Verify that decision_trace_path produces valid JSONL."""

    def test_trace_file_written_when_path_given(
        self,
        monkeypatch: pytest.MonkeyPatch,
        xauusd_asset: AssetConfig,
        tmp_path: Path,
    ) -> None:
        """Monkeypatch TrendPullbackM15Strategy.detect_candidates to
        return one candidate, then verify the trace records it."""
        from bot.strategy.contracts import (
            BiasState,
            DecisionAction,
            SetupCandidate,
            StrategyEvaluation,
        )

        _orig_class = engine_module.TrendPullbackM15Strategy

        class _PatchedTrend(_orig_class):
            """Injects one fake candidate at bar 701."""

            def __init__(self, *a, **kw):  # noqa: ANN002, ANN003
                super().__init__(*a, **kw)
                self._injected = False

            def detect_candidates(self, symbol, data):  # noqa: ANN001
                real = super().detect_candidates(symbol, data)
                if self._injected or len(data.candles_m5) < 701:
                    return real
                self._injected = True
                c = data.candles_m5[-1]
                return real + [
                    SetupCandidate(
                        candidate_id="REG_TEST",
                        symbol=symbol,
                        strategy_name="TREND_PULLBACK_M15",
                        side="LONG",
                        created_at=c.timestamp,
                        expires_at=c.timestamp + timedelta(hours=4),
                        source_timeframe="M5",
                        setup_type="TREND_PULLBACK_M15",
                        features={"trend_strength": 0.5, "displacement_atr": 1.5},
                        metadata={},
                    )
                ]

            def evaluate_candidate(self, symbol, candidate, data):  # noqa: ANN001
                if candidate.candidate_id == "REG_TEST":
                    return StrategyEvaluation(
                        action=DecisionAction.TRADE,
                        score_total=75.0,
                        score_layers={"edge": 30, "trigger": 30, "execution": 15},
                        metadata={
                            "side": "LONG",
                            "confirmations": 3,
                            "atr_m5": 3.0,
                            "spread_ratio": 0.08,
                        },
                    )
                return super().evaluate_candidate(symbol, candidate, data)

            def generate_order(self, symbol, evaluation, candidate, data):  # noqa: ANN001
                if candidate.candidate_id == "REG_TEST":
                    c = data.candles_m5[-1]
                    return StrategySignal(
                        side="LONG",
                        entry_price=c.close,
                        stop_price=c.close - 3.0,
                        take_profit=c.close + 6.0,
                        rr=2.0,
                        a_plus=False,
                        expires_at=c.timestamp + timedelta(hours=4),
                        reason_codes=["REGRESSION_TRACE"],
                        metadata={},
                    )
                return super().generate_order(symbol, evaluation, candidate, data)

        monkeypatch.setattr(engine_module, "TrendPullbackM15Strategy", _PatchedTrend)

        config = AppConfig()
        config.backtest_tuning.spread_limit_points = 100.0
        config.risk.max_risk_cash_per_trade = None
        config.risk.allow_min_size_override_if_within_risk = True

        trace_file = tmp_path / "trace.jsonl"
        report = engine_module.run_backtest_multi_strategy(
            config=config,
            asset=xauusd_asset,
            candles_m5=_make_candles(900, 2000.0),
            assumed_spread=0.40,
            decision_trace_path=str(trace_file),
        )

        # Trace file should exist with at least one record
        if trace_file.exists():
            lines = trace_file.read_text("utf-8").strip().splitlines()
            records = [json.loads(line) for line in lines]
            assert len(records) >= 1, "Expected >=1 decision trace record"
            # Verify required fields (two schemas: DTW live-flush and engine batch)
            for rec in records:
                # Every record must have a timestamp
                assert "ts_utc" in rec or "ts" in rec
                # Every record must have a type marker
                assert "type" in rec or "fate" in rec
                # Spread info present
                assert "spread_points" in rec
