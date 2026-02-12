from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import bot.backtest.engine as engine_module
from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
)
from bot.strategy.state_machine import StrategySignal


def _candles(count: int = 900) -> list[Candle]:
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[Candle] = []
    price = 100.0
    for i in range(count):
        ts = start + timedelta(minutes=5 * i)
        close = price + (0.05 * i)
        rows.append(
            Candle(
                timestamp=ts,
                open=close - 0.02,
                high=close + 0.25,
                low=close - 0.20,
                close=close,
                bid=close - 0.01,
                ask=close + 0.01,
                volume=1.0,
            )
        )
    return rows


class _AlwaysTradeScalp:
    name = "SCALP_ICT_PA"

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        del symbol, data

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        return BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction="LONG",
            timeframe="M15",
            updated_at=data.now,
            metadata={},
        )

    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        if not data.m5_new_close:
            return []
        created = data.now
        return [
            SetupCandidate(
                candidate_id=f"C-{int(created.timestamp())}",
                symbol=symbol,
                strategy_name=self.name,
                side="LONG",
                created_at=created,
                expires_at=created + timedelta(minutes=20),
                source_timeframe="M5",
                setup_type="TEST",
                metadata={"setup_id": f"S-{int(created.timestamp())}"},
            )
        ]

    def evaluate_candidate(self, symbol: str, candidate: SetupCandidate, data: StrategyDataBundle) -> StrategyEvaluation:
        del symbol, candidate, data
        return StrategyEvaluation(
            action=DecisionAction.TRADE,
            score_total=90.0,
            score_breakdown={
                "bias": 20.0,
                "sweep": 20.0,
                "mss": 20.0,
                "displacement": 20.0,
                "fvg": 15.0,
                "spread": 5.0,
            },
            reasons_blocking=[],
            would_enter_if=[],
            snapshot={"spread": 0.02},
            metadata={
                "atr_m5": 2.0,
                "trigger_confirmations": 3,
                "setup_state": "READY",
                "side": "LONG",
            },
        )

    def generate_order(self, symbol: str, evaluation: StrategyEvaluation, candidate: SetupCandidate, data: StrategyDataBundle) -> StrategySignal | None:
        del symbol, evaluation, candidate
        entry = data.candles_m5[-2].close
        return StrategySignal(
            side="LONG",
            entry_price=entry,
            stop_price=entry - 1.0,
            take_profit=entry + 0.2,
            rr=2.0,
            a_plus=False,
            expires_at=data.now + timedelta(minutes=30),
            reason_codes=["TEST"],
            metadata={},
        )

    def manage_position(self, symbol: str, position, data: StrategyDataBundle) -> list[StrategySignal]:
        del symbol, position, data
        return []


class _ObserveScalp(_AlwaysTradeScalp):
    def evaluate_candidate(self, symbol: str, candidate: SetupCandidate, data: StrategyDataBundle) -> StrategyEvaluation:
        del symbol, candidate, data
        return StrategyEvaluation(
            action=DecisionAction.OBSERVE,
            score_total=70.0,
            score_breakdown={"bias": 20.0, "sweep": 20.0, "mss": 10.0, "displacement": 10.0, "fvg": 10.0},
            reasons_blocking=["NO_EDGE"],
            would_enter_if=["EDGE_OK"],
            snapshot={"spread": 0.02},
            metadata={"atr_m5": 2.0, "trigger_confirmations": 1, "setup_state": "READY", "side": "LONG"},
        )


def test_backtest_uses_router_plugins_and_places_trades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "ScalpIctPriceActionStrategy", lambda cfg: _AlwaysTradeScalp())

    config = AppConfig()
    asset = config.assets[0].model_copy(deep=True)
    asset.epic = "XAUUSD"
    report = engine_module.run_backtest_multi_strategy(
        config=config,
        asset=asset,
        candles_m5=_candles(),
        assumed_spread=0.02,
    )

    assert report.trades > 0
    assert report.signal_candidates > 0
    assert report.decision_counts.get("TRADE", 0) > 0


def test_backtest_reports_top_blockers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "ScalpIctPriceActionStrategy", lambda cfg: _ObserveScalp())

    config = AppConfig()
    asset = config.assets[0].model_copy(deep=True)
    asset.epic = "XAUUSD"
    report = engine_module.run_backtest_multi_strategy(
        config=config,
        asset=asset,
        candles_m5=_candles(),
        assumed_spread=0.02,
    )

    assert report.trades == 0
    assert report.top_blockers
    assert "NO_EDGE" in report.top_blockers
