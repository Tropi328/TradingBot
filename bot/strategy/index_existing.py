from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from bot.config import AppConfig
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyPlugin,
)
from bot.strategy.state_machine import H1Snapshot, M15Snapshot, M5Snapshot, StrategyDecision, StrategyEngine, StrategySignal
from bot.strategy.trace import closed_candles


@dataclass(slots=True)
class _IndexState:
    h1_snapshot: H1Snapshot | None = None
    m15_snapshot: M15Snapshot | None = None
    m5_snapshot: M5Snapshot | None = None
    last_decision: StrategyDecision | None = None


class IndexExistingStrategy(StrategyPlugin):
    name = "INDEX_EXISTING"

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = StrategyEngine(config)
        self._state_by_symbol: dict[str, _IndexState] = {}

    def _state(self, symbol: str) -> _IndexState:
        key = symbol.strip().upper()
        if key not in self._state_by_symbol:
            self._state_by_symbol[key] = _IndexState()
        return self._state_by_symbol[key]

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        self._state(symbol)

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        state = self._state(symbol)
        if data.h1_new_close and data.candles_h1:
            state.h1_snapshot = self.engine.evaluate_h1(closed_candles(data.candles_h1))
        elif state.h1_snapshot is None and data.candles_h1:
            state.h1_snapshot = self.engine.evaluate_h1(closed_candles(data.candles_h1))

        direction = "NEUTRAL"
        metadata: dict[str, float | str | int | bool | None] = {}
        if state.h1_snapshot is not None:
            direction = state.h1_snapshot.side or "NEUTRAL"
            metadata = {
                "bias": state.h1_snapshot.bias,
                "pd_state": state.h1_snapshot.pd_state,
                "ema_ready": state.h1_snapshot.ema200_ready,
            }
        return BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction=direction,
            timeframe=self.config.timeframes.h1,
            updated_at=data.now,
            metadata=metadata,
        )

    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        state = self._state(symbol)
        if state.h1_snapshot is None:
            return []
        if data.m15_new_close and data.candles_m15:
            state.m15_snapshot = self.engine.evaluate_m15(
                candles_m15=closed_candles(data.candles_m15),
                h1=state.h1_snapshot,
                minimal_tick_buffer=float(data.extra.get("minimal_tick_buffer", 0.05)),
                now=data.now,
                previous=state.m15_snapshot,
            )
        elif state.m15_snapshot is None and data.candles_m15:
            state.m15_snapshot = self.engine.evaluate_m15(
                candles_m15=closed_candles(data.candles_m15),
                h1=state.h1_snapshot,
                minimal_tick_buffer=float(data.extra.get("minimal_tick_buffer", 0.05)),
                now=data.now,
                previous=None,
            )
        if state.m15_snapshot is None:
            return []
        candidate = SetupCandidate(
            candidate_id=f"INDEX-{symbol}-{int(data.now.timestamp())}",
            symbol=symbol,
            strategy_name=self.name,
            side=state.h1_snapshot.side or "NEUTRAL",
            created_at=data.now,
            expires_at=data.now + timedelta(minutes=5),
            source_timeframe=self.config.timeframes.m5,
            setup_type="PIPELINE_TICK",
            metadata={
                "m15_setup_state": state.m15_snapshot.setup_state,
                "m15_reasons": list(state.m15_snapshot.reason_codes),
                "h1_reasons": list(state.h1_snapshot.reason_codes),
            },
        )
        return [candidate]

    def evaluate_candidate(
        self,
        symbol: str,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategyEvaluation:
        state = self._state(symbol)
        if state.h1_snapshot is None or state.m15_snapshot is None:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=None,
                reasons_blocking=["INDEX_PIPELINE_NOT_READY"],
                snapshot={"symbol": symbol, "stage": "h1_or_m15_missing"},
                metadata={"candidate_id": candidate.candidate_id},
            )
        if not data.m5_new_close:
            reasons = state.m5_snapshot.reason_codes if state.m5_snapshot is not None else ["M5_WAIT_NEW_CLOSE"]
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=None,
                reasons_blocking=list(reasons),
                snapshot={"symbol": symbol, "stage": "m5_wait_close"},
                metadata={"candidate_id": candidate.candidate_id},
            )
        if not data.candles_m5:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=None,
                reasons_blocking=["M5_INSUFFICIENT_BARS"],
                snapshot={"symbol": symbol, "stage": "m5_missing"},
                metadata={"candidate_id": candidate.candidate_id},
            )
        legacy, m5_snapshot = self.engine.evaluate_m5(
            epic=symbol,
            candles_m5=closed_candles(data.candles_m5),
            current_spread=data.spread,
            spread_history=data.spread_history,
            news_blocked=data.news_blocked,
            h1=state.h1_snapshot,
            m15=state.m15_snapshot,
            entry_state=data.entry_state,
        )
        state.m5_snapshot = m5_snapshot
        state.last_decision = legacy
        if legacy.signal is not None:
            return StrategyEvaluation(
                action=DecisionAction.TRADE,
                score_total=None,
                reasons_blocking=[],
                would_enter_if=[],
                snapshot={
                    "symbol": symbol,
                    "spread": data.spread,
                    "entry_price": legacy.signal.entry_price,
                    "rr": legacy.signal.rr,
                },
                metadata={"legacy_reasons": list(legacy.reason_codes)},
            )
        return StrategyEvaluation(
            action=DecisionAction.OBSERVE,
            score_total=None,
            reasons_blocking=list(legacy.reason_codes),
            would_enter_if=["INDEX_SIGNAL_READY"],
            snapshot={"symbol": symbol, "spread": data.spread},
            metadata={"legacy_payload": dict(legacy.payload)},
        )

    def generate_order(
        self,
        symbol: str,
        evaluation: StrategyEvaluation,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategySignal | None:
        state = self._state(symbol)
        if evaluation.action not in {DecisionAction.TRADE, DecisionAction.SMALL}:
            return None
        if state.last_decision is None or state.last_decision.signal is None:
            return None
        return state.last_decision.signal

    def manage_position(self, symbol: str, position, data: StrategyDataBundle) -> list[StrategySignal]:
        return []

    def last_snapshots(self, symbol: str) -> tuple[H1Snapshot | None, M15Snapshot | None, M5Snapshot | None]:
        state = self._state(symbol)
        return state.h1_snapshot, state.m15_snapshot, state.m5_snapshot

    def last_legacy_decision(self, symbol: str) -> StrategyDecision | None:
        return self._state(symbol).last_decision

