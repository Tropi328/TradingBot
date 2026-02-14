from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bot.config import AppConfig
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyPlugin,
)
from bot.strategy.indicators import atr, latest_value, real_body
from bot.strategy.state_machine import StrategySignal
from bot.strategy.trace import closed_candles


@dataclass(slots=True)
class _OrbState:
    bias: BiasState | None = None


class OrbH4RetestStrategy(StrategyPlugin):
    name = "ORB_H4_RETEST"

    def __init__(self, config: AppConfig):
        self.config = config
        self._state: dict[str, _OrbState] = {}

    def _symbol_state(self, symbol: str) -> _OrbState:
        key = symbol.strip().upper()
        if key not in self._state:
            self._state[key] = _OrbState()
        return self._state[key]

    @staticmethod
    def _params(data: StrategyDataBundle) -> dict[str, Any]:
        raw = data.extra.get("strategy_params")
        if isinstance(raw, dict):
            return raw
        return {}

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        self._symbol_state(symbol)

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        state = self._symbol_state(symbol)
        params = self._params(data)
        h1 = closed_candles(data.candles_h1)
        if len(h1) < 8:
            bias = BiasState(
                symbol=symbol,
                strategy_name=self.name,
                direction="NEUTRAL",
                timeframe="H4",
                updated_at=data.now,
                metadata={"reason": "H1_INSUFFICIENT"},
            )
            state.bias = bias
            return bias

        h4_bars = h1[:4]
        range_high = max(candle.high for candle in h4_bars)
        range_low = min(candle.low for candle in h4_bars)
        latest = h1[-1]
        break_buffer = self._as_float(params.get("break_buffer"), 0.0)
        direction = "NEUTRAL"
        if latest.close > (range_high + break_buffer):
            direction = "LONG"
        elif latest.close < (range_low - break_buffer):
            direction = "SHORT"

        bias = BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction=direction,
            timeframe="H4",
            updated_at=data.now,
            metadata={
                "range_high": range_high,
                "range_low": range_low,
                "breakout_close": latest.close,
                "breakout_time": latest.timestamp.isoformat(),
            },
        )
        state.bias = bias
        return bias

    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        params = self._params(data)
        bias = self._symbol_state(symbol).bias or self.compute_bias(symbol, data)
        if bias.direction not in {"LONG", "SHORT"}:
            return []
        if not data.m5_new_close:
            return []
        ttl = self._as_int(params.get("candidate_ttl_minutes"), 60)
        breakout_time = str(bias.metadata.get("breakout_time", data.now.isoformat()))
        setup_id = f"{symbol}:{self.name}:{breakout_time}:{bias.direction}"
        return [
            SetupCandidate(
                candidate_id=f"ORB-{symbol}-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                strategy_name=self.name,
                side=bias.direction,
                created_at=data.now,
                expires_at=data.now + timedelta(minutes=ttl),
                source_timeframe="M5",
                setup_type="ORB_RETEST",
                origin_strategy=self.name,
                setup_id=setup_id,
                metadata={
                    "setup_id": setup_id,
                    "range_high": bias.metadata.get("range_high"),
                    "range_low": bias.metadata.get("range_low"),
                    "breakout_time": breakout_time,
                },
            )
        ]

    def evaluate_candidate(
        self,
        symbol: str,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategyEvaluation:
        params = self._params(data)
        m5 = closed_candles(data.candles_m5)
        if len(m5) < 12:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["ORB_M5_INSUFFICIENT"],
            )
        atr_m5 = latest_value(atr(m5, self.config.indicators.atr_period))
        if atr_m5 is None or atr_m5 <= 0:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["ORB_ATR_WARMUP"],
            )

        side = candidate.side
        range_high = float(candidate.metadata.get("range_high") or 0.0)
        range_low = float(candidate.metadata.get("range_low") or 0.0)
        zone = range_high if side == "LONG" else range_low
        tolerance_override = params.get("retest_tolerance")
        if tolerance_override is not None:
            tolerance = self._as_float(tolerance_override, atr_m5 * 0.18)
            tolerance_mult = tolerance / max(atr_m5, 1e-9)
        else:
            tolerance_mult = self._as_float(params.get("retest_tolerance_atr"), 0.18)
            tolerance_mult = max(0.10, min(0.25, tolerance_mult))
            tolerance = atr_m5 * tolerance_mult
        max_retest_minutes = self._as_int(params.get("max_retest_minutes"), 90)
        breakout_time = datetime.fromisoformat(str(candidate.metadata.get("breakout_time")))
        cutoff = breakout_time + timedelta(minutes=max_retest_minutes)
        recent = [candle for candle in m5 if candle.timestamp >= breakout_time]
        recent = [candle for candle in recent if candle.timestamp <= cutoff]
        if not recent:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["ORB_RETEST_TIMEOUT"],
            )

        retest_ok = any((candle.low <= zone + tolerance and candle.high >= zone - tolerance) for candle in recent)
        last = recent[-1]
        if side == "LONG":
            rejection = last.low <= zone + tolerance and last.close > zone
            micro = last.close > max(c.high for c in recent[-5:])
        else:
            rejection = last.high >= zone - tolerance and last.close < zone
            micro = last.close < min(c.low for c in recent[-5:])
        displacement = real_body(last)
        displacement_ok = displacement >= self._as_float(params.get("min_displacement_atr"), 0.9) * atr_m5
        confirmations = sum([int(rejection), int(displacement_ok), int(micro)])
        min_confirm = self._as_int(params.get("min_confirm"), 2)

        spread = data.spread or 0.0
        spread_ratio = spread / max(atr_m5, 1e-9)
        spread_ratio_max = self._as_float(params.get("spread_ratio_max"), 0.15)
        execution_score = 10.0 if spread_ratio <= spread_ratio_max else 3.0

        breakout_size = abs(float(candidate.metadata.get("range_high", 0.0)) - float(candidate.metadata.get("range_low", 0.0)))
        breakout_quality = min(30.0, (breakout_size / max(atr_m5, 1e-9)) * 6.0)
        retest_quality = 25.0 if retest_ok else 0.0
        confirmation_strength = min(25.0, (confirmations / 3.0) * 25.0)
        daily_context = 8.0
        base_score = breakout_quality + retest_quality + confirmation_strength + execution_score + daily_context

        soft_reasons: list[str] = []
        penalties_total = 0.0
        if confirmations < min_confirm:
            soft_reasons.append("ORB_CONFIRMATIONS_LOW")
            penalties_total += abs(float(self.config.backtest_tuning.penalty_orb_confirm_low))
        if not retest_ok:
            soft_reasons.append("ORB_NO_RETEST")
            penalties_total += abs(float(self.config.backtest_tuning.penalty_orb_no_retest))
        score_total = round(max(0.0, min(100.0, base_score - penalties_total)), 2)

        setup_state = "READY"
        if not retest_ok:
            setup_state = "WAIT_MITIGATION"
        elif confirmations < min_confirm:
            setup_state = "WAIT_REACTION"

        action = DecisionAction.OBSERVE
        if score_total >= self.config.decision_policy.trade_score_threshold:
            action = DecisionAction.TRADE
        elif self.config.decision_policy.small_score_min <= score_total <= self.config.decision_policy.small_score_max:
            action = DecisionAction.SMALL

        trade_min_confirm = max(min_confirm, self._as_int(params.get("min_confirm_trade"), max(min_confirm, 3)))
        small_min_confirm = max(1, self._as_int(params.get("min_confirm_small"), min_confirm))
        if action == DecisionAction.TRADE and confirmations < trade_min_confirm and confirmations >= small_min_confirm:
            action = DecisionAction.SMALL
        if action == DecisionAction.TRADE and not retest_ok:
            action = DecisionAction.SMALL

        min_displacement_atr = self._as_float(params.get("min_displacement_atr"), 0.9)
        a_plus_displacement_atr = max(
            min_displacement_atr,
            self._as_float(params.get("a_plus_displacement_atr"), 1.25),
        )
        a_plus_spread_ratio_max = min(
            spread_ratio_max,
            self._as_float(params.get("a_plus_spread_ratio_max"), 0.12),
        )
        a_plus_breakout_quality_min = self._as_float(params.get("a_plus_breakout_quality_min"), 18.0)
        a_plus = (
            retest_ok
            and confirmations >= max(trade_min_confirm, 3)
            and displacement >= (a_plus_displacement_atr * atr_m5)
            and spread_ratio <= a_plus_spread_ratio_max
            and breakout_quality >= a_plus_breakout_quality_min
        )

        return StrategyEvaluation(
            action=action,
            score_total=score_total,
            score_breakdown={
                "breakout_quality": round(breakout_quality, 2),
                "retest_quality": round(retest_quality, 2),
                "confirmation_strength": round(confirmation_strength, 2),
                "execution": round(execution_score, 2),
                "daily_context": round(daily_context, 2),
                "penalty_orb_no_retest": round(float(self.config.backtest_tuning.penalty_orb_no_retest), 2) if not retest_ok else 0.0,
                "penalty_orb_confirm_low": round(float(self.config.backtest_tuning.penalty_orb_confirm_low), 2) if confirmations < min_confirm else 0.0,
            },
            reasons_blocking=[],
            would_enter_if=["ORB_RETEST_AND_CONFIRMATIONS_STRONGER"] if soft_reasons else [],
            snapshot={
                "atr_m5": atr_m5,
                "spread_ratio": spread_ratio,
                "range_high": range_high,
                "range_low": range_low,
            },
            metadata={
                "trigger_confirmations": confirmations,
                "min_confirm": min_confirm,
                "setup_state": setup_state,
                "side": side,
                "zone": zone,
                "atr_m5": atr_m5,
                "range_high": range_high,
                "range_low": range_low,
                "retest_tolerance_atr": tolerance_mult,
                "retest_tolerance_abs": tolerance,
                "execution_penalty": 0.0 if execution_score >= 8.0 else 2.0,
                "a_plus": a_plus,
                "trade_min_confirm": trade_min_confirm,
                "small_min_confirm": small_min_confirm,
                "soft_reasons": soft_reasons,
            },
        )

    def generate_order(
        self,
        symbol: str,
        evaluation: StrategyEvaluation,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategySignal | None:
        if evaluation.action not in {DecisionAction.TRADE, DecisionAction.SMALL}:
            return None
        side = str(evaluation.metadata.get("side", candidate.side))
        zone = float(evaluation.metadata.get("zone", 0.0))
        atr_m5 = float(evaluation.metadata.get("atr_m5", 0.0))
        range_high = float(evaluation.metadata.get("range_high", zone))
        range_low = float(evaluation.metadata.get("range_low", zone))
        if atr_m5 <= 0:
            return None
        entry = zone
        a_plus = bool(evaluation.metadata.get("a_plus", False))
        rr = 3.0 if a_plus else 2.0
        if side == "LONG":
            stop = range_low - 0.2 * atr_m5
            risk = entry - stop
            if risk <= 0:
                return None
            tp = entry + rr * risk
        else:
            stop = range_high + 0.2 * atr_m5
            risk = stop - entry
            if risk <= 0:
                return None
            tp = entry - rr * risk
        reason_codes = [self.name, f"SCORE_{int(evaluation.score_total or 0)}"]
        if a_plus:
            reason_codes.append("A_PLUS")
        return StrategySignal(
            side=side,
            entry_price=entry,
            stop_price=stop,
            take_profit=tp,
            rr=rr,
            a_plus=a_plus,
            expires_at=data.now + timedelta(minutes=30),
            reason_codes=reason_codes,
            metadata={
                "strategy": self.name,
                "candidate_id": candidate.candidate_id,
                "setup_id": candidate.metadata.get("setup_id"),
                "a_plus": a_plus,
            },
        )

    def manage_position(self, symbol: str, position, data: StrategyDataBundle) -> list[StrategySignal]:
        return []
