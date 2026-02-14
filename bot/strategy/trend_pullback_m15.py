from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta
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
from bot.strategy.indicators import atr, ema, latest_value, real_body
from bot.strategy.state_machine import StrategySignal
from bot.strategy.trace import closed_candles


@dataclass(slots=True)
class _TrendState:
    bias: BiasState | None = None


class TrendPullbackM15Strategy(StrategyPlugin):
    name = "TREND_PULLBACK_M15"

    def __init__(self, config: AppConfig):
        self.config = config
        self._state: dict[str, _TrendState] = {}

    def _symbol_state(self, symbol: str) -> _TrendState:
        key = symbol.strip().upper()
        if key not in self._state:
            self._state[key] = _TrendState()
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
        m15 = closed_candles(data.candles_m15)
        if len(m15) < 60:
            bias = BiasState(
                symbol=symbol,
                strategy_name=self.name,
                direction="NEUTRAL",
                timeframe="M15",
                updated_at=data.now,
                metadata={"trend_strength": 0.0, "reason": "M15_INSUFFICIENT"},
            )
            state.bias = bias
            return bias
        closes = [candle.close for candle in m15]
        ema50 = latest_value(ema(closes, 50))
        if ema50 is None:
            bias = BiasState(
                symbol=symbol,
                strategy_name=self.name,
                direction="NEUTRAL",
                timeframe="M15",
                updated_at=data.now,
                metadata={"trend_strength": 0.0, "reason": "EMA_WARMUP"},
            )
            state.bias = bias
            return bias
        recent = closes[-20:]
        up_moves = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_moves = (len(recent) - 1) - up_moves
        trend_strength = abs(up_moves - down_moves) / max(1.0, len(recent) - 1)
        direction = "NEUTRAL"
        if closes[-1] > ema50 and up_moves >= down_moves:
            direction = "LONG"
        elif closes[-1] < ema50 and down_moves >= up_moves:
            direction = "SHORT"
        bias = BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction=direction,
            timeframe="M15",
            updated_at=data.now,
            metadata={
                "trend_strength": trend_strength,
                "ema50_m15": ema50,
                "last_close": closes[-1],
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
        m5 = closed_candles(data.candles_m5)
        if len(m5) < 20:
            return []
        atr_m5 = latest_value(atr(m5, self.config.indicators.atr_period))
        if atr_m5 is None:
            return []
        pullback_depth_atr = self._as_float(params.get("pullback_depth_atr"), 0.5)
        last = m5[-1]
        recent = m5[-8:]
        if bias.direction == "LONG":
            anchor = max(candle.high for candle in recent)
            pullback_ok = (anchor - last.low) >= pullback_depth_atr * atr_m5
        else:
            anchor = min(candle.low for candle in recent)
            pullback_ok = (last.high - anchor) >= pullback_depth_atr * atr_m5
        if not pullback_ok:
            return []

        setup_id = f"{symbol}:{self.name}:{last.timestamp.isoformat()}:{bias.direction}"
        ttl = self._as_int(params.get("candidate_ttl_minutes"), 45)
        return [
            SetupCandidate(
                candidate_id=f"TREND-{symbol}-{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                strategy_name=self.name,
                side=bias.direction,
                created_at=data.now,
                expires_at=data.now + timedelta(minutes=ttl),
                source_timeframe="M5",
                setup_type="TREND_PULLBACK",
                origin_strategy=self.name,
                setup_id=setup_id,
                metadata={
                    "setup_id": setup_id,
                    "anchor": anchor,
                    "atr_m5": atr_m5,
                    "pullback_depth_atr": pullback_depth_atr,
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
        bias = self._symbol_state(symbol).bias or self.compute_bias(symbol, data)
        m5 = closed_candles(data.candles_m5)
        if len(m5) < 10:
            return StrategyEvaluation(action=DecisionAction.OBSERVE, score_total=0.0, reasons_blocking=["TREND_M5_INSUFFICIENT"])

        atr_m5 = float(candidate.metadata.get("atr_m5", 0.0))
        if atr_m5 <= 0:
            atr_current = latest_value(atr(m5, self.config.indicators.atr_period))
            atr_m5 = float(atr_current or 0.0)
        if atr_m5 <= 0:
            return StrategyEvaluation(action=DecisionAction.OBSERVE, score_total=0.0, reasons_blocking=["TREND_ATR_WARMUP"])

        trend_strength = float(bias.metadata.get("trend_strength", 0.0))
        trend_thr = self._as_float(params.get("trend_strength_min"), 0.2)
        trend_ok = trend_strength >= trend_thr

        last = m5[-1]
        trigger_disp_thr = self._as_float(params.get("trigger_displacement_atr"), 1.0)
        displacement_ok = real_body(last) >= trigger_disp_thr * atr_m5
        if candidate.side == "LONG":
            trigger_ok = last.close >= max(candle.high for candle in m5[-5:])
            pullback_quality = min(25.0, (max(candle.high for candle in m5[-10:]) - last.low) / max(atr_m5, 1e-9) * 6.0)
        else:
            trigger_ok = last.close <= min(candle.low for candle in m5[-5:])
            pullback_quality = min(25.0, (last.high - min(candle.low for candle in m5[-10:])) / max(atr_m5, 1e-9) * 6.0)
        confirmations = sum([int(trend_ok), int(displacement_ok), int(trigger_ok)])
        min_confirm = self._as_int(params.get("min_confirm"), 2)

        spread = data.spread or 0.0
        spread_ratio = spread / max(atr_m5, 1e-9)
        execution_score = 10.0 if spread_ratio <= self._as_float(params.get("spread_ratio_max"), 0.15) else 4.0

        trend_score = min(35.0, trend_strength * 60.0)
        trigger_quality = min(20.0, (real_body(last) / max(atr_m5, 1e-9)) * 12.0)
        daily_context = 8.0
        score_total = round(trend_score + pullback_quality + trigger_quality + execution_score + daily_context, 2)

        reasons: list[str] = []
        if confirmations < min_confirm:
            reasons.append("TREND_CONFIRMATIONS_LOW")
        if not trend_ok:
            reasons.append("TREND_STRENGTH_LOW")
        setup_state = "READY" if confirmations >= min_confirm else "WAIT_REACTION"

        action = DecisionAction.OBSERVE
        if not reasons:
            if score_total >= self.config.decision_policy.trade_score_threshold:
                action = DecisionAction.TRADE
            elif self.config.decision_policy.small_score_min <= score_total <= self.config.decision_policy.small_score_max:
                action = DecisionAction.SMALL

        trade_min_confirm = max(min_confirm, self._as_int(params.get("min_confirm_trade"), max(min_confirm, 3)))
        small_min_confirm = max(1, self._as_int(params.get("min_confirm_small"), min_confirm))
        if action == DecisionAction.TRADE and confirmations < trade_min_confirm and confirmations >= small_min_confirm:
            action = DecisionAction.SMALL

        spread_ratio_max = self._as_float(params.get("spread_ratio_max"), 0.15)
        a_plus_trend_strength_min = max(
            trend_thr,
            self._as_float(params.get("a_plus_trend_strength_min"), max(0.35, trend_thr + 0.08)),
        )
        a_plus_displacement_atr = max(
            trigger_disp_thr,
            self._as_float(params.get("a_plus_displacement_atr"), 1.25),
        )
        a_plus_spread_ratio_max = min(
            spread_ratio_max,
            self._as_float(params.get("a_plus_spread_ratio_max"), 0.12),
        )
        a_plus = (
            trend_ok
            and trigger_ok
            and confirmations >= max(trade_min_confirm, 3)
            and trend_strength >= a_plus_trend_strength_min
            and real_body(last) >= (a_plus_displacement_atr * atr_m5)
            and spread_ratio <= a_plus_spread_ratio_max
        )

        return StrategyEvaluation(
            action=action,
            score_total=score_total,
            score_breakdown={
                "trend_strength": round(trend_score, 2),
                "pullback_quality": round(pullback_quality, 2),
                "trigger_quality": round(trigger_quality, 2),
                "execution": round(execution_score, 2),
                "daily_context": round(daily_context, 2),
            },
            reasons_blocking=reasons,
            would_enter_if=["TREND_CONFIRMATIONS>=MIN"] if reasons else [],
            snapshot={
                "trend_strength": trend_strength,
                "spread_ratio": spread_ratio,
                "atr_m5": atr_m5,
            },
            metadata={
                "trigger_confirmations": confirmations,
                "min_confirm": min_confirm,
                "setup_state": setup_state,
                "side": candidate.side,
                "anchor": candidate.metadata.get("anchor"),
                "atr_m5": atr_m5,
                "execution_penalty": 0.0 if execution_score >= 8.0 else 1.0,
                "a_plus": a_plus,
                "trade_min_confirm": trade_min_confirm,
                "small_min_confirm": small_min_confirm,
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
        atr_m5 = float(evaluation.metadata.get("atr_m5", 0.0))
        if atr_m5 <= 0:
            return None
        quote = data.quote
        if quote is not None:
            bid, ask, _ = quote
            entry = ask if side == "LONG" else bid
        else:
            entry = float(candidate.metadata.get("anchor", 0.0))
            if entry <= 0:
                return None
        a_plus = bool(evaluation.metadata.get("a_plus", False))
        rr = 3.0 if a_plus else 2.0
        if side == "LONG":
            stop = entry - 1.2 * atr_m5
            risk = entry - stop
            if risk <= 0:
                return None
            tp = entry + rr * risk
        else:
            stop = entry + 1.2 * atr_m5
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
