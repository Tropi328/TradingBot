from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from bot.config import AppConfig
from bot.data.spreads import median_spread
from bot.strategy.contracts import (
    BiasState,
    DecisionAction,
    SetupCandidate,
    StrategyDataBundle,
    StrategyEvaluation,
    StrategyPlugin,
)
from bot.strategy.ict import detect_latest_fvg, detect_mss, detect_sweep_reject
from bot.strategy.indicators import atr, ema, latest_value, real_body
from bot.strategy.state_machine import StrategySignal
from bot.strategy.swings import detect_swings, index_at_or_after
from bot.strategy.trace import closed_candles

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _MissProbe:
    probe_id: str
    miss_key: str
    symbol: str
    direction: str
    start_price: float
    target_price: float
    created_at: datetime
    expires_at: datetime


@dataclass(slots=True)
class _ScalpState:
    bias: BiasState | None = None
    queue: deque[SetupCandidate] = field(default_factory=deque)
    last_evaluation: StrategyEvaluation | None = None
    last_order: StrategySignal | None = None
    last_candidate_at: datetime | None = None
    missed_total: int = 0
    missed_hits: int = 0
    probes: list[_MissProbe] = field(default_factory=list)
    recent_probe_keys: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _SymbolProfile:
    bias_tf: str
    bos_tf: str
    h1_bos_mode: str
    h1_no_bos_penalty: float
    allow_neutral_bias: bool
    neutral_bias_trade_mode: str
    neutral_bias_penalty: float
    neutral_bias_risk_multiplier: float
    pd_filter_mode: str
    pd_fail_penalty: float
    candidate_relaxed: bool
    sweep_min_hours: int
    sweep_max_hours: int
    sweep_threshold_multiplier: float
    displacement_multiplier: float
    fvg_min_atr: float
    candidate_ttl_minutes: int
    candidate_min_interval_minutes: int
    relaxed_tick_buffer_multiplier: float
    miss_round_minutes: int


class ScalpIctPriceActionStrategy(StrategyPlugin):
    name = "SCALP_ICT_PA"

    def __init__(self, config: AppConfig):
        self.config = config
        self._state_by_symbol: dict[str, _ScalpState] = {}

    def _state(self, symbol: str) -> _ScalpState:
        key = symbol.strip().upper()
        if key not in self._state_by_symbol:
            self._state_by_symbol[key] = _ScalpState()
        return self._state_by_symbol[key]

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_mode(value: Any, default: str, allowed: set[str]) -> str:
        if value is None:
            return default
        mode = str(value).strip().upper()
        return mode if mode in allowed else default

    @staticmethod
    def _strategy_params(data: StrategyDataBundle) -> dict[str, Any]:
        raw = data.extra.get("strategy_params")
        if isinstance(raw, dict):
            return raw
        return {}

    def _profile(self, data: StrategyDataBundle) -> _SymbolProfile:
        params = self._strategy_params(data)
        candidate_relaxed = self._as_bool(params.get("candidate_relaxed"), False)

        sweep_min = self._as_int(params.get("sweep_lookback_min_hours"), 1)
        sweep_max_default = 6 if candidate_relaxed else 4
        sweep_max = self._as_int(params.get("sweep_lookback_max_hours"), sweep_max_default)
        if sweep_min > sweep_max:
            sweep_min, sweep_max = sweep_max, sweep_min

        threshold_default = self.config.sweep.threshold_atr_multiplier * (0.65 if candidate_relaxed else 1.0)
        displacement_default = self.config.displacement.base_multiplier * (0.8 if candidate_relaxed else 1.0)
        fvg_min_default = 0.02 if candidate_relaxed else 0.0

        return _SymbolProfile(
            bias_tf=str(params.get("bias_tf", self.config.scalp.bias_timeframe)).upper(),
            bos_tf=str(params.get("bos_tf", self.config.scalp.bias_timeframe)).upper(),
            h1_bos_mode=self._as_mode(
                params.get("h1_bos_mode"),
                default="IGNORE",
                allowed={"IGNORE", "SCORE", "BLOCK"},
            ),
            h1_no_bos_penalty=max(0.0, self._as_float(params.get("h1_no_bos_penalty"), 10.0)),
            allow_neutral_bias=self._as_bool(params.get("allow_neutral_bias"), False),
            neutral_bias_trade_mode=self._as_mode(
                params.get("neutral_bias_trade_mode"),
                default="SMALL",
                allowed={"SMALL", "TRADE"},
            ),
            neutral_bias_penalty=max(0.0, self._as_float(params.get("neutral_bias_penalty"), 5.0)),
            neutral_bias_risk_multiplier=min(
                1.0,
                max(
                    0.05,
                    self._as_float(
                        params.get("neutral_bias_risk_multiplier"),
                        self.config.scalp.small_risk_multiplier,
                    ),
                ),
            ),
            pd_filter_mode=self._as_mode(
                params.get("pd_filter_mode"),
                default="IGNORE",
                allowed={"IGNORE", "SCORE", "BLOCK"},
            ),
            pd_fail_penalty=max(0.0, self._as_float(params.get("pd_fail_penalty"), 5.0)),
            candidate_relaxed=candidate_relaxed,
            sweep_min_hours=max(1, sweep_min),
            sweep_max_hours=max(1, sweep_max),
            sweep_threshold_multiplier=max(
                0.01,
                self._as_float(params.get("sweep_threshold_multiplier"), threshold_default),
            ),
            displacement_multiplier=max(
                0.1,
                self._as_float(params.get("displacement_multiplier"), displacement_default),
            ),
            fvg_min_atr=max(0.0, self._as_float(params.get("fvg_min_atr"), fvg_min_default)),
            candidate_ttl_minutes=max(
                1,
                self._as_int(params.get("candidate_ttl_minutes"), self.config.scalp.candidate_ttl_minutes),
            ),
            candidate_min_interval_minutes=max(
                0,
                self._as_int(params.get("candidate_min_interval_minutes"), 0),
            ),
            relaxed_tick_buffer_multiplier=max(
                0.1,
                min(
                    1.0,
                    self._as_float(params.get("relaxed_tick_buffer_multiplier"), 0.5),
                ),
            ),
            miss_round_minutes=max(1, self._as_int(params.get("miss_round_minutes"), 5)),
        )

    @staticmethod
    def _candles_for_tf(data: StrategyDataBundle, timeframe: str) -> list:
        key = timeframe.strip().upper()
        if key == "H1":
            return closed_candles(data.candles_h1)
        if key == "M15":
            return closed_candles(data.candles_m15)
        if key == "M5":
            return closed_candles(data.candles_m5)
        return []

    def _bos_state(self, candles: list) -> tuple[str, float | None, float | None]:
        if len(candles) < (self.config.swings.fractal_left + self.config.swings.fractal_right + 1):
            return "NONE", None, None
        highs, lows = detect_swings(
            candles,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        last_close = candles[-1].close
        last_high = highs[-1].price if highs else None
        last_low = lows[-1].price if lows else None
        if last_high is not None and last_close > last_high:
            return "UP", last_high, last_low
        if last_low is not None and last_close < last_low:
            return "DOWN", last_high, last_low
        return "NONE", last_high, last_low

    def _h1_context(self, h1_candles: list) -> dict[str, Any]:
        context: dict[str, Any] = {
            "h1_ema_ready": False,
            "h1_ema200": None,
            "h1_close": None,
            "h1_bos_state": "NONE",
            "h1_pd_available": False,
            "h1_pd_eq": None,
            "h1_pd_low": None,
            "h1_pd_high": None,
            "h1_pd_long_fail": False,
            "h1_pd_short_fail": False,
        }
        if not h1_candles:
            return context

        closes = [c.close for c in h1_candles]
        ema200 = latest_value(ema(closes, period=self.config.indicators.ema_period_h1))
        bos_state, _, _ = self._bos_state(h1_candles)
        context["h1_close"] = closes[-1]
        context["h1_ema200"] = ema200
        context["h1_ema_ready"] = ema200 is not None
        context["h1_bos_state"] = bos_state

        highs, lows = detect_swings(
            h1_candles,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        if highs and lows:
            dealing_high = highs[-1].price
            dealing_low = lows[-1].price
            if dealing_high < dealing_low:
                dealing_high, dealing_low = dealing_low, dealing_high
            eq = (dealing_high + dealing_low) / 2.0
            close = closes[-1]
            context["h1_pd_available"] = True
            context["h1_pd_eq"] = eq
            context["h1_pd_low"] = dealing_low
            context["h1_pd_high"] = dealing_high
            context["h1_pd_long_fail"] = close >= eq
            context["h1_pd_short_fail"] = close <= eq
        return context

    @staticmethod
    def _round_timestamp(ts: datetime, minutes: int) -> datetime:
        rounded_minute = (ts.minute // minutes) * minutes
        return ts.replace(minute=rounded_minute, second=0, microsecond=0)

    def preprocess(self, symbol: str, data: StrategyDataBundle) -> None:
        state = self._state(symbol)
        now = data.now
        while state.queue and state.queue[0].expires_at <= now:
            state.queue.popleft()

        latest_close = None
        latest_high = None
        latest_low = None
        closed_m5 = closed_candles(data.candles_m5)
        if closed_m5:
            latest_close = closed_m5[-1].close
            latest_high = closed_m5[-1].high
            latest_low = closed_m5[-1].low

        active: list[_MissProbe] = []
        active_keys: set[str] = set()
        for probe in state.probes:
            if latest_close is not None and latest_high is not None and latest_low is not None:
                hit = (
                    probe.direction == "LONG"
                    and latest_high >= probe.target_price
                ) or (
                    probe.direction == "SHORT"
                    and latest_low <= probe.target_price
                )
                if hit:
                    state.missed_hits += 1
                    state.missed_total += 1
                    LOGGER.info(
                        "MissedOpportunity symbol=%s probe=%s hit=1 direction=%s start=%.5f target=%.5f",
                        probe.symbol,
                        probe.probe_id,
                        probe.direction,
                        probe.start_price,
                        probe.target_price,
                    )
                    continue
            if now >= probe.expires_at:
                state.missed_total += 1
                LOGGER.info(
                    "MissedOpportunity symbol=%s probe=%s hit=0 direction=%s start=%.5f target=%.5f",
                    probe.symbol,
                    probe.probe_id,
                    probe.direction,
                    probe.start_price,
                    probe.target_price,
                )
                continue
            active.append(probe)
            active_keys.add(probe.miss_key)
        state.probes = active
        state.recent_probe_keys = active_keys

    def compute_bias(self, symbol: str, data: StrategyDataBundle) -> BiasState:
        state = self._state(symbol)
        profile = self._profile(data)
        bias_candles = self._candles_for_tf(data, profile.bias_tf)
        if not bias_candles:
            neutral = BiasState(
                symbol=symbol,
                strategy_name=self.name,
                direction="NEUTRAL",
                timeframe=profile.bias_tf,
                updated_at=data.now,
                metadata={"reason": f"{profile.bias_tf}_MISSING"},
            )
            state.bias = neutral
            return neutral

        closes = [c.close for c in bias_candles]
        ema_values = ema(closes, period=50)
        ema_now = latest_value(ema_values)
        atr_bias = latest_value(atr(bias_candles, self.config.indicators.atr_period))
        last_close = closes[-1]

        direction = "NEUTRAL"
        if ema_now is not None:
            if last_close > ema_now:
                direction = "LONG"
            elif last_close < ema_now:
                direction = "SHORT"

        bos_candles = self._candles_for_tf(data, profile.bos_tf)
        bos_state, _, _ = self._bos_state(bos_candles)
        h1_context = self._h1_context(self._candles_for_tf(data, "H1"))

        bias = BiasState(
            symbol=symbol,
            strategy_name=self.name,
            direction=direction,
            timeframe=profile.bias_tf,
            updated_at=data.now,
            metadata={
                "last_close": last_close,
                "bias_tf": profile.bias_tf,
                "bos_tf": profile.bos_tf,
                "ema50": ema_now,
                "atr_bias_tf": atr_bias,
                "bos_state": bos_state,
                **h1_context,
            },
        )
        state.bias = bias
        return bias
    def detect_candidates(self, symbol: str, data: StrategyDataBundle) -> list[SetupCandidate]:
        state = self._state(symbol)
        profile = self._profile(data)
        bias = state.bias or self.compute_bias(symbol, data)
        m5 = closed_candles(data.candles_m5)
        if not m5:
            return list(state.queue)
        if not data.m5_new_close:
            return list(state.queue)

        if profile.candidate_min_interval_minutes > 0 and state.last_candidate_at is not None:
            if data.now < state.last_candidate_at + timedelta(minutes=profile.candidate_min_interval_minutes):
                return list(state.queue)

        atr_m5 = latest_value(atr(m5, self.config.indicators.atr_period))
        if atr_m5 is None or atr_m5 <= 0:
            return list(state.queue)

        sides: list[str] = []
        if bias.direction in {"LONG", "SHORT"}:
            sides = [bias.direction]
        elif profile.allow_neutral_bias:
            sides = ["LONG", "SHORT"]
        if not sides:
            return list(state.queue)

        minimal_tick_buffer = float(data.extra.get("minimal_tick_buffer", 0.05))
        if profile.candidate_relaxed:
            minimal_tick_buffer *= profile.relaxed_tick_buffer_multiplier
        minimal_tick_buffer = max(1e-9, minimal_tick_buffer)

        created_any = False
        side_diagnostics: list[str] = []
        for side in sides:
            sweep = detect_sweep_reject(
                m5,
                side=side,
                lookback_min_hours=profile.sweep_min_hours,
                lookback_max_hours=profile.sweep_max_hours,
                atr_m15=atr_m5,
                threshold_atr_multiplier=profile.sweep_threshold_multiplier,
                minimal_tick_buffer=minimal_tick_buffer,
                fractal_left=self.config.swings.fractal_left,
                fractal_right=self.config.swings.fractal_right,
            )
            if sweep is None:
                side_diagnostics.append(
                    f"{side}:NO_SWEEP(min_h={profile.sweep_min_hours},max_h={profile.sweep_max_hours},thr={profile.sweep_threshold_multiplier:.3f})"
                )
                continue
            sweep_ts = sweep.sweep_time.replace(second=0, microsecond=0).isoformat()
            setup_id = f"{symbol}:{side}:{sweep_ts}:{sweep.reference_swing_index}"
            if any(item.metadata.get("setup_id") == setup_id for item in state.queue):
                side_diagnostics.append(f"{side}:DUP_SETUP")
                continue
            candidate = SetupCandidate(
                candidate_id=f"SCALP-{symbol}-{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                strategy_name=self.name,
                side=side,
                created_at=data.now,
                expires_at=data.now + timedelta(minutes=profile.candidate_ttl_minutes),
                source_timeframe=self.config.scalp.trigger_timeframe,
                setup_type="SWEEP_REJECT",
                origin_strategy=self.name,
                setup_id=setup_id,
                metadata={
                    "setup_id": setup_id,
                    "sweep_level": sweep.swept_level,
                    "sweep_magnitude": sweep.magnitude,
                    "sweep_time": sweep.sweep_time.isoformat(),
                    "atr_m5": atr_m5,
                },
            )
            state.queue.append(candidate)
            created_any = True

        if created_any:
            state.last_candidate_at = data.now
        elif data.m5_new_close and self.config.monitoring.log_decision_reasons and side_diagnostics:
            LOGGER.info(
                "SCALP candidate wait symbol=%s bias=%s diag=%s",
                symbol,
                bias.direction,
                ";".join(side_diagnostics),
            )
        return list(state.queue)

    def evaluate_candidate(
        self,
        symbol: str,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategyEvaluation:
        state = self._state(symbol)
        profile = self._profile(data)
        bias = state.bias or self.compute_bias(symbol, data)
        m5 = closed_candles(data.candles_m5)
        if not m5:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["SCALP_M5_MISSING"],
                would_enter_if=["M5_CANDLES_READY"],
                snapshot={"symbol": symbol},
                metadata={"setup_state": "WAIT_REACTION"},
            )
        if candidate.expires_at <= data.now:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["SCALP_CANDIDATE_EXPIRED"],
                would_enter_if=["NEW_CANDIDATE"],
                snapshot={"symbol": symbol, "candidate_id": candidate.candidate_id},
                metadata={"setup_state": "WAIT_MITIGATION"},
            )

        atr_m5 = latest_value(atr(m5, self.config.indicators.atr_period))
        if atr_m5 is None or atr_m5 <= 0:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["SCALP_ATR_WARMUP"],
                would_enter_if=["ATR_READY"],
                snapshot={"symbol": symbol},
                metadata={"setup_state": "WAIT_REACTION"},
            )

        sweep_time_raw = candidate.metadata.get("sweep_time")
        if sweep_time_raw is None:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=0.0,
                reasons_blocking=["SCALP_SWEEP_TIME_MISSING"],
                would_enter_if=["VALID_SWEEP"],
                snapshot={"symbol": symbol},
                metadata={"setup_state": "WAIT_MITIGATION"},
            )

        sweep_dt = datetime.fromisoformat(str(sweep_time_raw))
        sweep_idx = index_at_or_after(m5, sweep_dt)
        if sweep_idx is None:
            return StrategyEvaluation(
                action=DecisionAction.OBSERVE,
                score_total=20.0,
                reasons_blocking=["SCALP_WAIT_AFTER_SWEEP"],
                would_enter_if=["MSS_AFTER_SWEEP"],
                snapshot={"symbol": symbol, "atr_m5": atr_m5},
                metadata={"setup_state": "WAIT_REACTION"},
            )

        mss = detect_mss(
            m5,
            side=candidate.side,
            since_index=sweep_idx,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        mss_ok = mss is not None

        displacement_ok = False
        displacement_ratio = 0.0
        displacement_threshold = profile.displacement_multiplier * atr_m5
        if mss is not None:
            body = real_body(m5[mss.candle_index])
            displacement_ratio = body / max(displacement_threshold, 1e-9)
            displacement_ok = body > displacement_threshold

        # Search from sweep origin instead of MSS-only window to avoid missing valid FVG
        # that forms between sweep and later MSS confirmation.
        fvg_start_index = max(0, sweep_idx)
        fvg = detect_latest_fvg(
            m5,
            side=candidate.side,
            start_index=fvg_start_index,
        )
        fvg_size = (fvg.upper - fvg.lower) if fvg is not None else 0.0
        fvg_min_size = profile.fvg_min_atr * atr_m5
        fvg_ok = fvg is not None and fvg_size >= fvg_min_size

        spread = data.spread
        spread_med = median_spread(data.spread_history, 100) if data.spread_history else None
        spread_score = 0.0
        spread_ok_soft = True
        if spread is not None and spread_med is not None and spread_med > 0:
            spread_ratio = spread / spread_med
            if spread_ratio <= 1.5:
                spread_score = 5.0
            elif spread_ratio <= 2.0:
                spread_score = 2.0
            else:
                spread_score = 0.0
                spread_ok_soft = False
        elif spread is not None:
            spread_score = 3.0

        sweep_mag = float(candidate.metadata.get("sweep_magnitude", 0.0))
        sweep_score = min(20.0, 20.0 * (sweep_mag / max(atr_m5, 1e-9)))
        neutral_bias_allowed = bias.direction == "NEUTRAL" and profile.allow_neutral_bias
        bias_score = 20.0 if bias.direction == candidate.side else 8.0 if neutral_bias_allowed else 0.0
        mss_score = 20.0 if mss_ok else 0.0
        disp_score = max(0.0, min(20.0, 20.0 * displacement_ratio))
        fvg_score = 15.0 if fvg_ok else 0.0
        trigger_confirmations = sum([int(mss_ok), int(displacement_ok), int(fvg_ok)])

        penalties: dict[str, float] = {}
        hard_reasons: list[str] = []

        h1_bos_state = str(bias.metadata.get("h1_bos_state", "NONE")).upper()
        if profile.h1_bos_mode == "BLOCK" and h1_bos_state == "NONE":
            hard_reasons.append("H1_NO_BOS")
        elif profile.h1_bos_mode == "SCORE" and h1_bos_state == "NONE":
            penalties["H1_NO_BOS"] = profile.h1_no_bos_penalty

        if bias.direction == "NEUTRAL":
            if profile.allow_neutral_bias:
                penalties["H1_BIAS_NEUTRAL"] = profile.neutral_bias_penalty
            else:
                hard_reasons.append("H1_BIAS_NEUTRAL")
        elif bias.direction != candidate.side:
            hard_reasons.append("SCALP_BIAS_MISMATCH")

        pd_fail = False
        if bias.metadata.get("h1_pd_available"):
            if candidate.side == "LONG":
                pd_fail = bool(bias.metadata.get("h1_pd_long_fail"))
            else:
                pd_fail = bool(bias.metadata.get("h1_pd_short_fail"))
        if pd_fail:
            if profile.pd_filter_mode == "BLOCK":
                hard_reasons.append("H1_PD_FAIL")
            elif profile.pd_filter_mode == "SCORE":
                penalties["H1_PD_FAIL"] = profile.pd_fail_penalty

        score_breakdown = {
            "bias": round(bias_score, 2),
            "sweep": round(sweep_score, 2),
            "mss": round(mss_score, 2),
            "displacement": round(disp_score, 2),
            "fvg": round(fvg_score, 2),
            "spread": round(spread_score, 2),
        }
        for key, value in penalties.items():
            score_breakdown[f"penalty_{key.lower()}"] = round(-value, 2)

        if not mss_ok:
            hard_reasons.append("SCALP_NO_MSS")
        if not displacement_ok:
            hard_reasons.append("SCALP_NO_DISPLACEMENT")
        if not fvg_ok:
            hard_reasons.append("SCALP_NO_FVG")
        if data.news_blocked:
            hard_reasons.append("NEWS_BLOCKED")

        score_total = round(max(0.0, sum(score_breakdown.values())), 2)
        hard_ok = not hard_reasons

        action = DecisionAction.OBSERVE
        if hard_ok and score_total >= self.config.scalp.trade_score_threshold:
            action = DecisionAction.TRADE
        elif hard_ok and self.config.scalp.small_score_min <= score_total <= self.config.scalp.small_score_max:
            action = DecisionAction.SMALL

        risk_multiplier_override: float | None = None
        if neutral_bias_allowed and action in {DecisionAction.TRADE, DecisionAction.SMALL}:
            if profile.neutral_bias_trade_mode == "SMALL":
                action = DecisionAction.SMALL
            risk_multiplier_override = profile.neutral_bias_risk_multiplier
        elif action == DecisionAction.SMALL:
            risk_multiplier_override = self.config.scalp.small_risk_multiplier

        reasons_blocking: list[str] = list(dict.fromkeys(hard_reasons))
        if action == DecisionAction.OBSERVE and not reasons_blocking:
            reasons_blocking.append("SCALP_SCORE_TOO_LOW")
        if action == DecisionAction.OBSERVE and not spread_ok_soft and "SCALP_SPREAD_ELEVATED" not in reasons_blocking:
            reasons_blocking.append("SCALP_SPREAD_ELEVATED")

        would_enter_if: list[str] = []
        if action == DecisionAction.OBSERVE:
            if reasons_blocking:
                would_enter_if = [item.replace("SCALP_", "") for item in reasons_blocking]
            else:
                would_enter_if = ["SCORE_THRESHOLD"]
            if data.m5_new_close:
                setup_id = str(candidate.metadata.get("setup_id", candidate.candidate_id))
                self._register_miss_probe(
                    symbol=symbol,
                    setup_id=setup_id,
                    direction=candidate.side,
                    start_price=m5[-1].close,
                    start_ts=m5[-1].timestamp,
                    atr_value=atr_m5,
                    data=data,
                )
        setup_state = "READY"
        if not fvg_ok:
            setup_state = "WAIT_MITIGATION"
        elif not (mss_ok and displacement_ok):
            setup_state = "WAIT_REACTION"

        signal_meta: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "setup_id": candidate.metadata.get("setup_id"),
            "side": candidate.side,
            "score_total": score_total,
            "score_breakdown": score_breakdown,
            "sweep_level": candidate.metadata.get("sweep_level"),
            "spread": spread,
            "atr_m5": atr_m5,
            "action": action.value,
            "h1_bos_state": h1_bos_state,
            "h1_pd_fail": pd_fail,
            "trigger_confirmations": trigger_confirmations,
            "execution_penalty": 0.0 if spread_ok_soft else 2.0,
            "setup_state": setup_state,
            "sweep_index": sweep_idx,
            "fvg_search_start_index": fvg_start_index,
            "fvg_detected": fvg is not None,
            "fvg_size": fvg_size,
            "fvg_min_size": fvg_min_size,
        }
        if penalties:
            signal_meta["score_penalties"] = penalties
        if risk_multiplier_override is not None:
            signal_meta["risk_multiplier_override"] = risk_multiplier_override
        if bias.metadata.get("h1_pd_eq") is not None:
            signal_meta["h1_pd_eq"] = bias.metadata.get("h1_pd_eq")
            signal_meta["h1_pd_low"] = bias.metadata.get("h1_pd_low")
            signal_meta["h1_pd_high"] = bias.metadata.get("h1_pd_high")
            signal_meta["h1_close"] = bias.metadata.get("h1_close")
        if not spread_ok_soft:
            signal_meta["spread_warning"] = "ELEVATED"
        if fvg is not None:
            signal_meta["fvg_mid"] = fvg.midpoint
            signal_meta["fvg_lower"] = fvg.lower
            signal_meta["fvg_upper"] = fvg.upper
            signal_meta["fvg_c1_index"] = fvg.c1_index
            signal_meta["fvg_c3_index"] = fvg.c3_index
        if mss is not None:
            signal_meta["mss_index"] = mss.candle_index
            signal_meta["displacement_threshold"] = displacement_threshold

        evaluation = StrategyEvaluation(
            action=action,
            score_total=score_total,
            score_breakdown=score_breakdown,
            reasons_blocking=reasons_blocking,
            would_enter_if=would_enter_if,
            snapshot={
                "symbol": symbol,
                "bias": bias.direction,
                "spread": spread,
                "spread_median": spread_med,
                "atr_m5": atr_m5,
                "h1_bos_state": h1_bos_state,
                "h1_pd_eq": bias.metadata.get("h1_pd_eq"),
                "h1_close": bias.metadata.get("h1_close"),
            },
            metadata=signal_meta,
        )
        state.last_evaluation = evaluation
        return evaluation

    def generate_order(
        self,
        symbol: str,
        evaluation: StrategyEvaluation,
        candidate: SetupCandidate,
        data: StrategyDataBundle,
    ) -> StrategySignal | None:
        if evaluation.action not in {DecisionAction.TRADE, DecisionAction.SMALL}:
            return None

        fvg_mid = evaluation.metadata.get("fvg_mid")
        sweep_level = evaluation.metadata.get("sweep_level")
        atr_m5 = evaluation.metadata.get("atr_m5")
        side = str(evaluation.metadata.get("side", candidate.side))
        if fvg_mid is None or sweep_level is None or atr_m5 is None:
            return None

        entry = float(fvg_mid)
        atr_value = float(atr_m5)
        rr = 3.0 if (evaluation.score_total or 0.0) >= 85.0 else 2.0
        if side == "LONG":
            stop = float(sweep_level) - (0.2 * atr_value)
            risk_distance = entry - stop
            if risk_distance <= 0:
                return None
            take_profit = entry + (rr * risk_distance)
        else:
            stop = float(sweep_level) + (0.2 * atr_value)
            risk_distance = stop - entry
            if risk_distance <= 0:
                return None
            take_profit = entry - (rr * risk_distance)

        reason_codes = [self.name, f"SCORE_{int(evaluation.score_total or 0)}"]
        if evaluation.action == DecisionAction.SMALL:
            reason_codes.append("SCALP_SMALL")

        risk_multiplier = evaluation.metadata.get("risk_multiplier_override")
        if risk_multiplier is None:
            risk_multiplier = (
                self.config.scalp.small_risk_multiplier
                if evaluation.action == DecisionAction.SMALL
                else 1.0
            )

        signal = StrategySignal(
            side=side,
            entry_price=entry,
            stop_price=stop,
            take_profit=take_profit,
            rr=rr,
            a_plus=rr >= 3.0,
            expires_at=data.now + timedelta(minutes=self.config.execution.limit_ttl_bars * 5),
            reason_codes=reason_codes,
            metadata={
                "strategy": self.name,
                "candidate_id": candidate.candidate_id,
                "setup_id": candidate.metadata.get("setup_id"),
                "score_total": evaluation.score_total,
                "score_breakdown": evaluation.score_breakdown,
                "risk_multiplier": float(risk_multiplier),
            },
        )
        self._state(symbol).last_order = signal
        return signal

    def manage_position(self, symbol: str, position, data: StrategyDataBundle) -> list[StrategySignal]:
        return []

    def _register_miss_probe(
        self,
        symbol: str,
        setup_id: str,
        direction: str,
        start_price: float,
        start_ts: datetime,
        atr_value: float,
        data: StrategyDataBundle,
    ) -> None:
        if direction not in {"LONG", "SHORT"}:
            return

        profile = self._profile(data)
        rounded_start = self._round_timestamp(start_ts, profile.miss_round_minutes)
        miss_key = f"{symbol}:{self.name}:{setup_id}:{direction}:{rounded_start.isoformat()}"
        state = self._state(symbol)
        if miss_key in state.recent_probe_keys:
            return

        move = self.config.scalp.miss_move_atr * atr_value
        target = start_price + move if direction == "LONG" else start_price - move
        created_at = data.now
        state.recent_probe_keys.add(miss_key)
        state.probes.append(
            _MissProbe(
                probe_id=f"MISS-{uuid.uuid4().hex[:8]}",
                miss_key=miss_key,
                symbol=symbol,
                direction=direction,
                start_price=start_price,
                target_price=target,
                created_at=created_at,
                expires_at=created_at + timedelta(minutes=self.config.scalp.miss_window_minutes),
            )
        )

    def missed_opportunity_rate(self, symbol: str) -> float | None:
        state = self._state(symbol)
        if state.missed_total <= 0:
            return None
        return round(state.missed_hits / state.missed_total, 4)

    def missed_opportunity_stats(self, symbol: str) -> tuple[int, int]:
        state = self._state(symbol)
        return state.missed_hits, state.missed_total
