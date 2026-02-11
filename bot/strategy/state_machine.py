from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from bot.config import AppConfig
from bot.data.candles import Candle
from bot.strategy.bias import (
    BiasDecision,
    dealing_range,
    determine_h1_bias,
    premium_discount_state,
)
from bot.strategy.filters import spread_is_ok
from bot.strategy.ict import (
    FVGSignal,
    MSSSignal,
    SweepSignal,
    detect_latest_fvg,
    detect_mss,
    detect_sweep_reject,
    is_a_plus_setup,
)
from bot.strategy.indicators import atr, latest_value, real_body
from bot.strategy.swings import index_at_or_after


@dataclass(slots=True)
class StrategySignal:
    side: str
    entry_price: float
    stop_price: float
    take_profit: float
    rr: float
    a_plus: bool
    expires_at: datetime
    reason_codes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyDecision:
    signal: StrategySignal | None
    reason_codes: list[str]
    bias: str
    pd_state: str
    sweep_ok: bool
    mss_ok: bool
    displacement_ok: bool
    fvg_ok: bool
    spread_ok: bool
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class H1Snapshot:
    bias: str
    side: str | None
    safe_mode: bool
    pd_state: str
    pd_ok: bool
    ema200_ready: bool
    ema200_value: float | None
    bos_state: str
    bos_age: int | None
    bars: int
    required_bars: int
    last_swing_high: float | None
    last_swing_low: float | None
    dealing_low: float | None
    dealing_high: float | None
    eq: float | None
    last_close: float
    reason_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class M15Snapshot:
    setup_state: str
    sweep_dir: str
    reject_ok: bool
    sweep_level: float | None
    invalidation_level: float | None
    setup_age_minutes: float | None
    atr_m15: float | None
    sweep: SweepSignal | None
    reason_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class M5Snapshot:
    entry_state: str
    mss_ok: bool
    displacement_ok: bool
    fvg_ok: bool
    fvg_range: tuple[float, float] | None
    fvg_mid: float | None
    limit_price: float | None
    reason_codes: list[str] = field(default_factory=list)


def _merge_reason_codes(*chunks: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for item in chunk:
            if item in seen:
                continue
            merged.append(item)
            seen.add(item)
    return merged


def _bos_state_and_age(
    candles_h1: list[Candle],
    last_swing_high: float | None,
    last_swing_low: float | None,
) -> tuple[str, int | None]:
    if not candles_h1:
        return "NONE", None
    latest_close = candles_h1[-1].close
    if last_swing_high is not None and latest_close > last_swing_high:
        state = "UP"
        level = last_swing_high
    elif last_swing_low is not None and latest_close < last_swing_low:
        state = "DOWN"
        level = last_swing_low
    else:
        return "NONE", None

    for idx in range(len(candles_h1) - 1, -1, -1):
        close_val = candles_h1[idx].close
        if state == "UP" and close_val > level:
            return state, (len(candles_h1) - 1) - idx
        if state == "DOWN" and close_val < level:
            return state, (len(candles_h1) - 1) - idx
    return state, None


class StrategyEngine:
    def __init__(self, config: AppConfig):
        self.config = config

    def _no_signal(
        self,
        *,
        reasons: list[str],
        bias: str,
        pd_state: str,
        sweep_ok: bool,
        mss_ok: bool,
        displacement_ok: bool,
        fvg_ok: bool,
        spread_ok: bool,
        payload: dict[str, Any],
    ) -> StrategyDecision:
        return StrategyDecision(
            signal=None,
            reason_codes=_merge_reason_codes(reasons),
            bias=bias,
            pd_state=pd_state,
            sweep_ok=sweep_ok,
            mss_ok=mss_ok,
            displacement_ok=displacement_ok,
            fvg_ok=fvg_ok,
            spread_ok=spread_ok,
            payload=payload,
        )

    def evaluate_h1(self, candles_h1: list[Candle]) -> H1Snapshot:
        bars = len(candles_h1)
        required = self.config.indicators.ema_period_h1
        if not candles_h1:
            return H1Snapshot(
                bias="NEUTRAL",
                side=None,
                safe_mode=False,
                pd_state="UNKNOWN",
                pd_ok=False,
                ema200_ready=False,
                ema200_value=None,
                bos_state="NONE",
                bos_age=None,
                bars=0,
                required_bars=required,
                last_swing_high=None,
                last_swing_low=None,
                dealing_low=None,
                dealing_high=None,
                eq=None,
                last_close=0.0,
                reason_codes=["H1_INSUFFICIENT_BARS"],
            )

        bias_decision: BiasDecision = determine_h1_bias(
            candles_h1,
            ema_period=required,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        low, high, eq = dealing_range(
            bias_decision.last_swing_high,
            bias_decision.last_swing_low,
        )
        pd_state = premium_discount_state(bias_decision.last_close, low, high)
        bos_state, bos_age = _bos_state_and_age(
            candles_h1,
            bias_decision.last_swing_high,
            bias_decision.last_swing_low,
        )
        side = "LONG" if bias_decision.bias == "UP" else "SHORT" if bias_decision.bias == "DOWN" else None
        safe_mode = False
        if (
            side is None
            and self.config.strategy_runtime.allow_neutral_bias
            and bos_state in {"UP", "DOWN"}
            and bias_decision.ema_value is not None
        ):
            side = "LONG" if bos_state == "UP" else "SHORT"
            safe_mode = True

        if side == "LONG":
            pd_ok = pd_state == "DISCOUNT"
        elif side == "SHORT":
            pd_ok = pd_state == "PREMIUM"
        else:
            pd_ok = False
        reasons: list[str] = []
        if bars < required or bias_decision.ema_value is None:
            reasons.append("H1_EMA_WARMUP")
        if bias_decision.last_swing_high is None or bias_decision.last_swing_low is None:
            reasons.append("H1_INSUFFICIENT_SWINGS")
        if bos_state == "NONE":
            reasons.append("H1_NO_BOS")
        if bias_decision.bias == "NEUTRAL" and not safe_mode:
            reasons.append("H1_BIAS_NEUTRAL")
        if safe_mode:
            reasons.append("H1_NEUTRAL_SAFE_MODE")
        if pd_state == "UNKNOWN":
            reasons.append("H1_PD_UNKNOWN")
        elif not pd_ok:
            reasons.append("H1_PD_FAIL")
            reasons.append("H1_PD_DIAG")

        return H1Snapshot(
            bias=bias_decision.bias,
            side=side,
            safe_mode=safe_mode,
            pd_state=pd_state,
            pd_ok=pd_ok,
            ema200_ready=bias_decision.ema_value is not None,
            ema200_value=bias_decision.ema_value,
            bos_state=bos_state,
            bos_age=bos_age,
            bars=bars,
            required_bars=required,
            last_swing_high=bias_decision.last_swing_high,
            last_swing_low=bias_decision.last_swing_low,
            dealing_low=low,
            dealing_high=high,
            eq=eq,
            last_close=bias_decision.last_close,
            reason_codes=_merge_reason_codes(reasons),
        )

    def evaluate_m15(
        self,
        *,
        candles_m15: list[Candle],
        h1: H1Snapshot,
        minimal_tick_buffer: float,
        now: datetime,
        previous: M15Snapshot | None = None,
    ) -> M15Snapshot:
        if not candles_m15:
            return M15Snapshot(
                setup_state="WAIT",
                sweep_dir="NONE",
                reject_ok=False,
                sweep_level=None,
                invalidation_level=None,
                setup_age_minutes=None,
                atr_m15=None,
                sweep=None,
                reason_codes=["M15_INSUFFICIENT_BARS"],
            )
        if h1.side is None:
            return M15Snapshot(
                setup_state="WAIT",
                sweep_dir="NONE",
                reject_ok=False,
                sweep_level=None,
                invalidation_level=None,
                setup_age_minutes=None,
                atr_m15=None,
                sweep=None,
                reason_codes=_merge_reason_codes(h1.reason_codes, ["M15_H1_GATE_FAIL"]),
            )
        if not h1.pd_ok:
            return M15Snapshot(
                setup_state="WAIT",
                sweep_dir=h1.side,
                reject_ok=False,
                sweep_level=None,
                invalidation_level=None,
                setup_age_minutes=None,
                atr_m15=None,
                sweep=None,
                reason_codes=_merge_reason_codes(h1.reason_codes, ["M15_H1_PD_FAIL"]),
            )

        atr_m15 = latest_value(atr(candles_m15, self.config.indicators.atr_period))
        if atr_m15 is None:
            return M15Snapshot(
                setup_state="WAIT",
                sweep_dir=h1.side,
                reject_ok=False,
                sweep_level=None,
                invalidation_level=None,
                setup_age_minutes=None,
                atr_m15=None,
                sweep=None,
                reason_codes=["M15_ATR_WARMUP"],
            )

        sweep = detect_sweep_reject(
            candles_m15,
            side=h1.side,
            lookback_min_hours=self.config.sweep.lookback_min_hours,
            lookback_max_hours=self.config.sweep.lookback_max_hours,
            atr_m15=atr_m15,
            threshold_atr_multiplier=self.config.sweep.threshold_atr_multiplier,
            minimal_tick_buffer=minimal_tick_buffer,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        if sweep is None:
            state = "EXPIRED" if previous and previous.setup_state == "ARMED" else "WAIT"
            reason = "M15_SETUP_EXPIRED" if state == "EXPIRED" else "M15_NO_SWEEP"
            return M15Snapshot(
                setup_state=state,
                sweep_dir=h1.side,
                reject_ok=False,
                sweep_level=None,
                invalidation_level=None,
                setup_age_minutes=None,
                atr_m15=atr_m15,
                sweep=None,
                reason_codes=[reason],
            )

        setup_age = max(0.0, (now - sweep.sweep_time).total_seconds() / 60.0)
        return M15Snapshot(
            setup_state="ARMED",
            sweep_dir=h1.side,
            reject_ok=sweep.rejected,
            sweep_level=sweep.swept_level,
            invalidation_level=sweep.swept_level,
            setup_age_minutes=setup_age,
            atr_m15=atr_m15,
            sweep=sweep,
            reason_codes=[],
        )

    def evaluate_m5(
        self,
        *,
        epic: str,
        candles_m5: list[Candle],
        current_spread: float | None,
        spread_history: list[float],
        news_blocked: bool,
        h1: H1Snapshot,
        m15: M15Snapshot,
        entry_state: str = "WAIT",
    ) -> tuple[StrategyDecision, M5Snapshot]:
        payload: dict[str, Any] = {
            "epic": epic,
            "bias": h1.bias,
            "ema_h1": h1.ema200_value,
            "last_swing_high_h1": h1.last_swing_high,
            "last_swing_low_h1": h1.last_swing_low,
            "dealing_range_low": h1.dealing_low,
            "dealing_range_high": h1.dealing_high,
            "eq": h1.eq,
            "pd_state": h1.pd_state,
        }
        base_reasons = _merge_reason_codes(h1.reason_codes, m15.reason_codes)
        if not candles_m5:
            reasons = _merge_reason_codes(base_reasons, ["M5_INSUFFICIENT_BARS"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=False,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        if h1.side is None:
            reasons = _merge_reason_codes(base_reasons, ["H1_GATE_FAIL"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=False,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        if not h1.pd_ok:
            reasons = _merge_reason_codes(base_reasons, ["H1_PD_FAIL"])
            payload["pd_diag"] = {
                "close": h1.last_close,
                "eq": h1.eq,
                "low": h1.dealing_low,
                "high": h1.dealing_high,
            }
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=False,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        if m15.sweep is None:
            reasons = _merge_reason_codes(base_reasons, ["M15_NO_SWEEP"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=False,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )
        payload["sweep"] = {
            "level": m15.sweep.swept_level,
            "magnitude": m15.sweep.magnitude,
            "sweep_time": m15.sweep.sweep_time.isoformat(),
            "reference_swing_index": m15.sweep.reference_swing_index,
        }

        sweep_idx_m5 = index_at_or_after(candles_m5, m15.sweep.sweep_time)
        if sweep_idx_m5 is None:
            reasons = _merge_reason_codes(base_reasons, ["M5_WAIT_AFTER_SWEEP"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        mss: MSSSignal | None = detect_mss(
            candles_m5,
            side=h1.side,
            since_index=sweep_idx_m5,
            fractal_left=self.config.swings.fractal_left,
            fractal_right=self.config.swings.fractal_right,
        )
        if mss is None:
            reasons = _merge_reason_codes(base_reasons, ["M5_NO_MSS"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=False,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )
        payload["mss"] = {
            "broken_level": mss.broken_level,
            "candle_index": mss.candle_index,
            "source_swing_index": mss.source_swing_index,
        }

        atr_m5 = latest_value(atr(candles_m5, self.config.indicators.atr_period))
        if atr_m5 is None:
            reasons = _merge_reason_codes(base_reasons, ["M5_ATR_WARMUP"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=True,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=True,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        displacement_body = real_body(candles_m5[mss.candle_index])
        displacement_ok = displacement_body > (self.config.displacement.base_multiplier * atr_m5)
        payload["displacement_body"] = displacement_body
        payload["atr_m5"] = atr_m5
        if not displacement_ok:
            reasons = _merge_reason_codes(base_reasons, ["M5_NO_DISPLACEMENT"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=True,
                    displacement_ok=False,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=True,
                    displacement_ok=False,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        fvg: FVGSignal | None = detect_latest_fvg(
            candles_m5,
            side=h1.side,
            start_index=max(0, mss.candle_index - 2),
        )
        if fvg is None:
            reasons = _merge_reason_codes(base_reasons, ["M5_NO_FVG"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=False,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=False,
                    fvg_range=None,
                    fvg_mid=None,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )
        payload["fvg"] = {
            "lower": fvg.lower,
            "upper": fvg.upper,
            "midpoint": fvg.midpoint,
            "c3_index": fvg.c3_index,
        }

        spread_ok = spread_is_ok(
            current_spread=current_spread,
            spread_history=spread_history,
            window=self.config.spread_filter.window,
            max_multiple_of_median=self.config.spread_filter.max_multiple_of_median,
        )
        if not spread_ok:
            reasons = _merge_reason_codes(base_reasons, ["M5_SPREAD_FAIL"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=True,
                    spread_ok=False,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=True,
                    fvg_range=(fvg.lower, fvg.upper),
                    fvg_mid=fvg.midpoint,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        if news_blocked:
            reasons = _merge_reason_codes(base_reasons, ["NEWS_BLOCKED"])
            return (
                self._no_signal(
                    reasons=reasons,
                    bias=h1.bias,
                    pd_state=h1.pd_state,
                    sweep_ok=True,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=True,
                    spread_ok=True,
                    payload=payload,
                ),
                M5Snapshot(
                    entry_state=entry_state,
                    mss_ok=True,
                    displacement_ok=True,
                    fvg_ok=True,
                    fvg_range=(fvg.lower, fvg.upper),
                    fvg_mid=fvg.midpoint,
                    limit_price=None,
                    reason_codes=reasons,
                ),
            )

        pd_ideal = h1.pd_state in {"DISCOUNT", "PREMIUM"}
        a_plus = is_a_plus_setup(
            pd_ideal=pd_ideal,
            displacement_body=displacement_body,
            atr_m5=atr_m5,
            spread_ok=spread_ok,
            a_plus_multiplier=self.config.displacement.a_plus_multiplier,
        )
        rr = 3.0 if a_plus else 2.0

        entry = fvg.midpoint
        if h1.side == "LONG":
            stop = m15.sweep.swept_level - (0.2 * atr_m5)
            risk_distance = entry - stop
            if risk_distance <= 0:
                reasons = _merge_reason_codes(base_reasons, ["M5_INVALID_RISK_DISTANCE"])
                return (
                    self._no_signal(
                        reasons=reasons,
                        bias=h1.bias,
                        pd_state=h1.pd_state,
                        sweep_ok=True,
                        mss_ok=True,
                        displacement_ok=True,
                        fvg_ok=True,
                        spread_ok=True,
                        payload=payload,
                    ),
                    M5Snapshot(
                        entry_state=entry_state,
                        mss_ok=True,
                        displacement_ok=True,
                        fvg_ok=True,
                        fvg_range=(fvg.lower, fvg.upper),
                        fvg_mid=fvg.midpoint,
                        limit_price=None,
                        reason_codes=reasons,
                    ),
                )
            take_profit = entry + (rr * risk_distance)
        else:
            stop = m15.sweep.swept_level + (0.2 * atr_m5)
            risk_distance = stop - entry
            if risk_distance <= 0:
                reasons = _merge_reason_codes(base_reasons, ["M5_INVALID_RISK_DISTANCE"])
                return (
                    self._no_signal(
                        reasons=reasons,
                        bias=h1.bias,
                        pd_state=h1.pd_state,
                        sweep_ok=True,
                        mss_ok=True,
                        displacement_ok=True,
                        fvg_ok=True,
                        spread_ok=True,
                        payload=payload,
                    ),
                    M5Snapshot(
                        entry_state=entry_state,
                        mss_ok=True,
                        displacement_ok=True,
                        fvg_ok=True,
                        fvg_range=(fvg.lower, fvg.upper),
                        fvg_mid=fvg.midpoint,
                        limit_price=None,
                        reason_codes=reasons,
                    ),
                )
            take_profit = entry - (rr * risk_distance)

        expires_at = candles_m5[-1].timestamp + timedelta(
            minutes=self.config.execution.limit_ttl_bars * 5
        )
        reasons = _merge_reason_codes(base_reasons, ["M5_SIGNAL_READY"])
        if h1.safe_mode:
            reasons = _merge_reason_codes(reasons, ["H1_NEUTRAL_SAFE_MODE"])
        payload["entry"] = entry
        payload["stop"] = stop
        payload["take_profit"] = take_profit
        payload["expires_at"] = expires_at.isoformat()
        payload["rr"] = rr
        payload["safe_mode"] = h1.safe_mode
        payload["risk_multiplier"] = (
            self.config.strategy_runtime.neutral_bias_risk_multiplier if h1.safe_mode else 1.0
        )

        signal = StrategySignal(
            side=h1.side,
            entry_price=entry,
            stop_price=stop,
            take_profit=take_profit,
            rr=rr,
            a_plus=a_plus,
            expires_at=expires_at,
            reason_codes=reasons.copy(),
            metadata=payload,
        )
        return (
            StrategyDecision(
                signal=signal,
                reason_codes=reasons,
                bias=h1.bias,
                pd_state=h1.pd_state,
                sweep_ok=True,
                mss_ok=True,
                displacement_ok=True,
                fvg_ok=True,
                spread_ok=True,
                payload=payload,
            ),
            M5Snapshot(
                entry_state=entry_state,
                mss_ok=True,
                displacement_ok=True,
                fvg_ok=True,
                fvg_range=(fvg.lower, fvg.upper),
                fvg_mid=fvg.midpoint,
                limit_price=entry,
                reason_codes=reasons,
            ),
        )

    def evaluate(
        self,
        *,
        epic: str,
        minimal_tick_buffer: float,
        candles_h1: list[Candle],
        candles_m15: list[Candle],
        candles_m5: list[Candle],
        current_spread: float | None,
        spread_history: list[float],
        news_blocked: bool,
    ) -> StrategyDecision:
        h1 = self.evaluate_h1(candles_h1)
        m15 = self.evaluate_m15(
            candles_m15=candles_m15,
            h1=h1,
            minimal_tick_buffer=minimal_tick_buffer,
            now=candles_m15[-1].timestamp if candles_m15 else datetime.now(timezone.utc),
            previous=None,
        )
        decision, _ = self.evaluate_m5(
            epic=epic,
            candles_m5=candles_m5,
            current_spread=current_spread,
            spread_history=spread_history,
            news_blocked=news_blocked,
            h1=h1,
            m15=m15,
            entry_state="WAIT",
        )
        return decision
