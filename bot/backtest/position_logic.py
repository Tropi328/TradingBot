"""Position exit, TP management, trailing stop — extracted from engine.py."""

from __future__ import annotations

from datetime import UTC, datetime

from bot.backtest.costs import _convert_cash_to_account
from bot.backtest.models import _OpenPosition
from bot.data.candles import Candle
from bot.execution.fx import FxConverter


def _calc_exit(
    position: _OpenPosition,
    candle: Candle,
    *,
    assumed_spread: float,
    slippage: float,
) -> tuple[bool, float, str]:
    if position.side == "LONG":
        stop_hit = candle.low <= position.stop
        tp_hit = candle.high >= position.tp
    else:
        stop_hit = candle.high >= position.stop
        tp_hit = candle.low <= position.tp
    # Defensive: if TP is on the wrong side of actual entry (adverse fill),
    # the TP exit would be a loss — invalidate it.
    if tp_hit:
        if (position.side == "LONG" and position.tp <= position.entry) or (
            position.side == "SHORT" and position.tp >= position.entry
        ):
            tp_hit = False
    if not stop_hit and not tp_hit:
        return False, 0.0, ""
    # Conservative fill order when both happen in one bar.
    reason = "STOP" if stop_hit else "TP"
    if reason == "STOP":
        be_stop = False
        if position.be_moved:
            if position.side == "LONG":
                be_stop = position.stop >= (position.entry - 1e-9)
            else:
                be_stop = position.stop <= (position.entry + 1e-9)
        if be_stop:
            reason = "BE"
        if position.side == "LONG":
            fill_price = position.stop - slippage
            max_loss_price = position.entry - (position.initial_risk * position.max_loss_r_cap)
            fill_price = max(fill_price, max_loss_price)
        else:
            fill_price = position.stop + slippage
            max_loss_price = position.entry + (position.initial_risk * position.max_loss_r_cap)
            fill_price = min(fill_price, max_loss_price)
        return True, fill_price, reason

    # TP is a limit order: fill at the TP price level, not the bar's bid/ask.
    # Slippage on limit exits is conservative (goes against us).
    if position.side == "LONG":
        fill_price = position.tp - slippage
        # Final safety: slippage must not push TP fill below entry.
        if fill_price < position.entry:
            return False, 0.0, ""
    else:
        fill_price = position.tp + slippage
        if fill_price > position.entry:
            return False, 0.0, ""
    return True, fill_price, reason


def _append_live_placeholder(candles: list[Candle], timeframe_minutes: int) -> list[Candle]:
    if not candles:
        return []
    last = candles[-1]
    return candles + [_live_placeholder_from(last, timeframe_minutes)]


def _live_placeholder_from(last: Candle, timeframe_minutes: int) -> Candle:
    live_ts = datetime.fromtimestamp(last.timestamp.timestamp() + (timeframe_minutes * 60), tz=UTC)
    return Candle(
        timestamp=live_ts,
        open=last.close,
        high=last.close,
        low=last.close,
        close=last.close,
        bid=last.bid,
        ask=last.ask,
        volume=0.0,
    )


def _estimate_structure_target(
    *,
    side: str,
    entry: float,
    candles: list[Candle],
    lookback_bars: int,
) -> float | None:
    if not candles:
        return None
    recent = candles[-max(2, int(lookback_bars)) :]
    if side == "LONG":
        candidates = [float(candle.high) for candle in recent if float(candle.high) > entry]
        return max(candidates) if candidates else None
    candidates = [float(candle.low) for candle in recent if float(candle.low) < entry]
    return min(candidates) if candidates else None


def _normalize_tp_by_r(
    *,
    side: str,
    entry: float,
    stop: float,
    requested_tp: float,
    min_r: float,
    max_r: float,
) -> tuple[float, float]:
    risk = abs(entry - stop)
    if risk <= 0:
        return requested_tp, 0.0
    if side == "LONG":
        requested_r = (requested_tp - entry) / risk
    else:
        requested_r = (entry - requested_tp) / risk
    target_r = max(min_r, min(max_r, float(requested_r)))
    if side == "LONG":
        tp = entry + (risk * target_r)
    else:
        tp = entry - (risk * target_r)
    return tp, target_r


def _resolve_tp_target_r(
    *,
    is_a_plus: bool,
    score: float | None,
    tuning: object,
) -> tuple[float, str]:
    """Return (target_total_r, profile_label) based on score gradient or a_plus fallback."""
    if getattr(tuning, "tp_gradient_enabled", False) and score is not None:
        high_score = float(getattr(tuning, "tp_gradient_tier_high_score", 75.0))
        mid_score = float(getattr(tuning, "tp_gradient_tier_mid_score", 62.0))
        if score >= high_score:
            return float(getattr(tuning, "tp_gradient_tier_high_r", 3.0)), "GRADIENT_HIGH"
        if score >= mid_score:
            return float(getattr(tuning, "tp_gradient_tier_mid_r", 2.5)), "GRADIENT_MID"
        return float(getattr(tuning, "tp_gradient_tier_low_r", 2.0)), "GRADIENT_LOW"
    if is_a_plus:
        return float(tuning.tp_target_a_plus_r), "A_PLUS_3R"
    return float(tuning.tp_target_min_r), "STANDARD_2R"


def _expected_rr(
    *,
    side: str,
    entry: float,
    stop: float,
    target: float,
) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if side == "LONG":
        reward = target - entry
    else:
        reward = entry - target
    if reward <= 0:
        return 0.0
    return reward / risk


def _manage_open_position(
    *,
    position: _OpenPosition,
    candle: Candle,
    candles_m5: list[Candle],
    index: int,
    spread: float = 0.0,
    slippage: float,
    fx_converter: FxConverter | None = None,
    instrument_currency: str = "USD",
    account_currency: str = "USD",
    fx_apply_to: set[str] | None = None,
) -> tuple[bool, bool]:
    risk_dist = max(0.0, float(position.initial_risk))
    if risk_dist <= 0:
        return False, False

    tp1_hit = False
    be_moved_now = False
    if not position.tp1_taken:
        if position.side == "LONG":
            tp1_level = position.entry + (risk_dist * position.tp1_trigger_r)
            reached_tp1 = candle.high >= tp1_level
            partial_fill = tp1_level - slippage
        else:
            tp1_level = position.entry - (risk_dist * position.tp1_trigger_r)
            reached_tp1 = candle.low <= tp1_level
            partial_fill = tp1_level + slippage
        if reached_tp1:
            close_size = position.size * position.tp1_fraction
            close_size = max(0.0, min(position.size, close_size))
            if close_size > 0:
                position.spread_cost_total += max(0.0, float(spread)) * 0.5 * close_size
                position.slippage_cost_total += abs(float(slippage)) * close_size
                if position.side == "LONG":
                    partial_pnl_instr = (partial_fill - position.entry) * close_size
                else:
                    partial_pnl_instr = (position.entry - partial_fill) * close_size
                partial_pnl, partial_fx_cost = _convert_cash_to_account(
                    amount=partial_pnl_instr,
                    category="pnl",
                    fx_converter=fx_converter,
                    instrument_currency=instrument_currency,
                    account_currency=account_currency,
                    fx_apply_to=(fx_apply_to or {"pnl", "swap", "commission"}),
                )
                position.realized_partial += partial_pnl
                position.fx_conversion_total += partial_fx_cost
                position.fx_cost_total += partial_fx_cost
                position.size -= close_size
                position.tp1_taken = True
                position.tp1_hit_index = int(index)
                tp1_hit = True

    if position.tp1_taken and not position.be_moved and position.tp1_hit_index is not None:
        elapsed_since_tp1 = int(index - position.tp1_hit_index)
        if elapsed_since_tp1 >= int(max(0, position.be_delay_bars_after_tp1)):
            if position.side == "LONG":
                confirm_ok = candle.close >= position.entry
                be_price = position.entry + (risk_dist * position.be_offset_r)
                if confirm_ok and be_price > position.stop:
                    position.stop = be_price
                    be_moved_now = True
            else:
                confirm_ok = candle.close <= position.entry
                be_price = position.entry - (risk_dist * position.be_offset_r)
                if confirm_ok and be_price < position.stop:
                    position.stop = be_price
                    be_moved_now = True
            if be_moved_now:
                position.be_moved = True

    if position.trailing_after_tp1 and position.tp1_taken and position.size > 0 and position.be_moved:
        window = max(2, int(position.trailing_window_bars))
        recent = candles_m5[max(0, index - window + 1) : index + 1]
        if recent:
            buffer_value = risk_dist * max(0.0, float(position.trailing_buffer_r))
            if position.side == "LONG":
                swing_low = min(float(item.low) for item in recent)
                trail_stop = swing_low - buffer_value
                if trail_stop > position.stop:
                    position.stop = trail_stop
            else:
                swing_high = max(float(item.high) for item in recent)
                trail_stop = swing_high + buffer_value
                if trail_stop < position.stop:
                    position.stop = trail_stop

    return tp1_hit, be_moved_now
