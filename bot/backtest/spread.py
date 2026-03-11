"""Dynamic spread resolution — extracted from engine.py."""

from __future__ import annotations

from bot.backtest.math_utils import _clamp, _quantile_from_sorted
from bot.config import AppConfig
from bot.data.candles import Candle


def _resolve_dynamic_spread_bounds(
    *,
    config: AppConfig,
    symbol: str,
    fallback_spread: float,
) -> tuple[float, float] | None:
    tuning = config.backtest_tuning
    if not bool(tuning.dynamic_assumed_spread_enabled):
        return None
    symbol_norm = str(symbol).strip().upper()
    min_map = tuning.dynamic_assumed_spread_min_by_symbol
    max_map = tuning.dynamic_assumed_spread_max_by_symbol
    min_spread = min_map.get(symbol_norm)
    max_spread = max_map.get(symbol_norm)
    if min_spread is None and max_spread is None:
        return None
    fallback = max(0.0, float(fallback_spread))
    lo = float(min_spread) if min_spread is not None else fallback
    hi = float(max_spread) if max_spread is not None else fallback
    lo = max(0.0, lo)
    hi = max(0.0, hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _build_dynamic_assumed_spread_series(
    *,
    candles_m5: list[Candle],
    atr_values: list[float | None],
    min_spread: float,
    max_spread: float,
) -> list[float]:
    count = len(candles_m5)
    if count == 0:
        return []
    lo = max(0.0, float(min_spread))
    hi = max(lo, float(max_spread))
    if hi - lo <= 1e-12:
        return [lo] * count

    atr_clean: list[float] = []
    for value in atr_values:
        if value is None:
            continue
        try:
            atr_value = float(value)
        except (TypeError, ValueError):
            continue
        if atr_value > 0:
            atr_clean.append(atr_value)

    if not atr_clean:
        mid = (lo + hi) * 0.5
        return [mid] * count

    atr_sorted = sorted(atr_clean)
    q10 = _quantile_from_sorted(atr_sorted, 0.10)
    q90 = _quantile_from_sorted(atr_sorted, 0.90)
    if q90 <= q10:
        mid = (lo + hi) * 0.5
        return [mid] * count

    out: list[float] = []
    for index in range(count):
        atr_value_raw = atr_values[index] if index < len(atr_values) else None
        ratio = 0.5
        if atr_value_raw is not None:
            try:
                atr_value = float(atr_value_raw)
                if atr_value > 0:
                    ratio = _clamp((atr_value - q10) / (q90 - q10), 0.0, 1.0)
            except (TypeError, ValueError):
                ratio = 0.5
        out.append(lo + ((hi - lo) * ratio))
    return out
