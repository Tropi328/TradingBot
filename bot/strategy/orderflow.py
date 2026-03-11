"""Orderflow analysis — candle-derived heuristic with enricher extension point.

Architecture:
  - **LITE mode** (default and only built-in mode): computes orderflow metrics
    purely from OHLC candle data + optional bid/ask quote.
  - **ExternalOrderflowEnricher** protocol: clean extension point for future
    integrations (e.g. Binance futures L2, CME Gold futures tape).  When an
    enricher is attached to ``CompositeOrderflowProvider`` it post-processes
    the LITE snapshot with real trade/book data.

The legacy FULL mode (which required a manual ``extra["orderflow_full"]``
payload that was never provided in practice) has been removed.  Enrichers
provide a cleaner, more composable replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from bot.data.candles import Candle
from bot.strategy.indicators import atr, latest_value
from bot.strategy.trace import closed_candles


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class OrderflowMetrics:
    delta_ratio: float = 0.0
    aggression: float = 0.0
    obi_k: float = 0.0
    microprice_bias: float = 0.0
    absorption_score: float = 0.0
    chop_score: float = 1.0
    spread_ratio: float = 0.0
    efficiency_ratio: float = 0.0


@dataclass(slots=True)
class OrderflowSnapshot:
    confidence: float
    mode: str
    metrics: OrderflowMetrics = field(default_factory=OrderflowMetrics)
    pressure: float = 0.0
    direction: str = "NEUTRAL"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.upper()
        payload["direction"] = self.direction.upper()
        return payload


def infer_orderflow_direction(metrics: OrderflowMetrics) -> tuple[str, float]:
    signed_aggression = metrics.aggression if metrics.delta_ratio >= 0 else -metrics.aggression
    pressure = (
        (metrics.delta_ratio * 0.45)
        + (metrics.microprice_bias * 0.3)
        + (metrics.obi_k * 0.2)
        + (signed_aggression * 0.05)
    )
    pressure = _clamp(pressure, -1.0, 1.0)
    if abs(pressure) < 0.12:
        return "NEUTRAL", pressure
    return ("LONG" if pressure > 0 else "SHORT"), pressure


# =========================================================================
#  Provider protocol + enricher extension point
# =========================================================================


class OrderflowProvider(Protocol):
    def get_snapshot(
        self,
        symbol: str,
        tf: str,
        window: int,
        *,
        candles: list[Candle] | None = None,
        spread: float | None = None,
        quote: tuple[float, float, float] | None = None,
        atr_value: float | None = None,
        extra: dict[str, Any] | None = None,
        mode_override: str | None = None,
    ) -> OrderflowSnapshot: ...


class ExternalOrderflowEnricher(Protocol):
    """Extension point for future external orderflow data sources.

    Implementations receive a LITE snapshot and can replace/augment
    any metric (e.g. real delta_ratio from Binance futures, real obi_k
    from CME depth).  The enricher should update ``snapshot.mode`` to
    indicate the data source (e.g. ``"BINANCE"``, ``"CME"``).

    Example (future)::

        class BinanceFuturesEnricher:
            def enrich(self, snapshot, symbol):
                book = self._fetch_book(symbol)
                snapshot.metrics.obi_k = compute_obi(book)
                snapshot.mode = "BINANCE"
                return snapshot
    """

    def enrich(self, snapshot: OrderflowSnapshot, symbol: str) -> OrderflowSnapshot: ...


# =========================================================================
#  Composite provider — LITE + optional enricher
# =========================================================================


class CompositeOrderflowProvider(OrderflowProvider):
    """Orderflow provider using candle-derived LITE heuristics.

    Accepts an optional *enricher* that post-processes the LITE snapshot
    with external data.  When no enricher is provided (default), the
    provider operates in pure LITE mode.

    Legacy parameters (``default_mode``, ``symbol_modes``) are accepted
    for backward compatibility but ignored -- mode is always LITE unless
    an enricher overrides it.
    """

    def __init__(
        self,
        *,
        default_mode: str = "LITE",
        symbol_modes: dict[str, str] | None = None,
        enricher: ExternalOrderflowEnricher | None = None,
    ) -> None:
        # Legacy params accepted for backward compat -- not used.
        self._enricher = enricher

    def get_snapshot(
        self,
        symbol: str,
        tf: str,
        window: int,
        *,
        candles: list[Candle] | None = None,
        spread: float | None = None,
        quote: tuple[float, float, float] | None = None,
        atr_value: float | None = None,
        extra: dict[str, Any] | None = None,
        mode_override: str | None = None,
    ) -> OrderflowSnapshot:
        candles_view = closed_candles(candles or [])
        snapshot = self._snapshot_lite(
            candles=candles_view,
            spread=spread,
            quote=quote,
            atr_value=atr_value,
            window=window,
        )
        if self._enricher is not None:
            snapshot = self._enricher.enrich(snapshot, symbol)
        return snapshot

    def _snapshot_lite(
        self,
        *,
        candles: list[Candle],
        spread: float | None,
        quote: tuple[float, float, float] | None,
        atr_value: float | None,
        window: int,
    ) -> OrderflowSnapshot:
        if not candles:
            metrics = OrderflowMetrics()
            direction, pressure = infer_orderflow_direction(metrics)
            return OrderflowSnapshot(
                confidence=0.0, mode="LITE", metrics=metrics, pressure=pressure, direction=direction
            )

        view = candles[-max(3, int(window)) :]
        mids = [((c.high + c.low) / 2.0) for c in view]
        diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        up = sum(1 for d in diffs if d > 0)
        down = sum(1 for d in diffs if d < 0)
        ticks = up + down
        delta_ratio = ((up - down) / ticks) if ticks > 0 else 0.0

        total_path = sum(abs(d) for d in diffs)
        efficiency_ratio = (abs(mids[-1] - mids[0]) / total_path) if total_path > 0 else 0.0
        efficiency_ratio = _clamp(efficiency_ratio, 0.0, 1.0)
        chop_score = _clamp(1.0 - efficiency_ratio, 0.0, 1.0)

        recent = max(4, len(view) // 3)
        recent_range = max(c.high for c in view[-recent:]) - min(c.low for c in view[-recent:])
        base_slice = view[:-recent] if len(view[:-recent]) >= 3 else view
        base_range = max(c.high for c in base_slice) - min(c.low for c in base_slice)
        range_expansion = (recent_range / base_range) if base_range > 0 else 1.0
        range_expansion = _clamp(range_expansion, 0.0, 2.5)

        aggression = _clamp((abs(delta_ratio) * 0.65) + (max(0.0, range_expansion - 1.0) * 0.35), 0.0, 1.0)

        last = view[-1]
        candle_range = max(1e-9, last.high - last.low)
        microprice_bias = ((last.close - ((last.high + last.low) / 2.0)) / candle_range) * 2.0
        microprice_bias = _clamp(microprice_bias, -1.0, 1.0)
        if quote is not None:
            bid, ask, _ = quote
            mid = (bid + ask) / 2.0
            quote_bias = ((last.close - mid) / max(candle_range, 1e-9)) * 2.0
            microprice_bias = _clamp((0.7 * microprice_bias) + (0.3 * quote_bias), -1.0, 1.0)

        wick_ratios: list[float] = []
        for item in view[-min(12, len(view)) :]:
            rng = max(1e-9, item.high - item.low)
            body = abs(item.close - item.open)
            wick = max(0.0, rng - body)
            wick_ratios.append(wick / rng)
        absorption_score = _clamp(sum(wick_ratios) / len(wick_ratios), 0.0, 1.0) if wick_ratios else 0.0

        if atr_value is None:
            atr_value = latest_value(atr(view, period=14))
        spread_ratio = 0.0
        if spread is not None and atr_value is not None and atr_value > 0:
            spread_ratio = max(0.0, spread / atr_value)

        spread_quality = 1.0
        if spread_ratio > 0:
            spread_quality = _clamp(1.0 - (spread_ratio / 0.4), 0.0, 1.0)
        data_quality = _clamp(len(view) / max(10.0, float(window)), 0.0, 1.0)
        confidence = _clamp((0.35 * data_quality) + (0.35 * (1.0 - chop_score)) + (0.3 * spread_quality), 0.0, 1.0)

        metrics = OrderflowMetrics(
            delta_ratio=round(delta_ratio, 6),
            aggression=round(aggression, 6),
            obi_k=0.0,
            microprice_bias=round(microprice_bias, 6),
            absorption_score=round(absorption_score, 6),
            chop_score=round(chop_score, 6),
            spread_ratio=round(spread_ratio, 6),
            efficiency_ratio=round(efficiency_ratio, 6),
        )
        direction, pressure = infer_orderflow_direction(metrics)
        return OrderflowSnapshot(
            confidence=round(confidence, 6),
            mode="LITE",
            metrics=metrics,
            pressure=round(pressure, 6),
            direction=direction,
        )
