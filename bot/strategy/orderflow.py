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
    ) -> OrderflowSnapshot:
        ...


class CompositeOrderflowProvider(OrderflowProvider):
    def __init__(
        self,
        *,
        default_mode: str = "LITE",
        symbol_modes: dict[str, str] | None = None,
    ) -> None:
        self.default_mode = default_mode.strip().upper()
        self.symbol_modes = {k.strip().upper(): v.strip().upper() for k, v in (symbol_modes or {}).items()}

    def _resolve_mode(self, symbol: str, mode_override: str | None) -> str:
        if mode_override:
            mode = mode_override.strip().upper()
            if mode in {"LITE", "FULL"}:
                return mode
        mode = self.symbol_modes.get(symbol.strip().upper(), self.default_mode).upper()
        return mode if mode in {"LITE", "FULL"} else "LITE"

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
        mode = self._resolve_mode(symbol, mode_override)
        candles_view = closed_candles(candles or [])
        if mode == "FULL":
            return self._snapshot_full(
                candles=candles_view,
                spread=spread,
                quote=quote,
                atr_value=atr_value,
                window=window,
                extra=extra or {},
            )
        return self._snapshot_lite(
            candles=candles_view,
            spread=spread,
            quote=quote,
            atr_value=atr_value,
            window=window,
        )

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
            return OrderflowSnapshot(confidence=0.0, mode="LITE", metrics=metrics, pressure=pressure, direction=direction)

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

    def _snapshot_full(
        self,
        *,
        candles: list[Candle],
        spread: float | None,
        quote: tuple[float, float, float] | None,
        atr_value: float | None,
        window: int,
        extra: dict[str, Any],
    ) -> OrderflowSnapshot:
        lite = self._snapshot_lite(
            candles=candles,
            spread=spread,
            quote=quote,
            atr_value=atr_value,
            window=window,
        )
        payload = extra.get("orderflow_full")
        if not isinstance(payload, dict):
            # FULL requested but no depth/trade payload available: keep deterministic fallback.
            return OrderflowSnapshot(
                confidence=round(_clamp(lite.confidence * 0.6, 0.0, 1.0), 6),
                mode="FULL",
                metrics=lite.metrics,
                pressure=lite.pressure,
                direction=lite.direction,
            )

        trades_raw = payload.get("trades")
        trades = trades_raw if isinstance(trades_raw, list) else []
        buy_volume = 0.0
        sell_volume = 0.0
        for item in trades:
            if not isinstance(item, dict):
                continue
            side = str(item.get("side", "")).strip().lower()
            try:
                size = float(item.get("size", 0.0))
            except (TypeError, ValueError):
                size = 0.0
            if size <= 0:
                continue
            if side in {"buy", "bid", "b"}:
                buy_volume += size
            elif side in {"sell", "ask", "s"}:
                sell_volume += size
        total_volume = buy_volume + sell_volume
        delta_ratio = ((buy_volume - sell_volume) / total_volume) if total_volume > 0 else lite.metrics.delta_ratio

        book_raw = payload.get("book")
        book = book_raw if isinstance(book_raw, dict) else {}
        try:
            bid_size = float(book.get("bid_size", 0.0))
            ask_size = float(book.get("ask_size", 0.0))
        except (TypeError, ValueError):
            bid_size = 0.0
            ask_size = 0.0
        size_total = bid_size + ask_size
        obi_k = ((bid_size - ask_size) / size_total) if size_total > 0 else 0.0
        obi_k = _clamp(obi_k, -1.0, 1.0)

        try:
            bid = float(book.get("bid", 0.0))
            ask = float(book.get("ask", 0.0))
        except (TypeError, ValueError):
            bid = 0.0
            ask = 0.0
        microprice_bias = lite.metrics.microprice_bias
        if bid > 0 and ask > 0 and size_total > 0:
            microprice = ((ask * bid_size) + (bid * ask_size)) / size_total
            mid = (bid + ask) / 2.0
            half_spread = max((ask - bid) / 2.0, 1e-9)
            microprice_bias = _clamp((microprice - mid) / half_spread, -1.0, 1.0)

        trades_quality = _clamp(len(trades) / max(5.0, float(window) / 2.0), 0.0, 1.0)
        aggression = _clamp((abs(delta_ratio) * 0.6) + (trades_quality * 0.4), 0.0, 1.0)

        absorption_override = payload.get("absorption_score")
        if absorption_override is not None:
            try:
                absorption_score = _clamp(float(absorption_override), 0.0, 1.0)
            except (TypeError, ValueError):
                absorption_score = lite.metrics.absorption_score
        else:
            absorption_score = _clamp((lite.metrics.absorption_score * 0.6) + ((1.0 - abs(delta_ratio)) * 0.4), 0.0, 1.0)

        confidence = _clamp(
            (0.2 + (0.35 if trades else 0.0) + (0.25 if size_total > 0 else 0.0) + (0.2 * (1.0 - lite.metrics.chop_score))),
            0.0,
            1.0,
        )

        metrics = OrderflowMetrics(
            delta_ratio=round(delta_ratio, 6),
            aggression=round(aggression, 6),
            obi_k=round(obi_k, 6),
            microprice_bias=round(microprice_bias, 6),
            absorption_score=round(absorption_score, 6),
            chop_score=lite.metrics.chop_score,
            spread_ratio=lite.metrics.spread_ratio,
            efficiency_ratio=lite.metrics.efficiency_ratio,
        )
        direction, pressure = infer_orderflow_direction(metrics)
        return OrderflowSnapshot(
            confidence=round(confidence, 6),
            mode="FULL",
            metrics=metrics,
            pressure=round(pressure, 6),
            direction=direction,
        )
