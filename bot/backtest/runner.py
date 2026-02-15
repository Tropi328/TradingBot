from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from bot.backtest.data_provider import AutoDataLoader
from bot.backtest.engine import BacktestReport, run_backtest_multi_strategy
from bot.config import AppConfig, AssetConfig
from bot.data.candles import Candle


@dataclass(slots=True)
class BacktestBatchReport:
    timeframe: str
    price_mode: str
    start: datetime
    end: datetime
    symbols: list[str]
    reports: dict[str, BacktestReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "price_mode": self.price_mode,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "symbols": list(self.symbols),
            "reports": {symbol: report.to_dict() for symbol, report in self.reports.items()},
        }


class BacktestRunner:
    def __init__(self, *, config: AppConfig, data_loader: AutoDataLoader):
        self.config = config
        self.data_loader = data_loader

    def run(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        price_mode: str,
        slippage_points: float = 0.0,
        slippage_atr_multiplier: float = 0.0,
    ) -> BacktestBatchReport:
        reports: dict[str, BacktestReport] = {}
        for symbol in symbols:
            normalized = symbol.strip().upper()
            if not normalized:
                continue
            loaded = self.data_loader.load_symbol_data(
                symbol=normalized,
                timeframe=timeframe,
                start=start,
                end=end,
                price_mode=price_mode,
            )
            candles = self._frame_to_candles(loaded.frame)
            asset = self._resolve_asset(normalized)
            assumed_spread, spread_mode = self._assumed_spread(
                frame=loaded.frame,
                symbol=normalized,
                asset=asset,
                diagnostics=loaded.diagnostics,
            )
            report = run_backtest_multi_strategy(
                config=self.config,
                asset=asset,
                candles_m5=candles,
                assumed_spread=assumed_spread,
                slippage_points=slippage_points,
                slippage_atr_multiplier=slippage_atr_multiplier,
                data_context={
                    **loaded.diagnostics,
                    "symbol": normalized,
                    "timeframe": timeframe,
                    "price_mode_requested": price_mode,
                    "spread_mode": spread_mode,
                    "assumed_spread_used": float(assumed_spread),
                },
            )
            reports[normalized] = report

        return BacktestBatchReport(
            timeframe=timeframe,
            price_mode=price_mode,
            start=start,
            end=end,
            symbols=[item.strip().upper() for item in symbols if item.strip()],
            reports=reports,
        )

    def _resolve_asset(self, symbol: str) -> AssetConfig:
        for asset in self.config.assets:
            if asset.epic.upper() == symbol:
                return asset
        template = self.config.assets[0] if self.config.assets else AssetConfig(**self.config.instrument.model_dump(), trade_enabled=True)
        return AssetConfig(
            epic=symbol,
            currency=template.currency,
            instrument_currency=template.instrument_currency,
            point_size=template.point_size,
            minimal_tick_buffer=template.minimal_tick_buffer,
            min_size=template.min_size,
            size_step=template.size_step,
            trade_enabled=True,
        )

    def _assumed_spread(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        asset: AssetConfig,
        diagnostics: dict[str, object],
    ) -> tuple[float, str]:
        spread_from_data = None
        if "spread" in frame.columns:
            spread = pd.to_numeric(frame["spread"], errors="coerce").dropna()
            if not spread.empty:
                spread_from_data = float(max(0.0, spread.median()))

        symbol_spread_map = self.config.backtest_tuning.assumed_spread_by_symbol
        configured = symbol_spread_map.get(symbol.upper()) or symbol_spread_map.get(asset.epic.upper())
        assumed = float(spread_from_data) if spread_from_data is not None else float(configured if configured is not None else 0.2)

        data_health = diagnostics.get("data_health", {}) if isinstance(diagnostics, dict) else {}
        nan_counts = data_health.get("nan_counts", {}) if isinstance(data_health, dict) else {}
        bars_count = int(data_health.get("bars", 0)) if isinstance(data_health, dict) and data_health.get("bars") is not None else 0
        close_bid_nan = int(nan_counts.get("close_bid", bars_count)) if isinstance(nan_counts, dict) else bars_count
        close_ask_nan = int(nan_counts.get("close_ask", bars_count)) if isinstance(nan_counts, dict) else bars_count
        spread_mode = "ASSUMED_OHLC" if bars_count > 0 and close_bid_nan >= bars_count and close_ask_nan >= bars_count else "REAL_BIDASK"
        return assumed, spread_mode

    @staticmethod
    def _frame_to_candles(frame: pd.DataFrame) -> list[Candle]:
        if frame.empty:
            return []
        rows = frame.sort_values("ts_utc").itertuples(index=False)
        candles: list[Candle] = []
        for row in rows:
            bid_close = getattr(row, "close_bid", None)
            ask_close = getattr(row, "close_ask", None)
            candle = Candle(
                timestamp=row.ts_utc.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                bid=float(bid_close) if bid_close is not None and pd.notna(bid_close) else None,
                ask=float(ask_close) if ask_close is not None and pd.notna(ask_close) else None,
                volume=float(row.volume) if row.volume is not None and pd.notna(row.volume) else 0.0,
            )
            candles.append(candle)
        return candles
