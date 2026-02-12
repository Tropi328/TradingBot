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
            assumed_spread = self._assumed_spread(loaded.frame)
            report = run_backtest_multi_strategy(
                config=self.config,
                asset=asset,
                candles_m5=candles,
                assumed_spread=assumed_spread,
                slippage_points=slippage_points,
                slippage_atr_multiplier=slippage_atr_multiplier,
                data_context=loaded.diagnostics,
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
            point_size=template.point_size,
            minimal_tick_buffer=template.minimal_tick_buffer,
            min_size=template.min_size,
            size_step=template.size_step,
            trade_enabled=True,
        )

    @staticmethod
    def _assumed_spread(frame: pd.DataFrame) -> float:
        if "spread" not in frame.columns:
            return 0.0
        spread = pd.to_numeric(frame["spread"], errors="coerce").dropna()
        if spread.empty:
            return 0.0
        return float(max(0.0, spread.median()))

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
