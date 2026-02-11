from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone

from bot.config import AppConfig
from bot.data.candles import Candle, candles_from_prices, resolution_to_capital
from bot.data.capital_client import CapitalAPIError, CapitalClient
from bot.data.spreads import compute_spread
from bot.storage.journal import Journal

LOGGER = logging.getLogger(__name__)


class MarketDataService:
    def __init__(
        self,
        client: CapitalClient | None,
        config: AppConfig,
        journal: Journal,
    ):
        self.client = client
        self.config = config
        self.journal = journal
        self._spread_cache_by_epic: dict[str, deque[float]] = {}
        self._last_quote_ts_by_epic: dict[str, datetime] = {}

    def _spread_cache(self, epic: str) -> deque[float]:
        key = epic.strip().upper()
        if key not in self._spread_cache_by_epic:
            self._spread_cache_by_epic[key] = deque(
                self.journal.load_recent_spreads(self.config.spread_filter.window, key),
                maxlen=self.config.spread_filter.window,
            )
        return self._spread_cache_by_epic[key]

    def fetch_candles(self, epic: str, timeframe: str, max_points: int) -> list[Candle]:
        if self.client is None:
            return []
        resolution = resolution_to_capital(timeframe)
        prices = self.client.get_prices(
            epic=epic,
            resolution=resolution,
            max_points=max_points,
        )
        return candles_from_prices(prices)

    def fetch_quote_and_spread(self, epic: str, *, persist_spread: bool = True) -> tuple[float | None, float | None, float | None]:
        if self.client is None:
            return None, None, None
        try:
            bid, ask = self.client.get_quote(epic)
            spread = compute_spread(bid, ask)
            key = epic.strip().upper()
            self._spread_cache(key).append(spread)
            now = datetime.now(timezone.utc)
            self._last_quote_ts_by_epic[key] = now
            if persist_spread:
                self.journal.save_spread(now, spread, key)
            return bid, ask, spread
        except CapitalAPIError as exc:
            LOGGER.warning("Could not fetch quote/spread for %s: %s", epic, exc)
            return None, None, None

    def fetch_quotes(self, epics: list[str]) -> dict[str, tuple[float, float, float]]:
        output: dict[str, tuple[float, float, float]] = {}
        for epic in epics:
            bid, ask, spread = self.fetch_quote_and_spread(epic, persist_spread=False)
            if bid is None or ask is None or spread is None:
                continue
            output[epic] = (bid, ask, spread)
        return output

    def spread_history(self, epic: str) -> list[float]:
        return list(self._spread_cache(epic))

    def quote_age_seconds(self, epic: str, now: datetime | None = None) -> float | None:
        key = epic.strip().upper()
        ts = self._last_quote_ts_by_epic.get(key)
        if ts is None:
            return None
        current = now or datetime.now(timezone.utc)
        return max(0.0, (current - ts).total_seconds())
