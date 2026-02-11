from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.config import AppConfig, StrategySymbolMappingConfig


@dataclass(slots=True)
class StrategyRoute:
    symbol: str
    strategy: str
    params: dict[str, Any]
    risk: dict[str, Any]
    priority: int
    cooldown_seconds: int


class StrategyRouter:
    def __init__(self, config: AppConfig):
        self._routes: dict[str, list[StrategyRoute]] = {}
        for item in config.strategy_router.symbols:
            self._routes.setdefault(item.symbol, []).append(self._to_route(item))
        for symbol in list(self._routes.keys()):
            self._routes[symbol].sort(key=lambda route: route.priority, reverse=True)

    @staticmethod
    def _to_route(item: StrategySymbolMappingConfig) -> StrategyRoute:
        return StrategyRoute(
            symbol=item.symbol,
            strategy=item.strategy,
            params=dict(item.params),
            risk=dict(item.risk),
            priority=item.priority,
            cooldown_seconds=item.cooldown_seconds,
        )

    def route_for(self, symbol: str) -> StrategyRoute:
        routes = self.routes_for(symbol)
        return routes[0]

    def routes_for(self, symbol: str) -> list[StrategyRoute]:
        key = symbol.strip().upper()
        if key in self._routes:
            return list(self._routes[key])
        # Symbol not configured explicitly: default to SCALP.
        return [
            StrategyRoute(
                symbol=key,
                strategy="SCALP_ICT_PA",
                params={},
                risk={},
                priority=50,
                cooldown_seconds=300,
            )
        ]
