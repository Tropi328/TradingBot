from __future__ import annotations

from bot.config import AppConfig, StrategyRouterConfig, StrategySymbolMappingConfig
from bot.strategy.router import StrategyRouter


def test_strategy_router_maps_symbol_to_strategy() -> None:
    config = AppConfig(
        strategy_router=StrategyRouterConfig(
            symbols=[
                StrategySymbolMappingConfig(symbol="US500", strategy="INDEX_EXISTING", priority=99),
                StrategySymbolMappingConfig(symbol="XAUUSD", strategy="SCALP_ICT_PA", priority=70),
            ]
        )
    )
    router = StrategyRouter(config)
    us500 = router.route_for("us500")
    xau = router.route_for("XAUUSD")
    unknown = router.route_for("EURUSD")

    assert us500.strategy == "INDEX_EXISTING"
    assert us500.priority == 99
    assert xau.strategy == "SCALP_ICT_PA"
    assert unknown.strategy == "SCALP_ICT_PA"


def test_strategy_router_returns_ordered_routes_for_symbol() -> None:
    config = AppConfig(
        strategy_router=StrategyRouterConfig(
            symbols=[
                StrategySymbolMappingConfig(symbol="BTCUSD", strategy="TREND_PULLBACK_M15", priority=70),
                StrategySymbolMappingConfig(symbol="BTCUSD", strategy="SCALP_ICT_PA", priority=90),
                StrategySymbolMappingConfig(symbol="BTCUSD", strategy="ORB_H4_RETEST", priority=60),
            ]
        )
    )
    router = StrategyRouter(config)
    routes = router.routes_for("btcusd")
    assert [route.strategy for route in routes] == ["SCALP_ICT_PA", "TREND_PULLBACK_M15", "ORB_H4_RETEST"]
