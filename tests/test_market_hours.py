from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.clock import is_symbol_market_open
from bot.config import AppConfig, MarketHoursConfig


def test_market_hours_defaults_are_weekdays() -> None:
    cfg = MarketHoursConfig()
    assert cfg.default_profile == "WEEKDAYS"
    assert cfg.profile_for("BTCUSD") == "WEEKDAYS"


def test_market_hours_normalizes_symbol_keys() -> None:
    cfg = MarketHoursConfig(
        default_profile="weekdays",
        symbol_profiles={"btcusd": "always"},
    )
    assert cfg.default_profile == "WEEKDAYS"
    assert cfg.symbol_profiles == {"BTCUSD": "ALWAYS"}


@pytest.mark.parametrize("default_profile", ["weekend", "24_7"])
def test_market_hours_rejects_invalid_default(default_profile: str) -> None:
    with pytest.raises(ValueError, match="market_hours.default_profile"):
        MarketHoursConfig(default_profile=default_profile)


def test_market_hours_empty_default_falls_back_to_weekdays() -> None:
    cfg = MarketHoursConfig(default_profile="")
    assert cfg.default_profile == "WEEKDAYS"


def test_market_hours_rejects_invalid_symbol_profile() -> None:
    with pytest.raises(ValueError, match=r"market_hours.symbol_profiles\[BTCUSD\]"):
        MarketHoursConfig(symbol_profiles={"BTCUSD": "INVALID"})


def test_btc_can_trade_on_weekend_while_fx_is_closed() -> None:
    saturday = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc)
    app_cfg = AppConfig(
        market_hours={
            "default_profile": "WEEKDAYS",
            "symbol_profiles": {"BTCUSD": "ALWAYS"},
        }
    )

    assert is_symbol_market_open(
        saturday,
        symbol="BTCUSD",
        timezone_name=app_cfg.timezone,
        default_profile=app_cfg.market_hours.default_profile,
        symbol_profiles=app_cfg.market_hours.symbol_profiles,
    )
    assert not is_symbol_market_open(
        saturday,
        symbol="XAUUSD",
        timezone_name=app_cfg.timezone,
        default_profile=app_cfg.market_hours.default_profile,
        symbol_profiles=app_cfg.market_hours.symbol_profiles,
    )


def test_btc_weekend_fx_closed_regression() -> None:
    test_btc_can_trade_on_weekend_while_fx_is_closed()
