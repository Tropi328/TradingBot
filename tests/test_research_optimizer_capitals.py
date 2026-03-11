from __future__ import annotations

import pytest

import bot.app.config_helpers as config_helpers_module
from bot.config import AppConfig, ResearchCapitalConfig


def test_parse_research_capitals_parses_amount_and_currency() -> None:
    parsed = config_helpers_module.parse_research_capitals("10000:USD,100:PLN,10000:USD")
    assert parsed == [
        {"equity": 10000.0, "currency": "USD"},
        {"equity": 100.0, "currency": "PLN"},
    ]


def test_parse_research_capitals_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected amount:CCY"):
        config_helpers_module.parse_research_capitals("10000USD")


def test_resolve_optimizer_capitals_uses_fallback_when_config_empty() -> None:
    config = AppConfig()
    config.risk.equity = 777.0
    config.account_currency = "USD"
    config.research.optimize.capitals = []
    resolved = config_helpers_module._resolve_optimizer_capitals(config)
    assert resolved == [{"equity": 777.0, "currency": "USD"}]


def test_resolve_optimizer_capitals_uses_configured_values() -> None:
    config = AppConfig()
    config.research.optimize.capitals = [
        ResearchCapitalConfig(equity=10000, currency="usd"),
        ResearchCapitalConfig(equity=100, currency="pln"),
    ]
    resolved = config_helpers_module._resolve_optimizer_capitals(config)
    assert resolved == [
        {"equity": 10000.0, "currency": "USD"},
        {"equity": 100.0, "currency": "PLN"},
    ]


def test_validate_optimizer_capitals_fails_for_pln_without_usdpln_rate() -> None:
    with pytest.raises(RuntimeError, match="USDPLN"):
        config_helpers_module._validate_optimizer_capitals(
            capitals=[{"equity": 100.0, "currency": "PLN"}],
            fx_static_rates={"EURUSD": 1.1},
        )


def test_optimizer_capital_path_and_file_tags() -> None:
    assert config_helpers_module._capital_dir_name(10000.0, "USD") == "capital_10000_USD"
    assert config_helpers_module._capital_dir_name(100.0, "PLN") == "capital_100_PLN"
    assert config_helpers_module._capital_file_tag(10000.0, "USD") == "10K_USD"
    assert config_helpers_module._capital_file_tag(100.0, "PLN") == "100PLN"
