"""Factory functions for building runtime objects — extracted from app_main.py."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from bot.app.config_helpers import parse_epics_csv
from bot.config import AppConfig, AssetConfig
from bot.data.capital_client import CapitalClient
from bot.execution.feasibility import estimate_required_margin
from bot.gating.daily_gate import DailyGateProvider
from bot.monitoring.alerts import AlertConfig, AlertDispatcher
from bot.news.calendar_provider import CalendarProvider, Event, build_calendar_provider

LOGGER = logging.getLogger("trading_bot")


def _build_daily_gate_provider(
    *,
    config: AppConfig,
    mode: str,
    candles: list | None = None,
    events: list[Event] | None = None,
    overrides: dict[str, float | int | None] | None = None,
) -> DailyGateProvider | None:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "off":
        return None
    params = {
        "thr": float(config.daily_gate.thr),
        "pre_minutes": int(config.daily_gate.pre_minutes),
        "post_minutes": int(config.daily_gate.post_minutes),
        "vol_max": float(config.daily_gate.vol_max),
        "max_spread": (float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None),
    }
    if overrides:
        params.update(overrides)
    provider = DailyGateProvider(
        mode=normalized_mode,
        ema_fast=int(config.daily_gate.ema_fast),
        ema_slow=int(config.daily_gate.ema_slow),
        thr=float(params["thr"]),
        atr_period=int(config.daily_gate.atr_period),
        vol_max=float(params["vol_max"]),
        max_spread=(float(params["max_spread"]) if params["max_spread"] is not None else None),
        pre_minutes=int(params["pre_minutes"]),
        post_minutes=int(params["post_minutes"]),
        rollover_start_utc=config.daily_gate.rollover_start_utc,
        rollover_end_utc=config.daily_gate.rollover_end_utc,
        allowed_strategies=config.daily_gate.allowed_strategies,
        events=events or [],
    )
    if candles:
        provider.refresh_from_candles(candles)
    return provider


def _gate_param_space_for_grid(config: AppConfig) -> list[dict[str, float | int | None]]:
    base_spread = float(config.daily_gate.max_spread) if config.daily_gate.max_spread is not None else None
    spread_candidates: list[float | None]
    if base_spread is None:
        spread_candidates = [None]
    else:
        spread_candidates = [base_spread, base_spread * 1.5]
    grid: list[dict[str, float | int | None]] = []
    for thr in [0.0005, 0.0010, 0.0020]:
        for pre_minutes in [15, 30, 45]:
            for post_minutes in [15, 30, 45]:
                for vol_max in [0.015, 0.020, 0.025]:
                    for max_spread in spread_candidates:
                        grid.append(
                            {
                                "thr": thr,
                                "pre_minutes": pre_minutes,
                                "post_minutes": post_minutes,
                                "vol_max": vol_max,
                                "max_spread": max_spread,
                            }
                        )
    return grid


def _asset_from_template(epic: str, template: AssetConfig, trade_enabled: bool) -> AssetConfig:
    return AssetConfig(
        epic=epic,
        currency=template.currency,
        instrument_currency=template.instrument_currency,
        point_size=template.point_size,
        minimal_tick_buffer=template.minimal_tick_buffer,
        min_size=template.min_size,
        size_step=template.size_step,
        trade_enabled=trade_enabled,
    )


def _estimated_open_margin(*, positions: list, config: AppConfig) -> float:
    total = 0.0
    margin_pct = float(config.backtest_tuning.broker_margin_requirement_pct)
    leverage = float(config.backtest_tuning.broker_leverage)
    for position in positions:
        try:
            entry = float(getattr(position, "entry_price", 0.0))
            size = float(getattr(position, "size", 0.0))
        except (TypeError, ValueError):
            continue
        total += estimate_required_margin(
            entry_price=entry,
            size=size,
            margin_requirement_pct=margin_pct,
            max_leverage=leverage,
        )
    return total


def build_asset_universe(config: AppConfig) -> list[AssetConfig]:
    assets = [asset.model_copy(deep=True) for asset in config.assets]
    if not assets:
        assets = [AssetConfig(**config.instrument.model_dump(), trade_enabled=True)]

    template = assets[0]
    by_epic = {a.epic.upper(): a for a in assets}

    primary = (os.getenv("CAPITAL_EPIC") or "").strip().upper()
    trade_epics = parse_epics_csv(os.getenv("CAPITAL_TRADE_EPICS"))
    watch_epics = parse_epics_csv(os.getenv("CAPITAL_WATCH_EPICS"))

    if primary:
        if primary not in by_epic:
            by_epic[primary] = _asset_from_template(primary, template, True)
        by_epic[primary].trade_enabled = True

    if trade_epics:
        for item in by_epic.values():
            item.trade_enabled = item.epic in trade_epics
        for epic in trade_epics:
            if epic not in by_epic:
                by_epic[epic] = _asset_from_template(epic, template, True)

    for epic in watch_epics:
        if epic not in by_epic:
            by_epic[epic] = _asset_from_template(epic, template, False)

    trading = sorted((a for a in by_epic.values() if a.trade_enabled), key=lambda a: a.epic)
    observing = sorted((a for a in by_epic.values() if not a.trade_enabled), key=lambda a: a.epic)
    return trading + observing


def build_client(config: AppConfig, paper_mode: bool) -> CapitalClient | None:
    base_url = os.getenv("CAPITAL_BASE_URL", config.capital.demo_base_url)
    api_key = os.getenv("CAPITAL_API_KEY")
    identifier = os.getenv("CAPITAL_IDENTIFIER")
    password = os.getenv("CAPITAL_API_PASSWORD") or os.getenv("CAPITAL_PASSWORD")
    account_id = os.getenv("CAPITAL_ACCOUNT_ID")

    if paper_mode and not (api_key and identifier and password):
        raise RuntimeError("Paper mode requires API credentials in .env")
    if not (api_key and identifier and password):
        LOGGER.warning("Credentials missing. Running without live market data.")
        return None
    return CapitalClient(
        base_url=base_url,
        api_key=api_key,
        identifier=identifier,
        password=password,
        account_id=account_id,
        rate_limit_rps=float(os.getenv("CAPITAL_RATE_LIMIT_RPS", str(config.capital.rate_limit_rps))),
        rate_limit_burst=int(os.getenv("CAPITAL_RATE_LIMIT_BURST", str(config.capital.rate_limit_burst))),
        request_max_attempts=int(os.getenv("CAPITAL_REQUEST_MAX_ATTEMPTS", str(config.capital.request_max_attempts))),
        backoff_base_seconds=float(os.getenv("CAPITAL_BACKOFF_BASE_SECONDS", str(config.capital.backoff_base_seconds))),
        backoff_max_seconds=float(os.getenv("CAPITAL_BACKOFF_MAX_SECONDS", str(config.capital.backoff_max_seconds))),
        reconnect_short_retries=int(
            os.getenv("CAPITAL_RECONNECT_SHORT_RETRIES", str(config.capital.reconnect_short_retries))
        ),
        session_refresh_min_interval_seconds=int(
            os.getenv(
                "CAPITAL_SESSION_REFRESH_MIN_INTERVAL_SECONDS",
                str(config.capital.session_refresh_min_interval_seconds),
            )
        ),
    )


def build_news_provider(config: AppConfig, root: Path) -> CalendarProvider:
    provider_name = os.getenv("NEWS_PROVIDER", config.calendar.provider)
    dummy_file = Path(config.calendar.dummy_file)
    if not dummy_file.is_absolute():
        dummy_file = root / dummy_file
    return build_calendar_provider(
        provider_name=provider_name,
        dummy_file=dummy_file,
        http_url=os.getenv("NEWS_HTTP_URL"),
        http_token=os.getenv("NEWS_HTTP_TOKEN"),
        timeout_seconds=config.calendar.http_timeout_seconds,
        cache_ttl_seconds=config.calendar.http_cache_ttl_seconds,
    )


def build_alert_dispatcher(config: AppConfig) -> AlertDispatcher:
    return AlertDispatcher(
        AlertConfig(
            enabled=config.monitoring.alerts_enabled,
            discord_webhook=os.getenv("ALERT_DISCORD_WEBHOOK"),
            telegram_bot_token=os.getenv("ALERT_TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("ALERT_TELEGRAM_CHAT_ID"),
            cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "30")),
        )
    )
