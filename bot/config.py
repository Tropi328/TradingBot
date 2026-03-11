from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class InstrumentConfig(BaseModel):
    epic: str = "XAUUSD"
    currency: str = "USD"
    instrument_currency: str | None = None
    point_size: float = 0.01
    minimal_tick_buffer: float = 0.05
    min_size: float = 0.01
    size_step: float = 0.01

    @model_validator(mode="after")
    def normalize_currency_fields(self) -> InstrumentConfig:
        self.currency = str(self.currency).strip().upper() or "USD"
        instrument_ccy = str(self.instrument_currency or "").strip().upper()
        self.instrument_currency = instrument_ccy or self.currency
        if self.point_size <= 0:
            raise ValueError("instrument.point_size must be > 0")
        if self.min_size <= 0:
            raise ValueError("instrument.min_size must be > 0")
        if self.size_step <= 0:
            raise ValueError("instrument.size_step must be > 0")
        return self


class AssetConfig(InstrumentConfig):
    trade_enabled: bool = True


class TimeframesConfig(BaseModel):
    h1: str = "H1"
    m15: str = "M15"
    m5: str = "M5"


class IndicatorsConfig(BaseModel):
    ema_period_h1: int = 200
    atr_period: int = 14

    @model_validator(mode="after")
    def validate_periods(self) -> IndicatorsConfig:
        if self.ema_period_h1 < 1:
            raise ValueError("indicators.ema_period_h1 must be >= 1")
        if self.atr_period < 1:
            raise ValueError("indicators.atr_period must be >= 1")
        return self


class SwingConfig(BaseModel):
    fractal_left: int = 2
    fractal_right: int = 2


class SweepConfig(BaseModel):
    lookback_min_hours: int = 2
    lookback_max_hours: int = 8
    threshold_atr_multiplier: float = 0.15

    @model_validator(mode="after")
    def validate_lookback(self) -> SweepConfig:
        if self.lookback_min_hours > self.lookback_max_hours:
            raise ValueError("lookback_min_hours must be <= lookback_max_hours")
        return self


class DisplacementConfig(BaseModel):
    base_multiplier: float = 1.3
    a_plus_multiplier: float = 1.6


class HistoryBarsConfig(BaseModel):
    h1: int = 600
    m15: int = 500
    m5: int = 500


class ExecutionConfig(BaseModel):
    limit_ttl_bars: int = 6
    loop_seconds: int = 45
    heartbeat_seconds: int = 300
    max_data_stale_seconds: int = 900
    quote_refresh_seconds_trade: int = 30
    quote_refresh_seconds_observe: int = 60
    candle_close_grace_seconds: int = 3
    candle_retry_seconds: int = 15
    sync_pending_seconds: int = 30
    sync_positions_seconds: int = 30
    history_bars: HistoryBarsConfig = Field(default_factory=HistoryBarsConfig)


class SpreadFilterConfig(BaseModel):
    window: int = 100
    max_multiple_of_median: float = 2.0


class SpreadGuardGateConfig(BaseModel):
    enabled: bool = False
    spread_cost_max: float = 0.0
    percentile_limit: float = 0.0
    window: int = 100


class NewsGateConfig(BaseModel):
    block_minutes: int = 60


class DailyGateConfig(BaseModel):
    mode: str | bool = "off"
    ema_fast: int = 20
    ema_slow: int = 50
    thr: float = 0.001
    atr_period: int = 14
    vol_max: float = 0.02
    max_spread: float | None = None
    pre_minutes: int = 30
    post_minutes: int = 30
    rollover_start_utc: str | None = None
    rollover_end_utc: str | None = None
    allowed_strategies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self) -> DailyGateConfig:
        if isinstance(self.mode, bool):
            mode_normalized = "off" if self.mode is False else "trend"
        else:
            mode_normalized = str(self.mode).strip().lower()
        if mode_normalized in {"false", "none", ""}:
            mode_normalized = "off"
        if mode_normalized not in {"off", "trend", "trend_vol_news"}:
            raise ValueError("daily_gate.mode must be one of: off, trend, trend_vol_news")
        self.mode = mode_normalized
        if self.ema_fast <= 0:
            raise ValueError("daily_gate.ema_fast must be > 0")
        if self.ema_slow <= 0:
            raise ValueError("daily_gate.ema_slow must be > 0")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("daily_gate.ema_fast must be < daily_gate.ema_slow")
        if self.thr < 0:
            raise ValueError("daily_gate.thr must be >= 0")
        if self.atr_period <= 0:
            raise ValueError("daily_gate.atr_period must be > 0")
        if self.vol_max <= 0:
            raise ValueError("daily_gate.vol_max must be > 0")
        if self.max_spread is not None and self.max_spread <= 0:
            raise ValueError("daily_gate.max_spread must be > 0 when provided")
        if self.pre_minutes < 0:
            raise ValueError("daily_gate.pre_minutes must be >= 0")
        if self.post_minutes < 0:
            raise ValueError("daily_gate.post_minutes must be >= 0")
        self.allowed_strategies = [str(item).strip().upper() for item in self.allowed_strategies if str(item).strip()]
        return self


class CorrelationGroupConfig(BaseModel):
    name: str
    epics: list[str]
    max_open_positions: int = 1

    @model_validator(mode="after")
    def normalize(self) -> CorrelationGroupConfig:
        self.epics = [str(epic).strip().upper() for epic in self.epics if str(epic).strip()]
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        return self


class RiskConfig(BaseModel):
    equity: float = 10000
    risk_per_trade: float = 0.005
    risk_mode: str = "percent"
    sizing_mode: str = "risk_pct_equity"
    fixed_qty: float = 0.0
    fixed_notional: float = 0.0
    min_risk_cash_per_trade: float = 0.0
    max_risk_cash_per_trade: float | None = None
    allow_min_size_override_if_within_risk: bool = False
    max_trades_per_day: int = 3
    daily_stop_pct: float = 0.015
    max_positions: int = 1
    global_max_positions: int = 3
    max_total_risk_pct: float = 0.015
    cooldown_loss_streak: int = 2
    cooldown_minutes: int = 60
    low_equity_mode_enabled: bool = True
    low_equity_threshold: float = 250.0
    low_equity_threshold_pct: float = 0.0
    low_equity_risk_multiplier: float = 0.35
    low_equity_risk_per_trade_cap: float = 0.02
    low_equity_max_trades_per_day: int = 2
    low_equity_daily_stop_pct: float = 0.01
    low_equity_min_size_fallback_enabled: bool = True
    low_equity_min_size_fallback_max_risk_pct: float = 0.02
    min_risk_cash_auto: bool = False
    min_risk_cash_auto_pct: float = 0.003
    correlation_groups: list[CorrelationGroupConfig] = Field(
        default_factory=lambda: [
            CorrelationGroupConfig(
                name="US_INDICES",
                epics=["US100", "US500"],
                max_open_positions=1,
            )
        ]
    )

    @model_validator(mode="after")
    def validate_risk(self) -> RiskConfig:
        if self.equity <= 0:
            raise ValueError("risk.equity must be > 0")
        if self.risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be > 0")
        if self.risk_per_trade > 1.0:
            raise ValueError("risk_per_trade cannot exceed 1.0 (100.0%)")
        mode = str(self.risk_mode or "percent").strip().lower()
        if mode not in {"percent", "cash"}:
            raise ValueError("risk_mode must be one of: percent, cash")
        self.risk_mode = mode
        sizing = str(self.sizing_mode or "risk_pct_equity").strip().lower()
        if sizing not in {"risk_pct_equity", "fixed_qty", "fixed_notional"}:
            raise ValueError("sizing_mode must be one of: risk_pct_equity, fixed_qty, fixed_notional")
        self.sizing_mode = sizing
        if sizing == "fixed_qty" and self.fixed_qty <= 0:
            raise ValueError("risk.fixed_qty must be > 0 when sizing_mode is fixed_qty")
        if sizing == "fixed_notional" and self.fixed_notional <= 0:
            raise ValueError("risk.fixed_notional must be > 0 when sizing_mode is fixed_notional")
        if float(self.min_risk_cash_per_trade) < 0:
            raise ValueError("min_risk_cash_per_trade must be >= 0")
        if self.max_risk_cash_per_trade is not None and float(self.max_risk_cash_per_trade) <= 0:
            raise ValueError("max_risk_cash_per_trade must be > 0 when provided")
        if self.max_risk_cash_per_trade is not None and float(self.min_risk_cash_per_trade) > float(
            self.max_risk_cash_per_trade
        ):
            raise ValueError("min_risk_cash_per_trade cannot exceed max_risk_cash_per_trade")
        if self.max_trades_per_day < 1:
            raise ValueError("risk.max_trades_per_day must be >= 1")
        if self.daily_stop_pct <= 0:
            raise ValueError("risk.daily_stop_pct must be > 0")
        if self.max_positions < 1:
            raise ValueError("risk.max_positions must be >= 1")
        if self.max_total_risk_pct <= 0:
            raise ValueError("max_total_risk_pct must be > 0")
        if self.global_max_positions <= 0:
            raise ValueError("global_max_positions must be > 0")
        if self.cooldown_loss_streak < 0:
            raise ValueError("risk.cooldown_loss_streak must be >= 0")
        if self.cooldown_minutes < 0:
            raise ValueError("risk.cooldown_minutes must be >= 0")
        if self.low_equity_threshold <= 0:
            raise ValueError("low_equity_threshold must be > 0")
        if self.low_equity_threshold_pct < 0 or self.low_equity_threshold_pct > 1.0:
            raise ValueError("low_equity_threshold_pct must be in [0,1]")
        if not (0 < self.low_equity_risk_multiplier <= 1.0):
            raise ValueError("low_equity_risk_multiplier must be in (0,1]")
        if not (0 < self.low_equity_risk_per_trade_cap <= 1.0):
            raise ValueError("low_equity_risk_per_trade_cap must be in (0,1]")
        if self.low_equity_max_trades_per_day <= 0:
            raise ValueError("low_equity_max_trades_per_day must be > 0")
        if not (0 < self.low_equity_daily_stop_pct <= 1.0):
            raise ValueError("low_equity_daily_stop_pct must be in (0,1]")
        if not (0 <= self.low_equity_min_size_fallback_max_risk_pct <= 1.0):
            raise ValueError("low_equity_min_size_fallback_max_risk_pct must be in [0,1]")
        if self.min_risk_cash_auto_pct < 0 or self.min_risk_cash_auto_pct > 1.0:
            raise ValueError("min_risk_cash_auto_pct must be in [0,1]")
        # Auto-compute min_risk_cash from equity if enabled
        if self.min_risk_cash_auto and self.equity > 0:
            self.min_risk_cash_per_trade = max(
                float(self.min_risk_cash_per_trade),
                float(self.equity) * float(self.min_risk_cash_auto_pct),
            )
        return self


class CapitalConfig(BaseModel):
    demo_base_url: str = "https://demo-api-capital.backend-capital.com/api/v1"
    rate_limit_rps: float = 2.0
    rate_limit_burst: int = 5
    request_max_attempts: int = 6
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 20.0
    reconnect_short_retries: int = 2
    session_refresh_min_interval_seconds: int = 5

    @model_validator(mode="after")
    def validate_capital(self) -> CapitalConfig:
        if self.rate_limit_rps <= 0:
            raise ValueError("capital.rate_limit_rps must be > 0")
        if self.rate_limit_burst < 1:
            raise ValueError("capital.rate_limit_burst must be >= 1")
        if self.request_max_attempts < 1:
            raise ValueError("capital.request_max_attempts must be >= 1")
        if self.backoff_base_seconds <= 0:
            raise ValueError("capital.backoff_base_seconds must be > 0")
        if self.backoff_max_seconds <= 0:
            raise ValueError("capital.backoff_max_seconds must be > 0")
        if self.reconnect_short_retries < 0:
            raise ValueError("capital.reconnect_short_retries must be >= 0")
        if self.session_refresh_min_interval_seconds < 0:
            raise ValueError("capital.session_refresh_min_interval_seconds must be >= 0")
        return self


class StrategyRuntimeConfig(BaseModel):
    allow_neutral_bias: bool = False
    neutral_bias_risk_multiplier: float = 0.5

    @model_validator(mode="after")
    def validate_values(self) -> StrategyRuntimeConfig:
        if not (0 < self.neutral_bias_risk_multiplier <= 1.0):
            raise ValueError("neutral_bias_risk_multiplier must be in (0, 1]")
        return self


class CalendarConfig(BaseModel):
    provider: str = "dummy"
    dummy_file: str = "news_data/events.json"
    http_timeout_seconds: int = 10
    http_cache_ttl_seconds: int = 300


class OpsConfig(BaseModel):
    heartbeat_stale_seconds: int = 900
    watchdog_interval_seconds: int = 60
    alert_cooldown_seconds: int = 300
    required_services: list[str] = Field(default_factory=lambda: ["bot-paper-fx.service", "bot-paper-btc.service"])
    backup_retention_days: int = 14
    backup_verify_on_create: bool = True

    @model_validator(mode="after")
    def validate_values(self) -> OpsConfig:
        if self.heartbeat_stale_seconds <= 0:
            raise ValueError("ops.heartbeat_stale_seconds must be > 0")
        if self.watchdog_interval_seconds <= 0:
            raise ValueError("ops.watchdog_interval_seconds must be > 0")
        if self.alert_cooldown_seconds < 0:
            raise ValueError("ops.alert_cooldown_seconds must be >= 0")
        if self.backup_retention_days < 0:
            raise ValueError("ops.backup_retention_days must be >= 0")

        normalized_services: list[str] = []
        seen: set[str] = set()
        for raw in self.required_services:
            name = str(raw).strip()
            if not name or name in seen:
                continue
            normalized_services.append(name)
            seen.add(name)
        self.required_services = normalized_services
        return self


class CapitalRampConfig(BaseModel):
    enabled: bool = False


class MarketHoursConfig(BaseModel):
    default_profile: str = "WEEKDAYS"
    symbol_profiles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize(self) -> MarketHoursConfig:
        allowed_profiles = {"WEEKDAYS", "ALWAYS"}
        default_profile = str(self.default_profile or "WEEKDAYS").strip().upper() or "WEEKDAYS"
        if default_profile not in allowed_profiles:
            raise ValueError("market_hours.default_profile must be WEEKDAYS or ALWAYS")
        self.default_profile = default_profile

        normalized_profiles: dict[str, str] = {}
        for symbol, profile in self.symbol_profiles.items():
            key = str(symbol).strip().upper()
            if not key:
                continue
            value = str(profile or "").strip().upper() or default_profile
            if value not in allowed_profiles:
                raise ValueError(f"market_hours.symbol_profiles[{key}] must be WEEKDAYS or ALWAYS")
            normalized_profiles[key] = value
        self.symbol_profiles = normalized_profiles
        return self

    def profile_for(self, symbol: str) -> str:
        key = str(symbol).strip().upper()
        return self.symbol_profiles.get(key, self.default_profile)


class WatchlistConfig(BaseModel):
    epics: list[str] = Field(default_factory=list)
    log_quotes: bool = True

    @model_validator(mode="after")
    def normalize_epics(self) -> WatchlistConfig:
        normalized: list[str] = []
        seen: set[str] = set()
        for epic in self.epics:
            item = str(epic).strip().upper()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        self.epics = normalized
        return self


class MonitoringConfig(BaseModel):
    dashboard_path: str = "runtime_dashboard.json"
    dashboard_interval_seconds: int = 30
    alerts_enabled: bool = True
    log_decision_reasons: bool = True


class StrategySymbolMappingConfig(BaseModel):
    symbol: str
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    priority: int = 50
    cooldown_seconds: int = 300

    @model_validator(mode="after")
    def normalize(self) -> StrategySymbolMappingConfig:
        self.symbol = self.symbol.strip().upper()
        self.strategy = self.strategy.strip().upper()
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        return self


class StrategyRouterConfig(BaseModel):
    symbols: list[StrategySymbolMappingConfig] = Field(
        default_factory=lambda: [
            StrategySymbolMappingConfig(symbol="GOLD", strategy="SCALP_ICT_PA", priority=80),
            StrategySymbolMappingConfig(symbol="BTCUSD", strategy="SCALP_ICT_PA", priority=75),
            StrategySymbolMappingConfig(symbol="EURUSD", strategy="SCALP_ICT_PA", priority=70),
            StrategySymbolMappingConfig(symbol="US100", strategy="INDEX_EXISTING", priority=95),
            StrategySymbolMappingConfig(symbol="US500", strategy="INDEX_EXISTING", priority=90),
        ]
    )


class ScalpStrategyConfig(BaseModel):
    bias_timeframe: str = "M15"
    trigger_timeframe: str = "M5"
    candidate_ttl_minutes: int = 60
    trade_score_threshold: float = 65.0
    small_score_min: float = 60.0
    small_score_max: float = 64.99
    small_risk_multiplier: float = 0.45
    miss_window_minutes: int = 60
    miss_move_atr: float = 1.0
    be_after_r: float = 1.0
    time_stop_minutes: int = 120

    @model_validator(mode="after")
    def validate_thresholds(self) -> ScalpStrategyConfig:
        if self.candidate_ttl_minutes <= 0:
            raise ValueError("candidate_ttl_minutes must be > 0")
        if not (0 <= self.small_score_min <= self.small_score_max <= 100):
            raise ValueError("small_score_min/small_score_max must be in [0,100] and min<=max")
        if not (0 < self.trade_score_threshold <= 100):
            raise ValueError("trade_score_threshold must be in (0,100]")
        if not (0 < self.small_risk_multiplier <= 1.0):
            raise ValueError("small_risk_multiplier must be in (0,1]")
        if self.miss_window_minutes <= 0:
            raise ValueError("miss_window_minutes must be > 0")
        if self.miss_move_atr <= 0:
            raise ValueError("miss_move_atr must be > 0")
        return self


class PortfolioSupervisorConfig(BaseModel):
    max_open_positions_total: int = 2
    max_per_symbol: int = 1
    daily_risk_r: float = 2.0
    default_cooldown_seconds: int = 300
    max_entries_per_cycle: int = 1

    @model_validator(mode="after")
    def validate_values(self) -> PortfolioSupervisorConfig:
        if self.max_open_positions_total <= 0:
            raise ValueError("max_open_positions_total must be > 0")
        if self.max_per_symbol <= 0:
            raise ValueError("max_per_symbol must be > 0")
        if self.daily_risk_r <= 0:
            raise ValueError("daily_risk_r must be > 0")
        if self.default_cooldown_seconds < 0:
            raise ValueError("default_cooldown_seconds must be >= 0")
        if self.max_entries_per_cycle <= 0:
            raise ValueError("max_entries_per_cycle must be > 0")
        return self


# ---------------------------------------------------------------------------
# Risk-budget (master safety layer)
# ---------------------------------------------------------------------------
class RiskBudgetConfig(BaseModel):
    enabled: bool = False
    risk_per_trade_pct: float = 0.0035
    max_open_risk_pct: float = 0.02
    daily_loss_limit_pct: float = 0.025
    daily_profit_lock_pct: float = 0.0

    @model_validator(mode="after")
    def validate_values(self) -> RiskBudgetConfig:
        if not (0 < self.risk_per_trade_pct <= 0.05):
            raise ValueError("risk_per_trade_pct must be in (0, 0.05]")
        if not (0 < self.max_open_risk_pct <= 0.10):
            raise ValueError("max_open_risk_pct must be in (0, 0.10]")
        if not (0 < self.daily_loss_limit_pct <= 0.10):
            raise ValueError("daily_loss_limit_pct must be in (0, 0.10]")
        if self.daily_profit_lock_pct < 0:
            raise ValueError("daily_profit_lock_pct must be >= 0")
        return self


# ---------------------------------------------------------------------------
# Score tiers
# ---------------------------------------------------------------------------
class ScoreTierEntry(BaseModel):
    min_score: float = 62.0
    size_mult: float = 1.0
    allow_market_after_ttl: bool = False


class ScoreTiersConfig(BaseModel):
    enabled: bool = False
    A_plus: ScoreTierEntry = Field(
        default_factory=lambda: ScoreTierEntry(min_score=72.0, size_mult=1.0, allow_market_after_ttl=True)
    )
    A: ScoreTierEntry = Field(default_factory=lambda: ScoreTierEntry(min_score=68.0, size_mult=0.8))
    B: ScoreTierEntry = Field(default_factory=lambda: ScoreTierEntry(min_score=62.0, size_mult=0.6))


# ---------------------------------------------------------------------------
# FVG entry ladder
# ---------------------------------------------------------------------------
class LadderLevelConfig(BaseModel):
    name: str = "mid"
    fvg_fraction: float = 0.50
    size_fraction: float = 0.70


class MarketFallbackConfig(BaseModel):
    enabled: bool = True
    only_tiers: list[str] = Field(default_factory=lambda: ["A_plus"])
    size_mult: float = 0.5


class FvgEntryLadderConfig(BaseModel):
    enabled: bool = False
    levels: list[LadderLevelConfig] = Field(
        default_factory=lambda: [
            LadderLevelConfig(name="mid", fvg_fraction=0.50, size_fraction=0.70),
            LadderLevelConfig(name="shallow", fvg_fraction=0.35, size_fraction=0.30),
        ]
    )
    ttl_bars_5m: int = 8
    market_fallback: MarketFallbackConfig = Field(default_factory=MarketFallbackConfig)


# ---------------------------------------------------------------------------
# Exits (partial TP)
# ---------------------------------------------------------------------------
class PartialTpConfig(BaseModel):
    enabled: bool = True
    at_r: float = 1.0
    close_fraction: float = 0.5
    move_sl_to_be: bool = True
    be_buffer_r: float = 0.1


class ExitsConfig(BaseModel):
    base_rr: float = 2.0
    partial_tp: PartialTpConfig = Field(default_factory=PartialTpConfig)


# ---------------------------------------------------------------------------
# Second-position correlation rule
# ---------------------------------------------------------------------------
class SecondPositionRuleConfig(BaseModel):
    first_position_sl_at_be: bool = True
    or_profit_r_greater_equal: float = 0.7


class CorrelationV2Config(BaseModel):
    max_positions_per_group: int = 1
    allow_second_same_symbol_only_if: SecondPositionRuleConfig = Field(default_factory=SecondPositionRuleConfig)


class DecisionPolicyConfig(BaseModel):
    trade_score_threshold: float = 65.0
    small_score_min: float = 60.0
    small_score_max: float = 64.99
    trade_risk_multiplier: float = 1.0
    small_risk_multiplier_default: float = 0.45

    @model_validator(mode="after")
    def validate_values(self) -> DecisionPolicyConfig:
        if not (0 < self.trade_score_threshold <= 100):
            raise ValueError("trade_score_threshold must be in (0,100]")
        if not (0 <= self.small_score_min <= self.small_score_max <= 100):
            raise ValueError("small_score_min/small_score_max must be in [0,100] and min<=max")
        if not (0 < self.trade_risk_multiplier <= 1.0):
            raise ValueError("trade_risk_multiplier must be in (0,1]")
        if not (0 < self.small_risk_multiplier_default <= 1.0):
            raise ValueError("small_risk_multiplier_default must be in (0,1]")
        return self


class ScoreV3ConfigModel(BaseModel):
    """Configuration for ScoreV3 scoring system."""

    enabled: bool = True
    mode: str = "heuristic"  # "heuristic" or "ml"
    model_path: str = ""
    trade_threshold: float = 55.0
    small_min: float = 45.0
    small_max: float = 54.99
    tier_enabled: bool = True
    tier_a_plus_pct: float = 0.10
    tier_a_pct: float = 0.25
    tier_b_pct: float = 0.30
    shadow_enabled: bool = True
    shadow_output_path: str = "reports/shadow_observe.jsonl"
    shadow_simulate_outcomes: bool = True
    fill_prob_weight: float = 0.3


# ---------------------------------------------------------------------------
# Paper cost model
# ---------------------------------------------------------------------------
class SlippageModelPydanticConfig(BaseModel):
    base_ticks: float = 0.02
    beta_spread: float = 0.15
    beta_atr: float = 0.005


class PaperCostPydanticConfig(BaseModel):
    enabled: bool = True
    slippage_market: SlippageModelPydanticConfig = Field(
        default_factory=lambda: SlippageModelPydanticConfig(base_ticks=0.03, beta_spread=0.20, beta_atr=0.008)
    )
    slippage_stop: SlippageModelPydanticConfig = Field(
        default_factory=lambda: SlippageModelPydanticConfig(base_ticks=0.05, beta_spread=0.25, beta_atr=0.010)
    )
    slippage_limit: SlippageModelPydanticConfig = Field(
        default_factory=lambda: SlippageModelPydanticConfig(base_ticks=0.01, beta_spread=0.10, beta_atr=0.003)
    )
    commission_per_side: float = 0.0
    swap_per_day: float = 0.0
    use_bid_ask_fills: bool = True


# ---------------------------------------------------------------------------
# Micro-loss defense
# ---------------------------------------------------------------------------
class MicroLossDefensePydanticConfig(BaseModel):
    enabled: bool = True
    micro_loss_k: float = 1.5
    min_stop_spread_mult: float = 5.0
    min_stop_atr_mult: float = 0.15
    edge_mult: float = 3.0
    be_buffer_atr_frac: float = 0.05
    be_buffer_ticks: float = 0.05


# ---------------------------------------------------------------------------
# Adaptive threshold (PAPER/LIVE throughput)
# ---------------------------------------------------------------------------
class AdaptiveThresholdPydanticConfig(BaseModel):
    enabled: bool = False
    base_threshold: float = 62.0
    range_adjust: float = -6.0  # lower threshold in range regime
    trend_adjust: float = -2.0  # slightly lower in strong trend (re-entry)
    high_vol_adjust: float = 2.0  # raise threshold in extreme vol
    low_vol_adjust: float = -3.0  # lower in calm markets
    # Soft-gate conversion
    soft_gates_enabled: bool = True
    soft_gate_penalty: float = 4.0  # penalty instead of hard block


# ---------------------------------------------------------------------------
# Multi-TP exit profile (PAPER/LIVE)
# ---------------------------------------------------------------------------
class TPLevelConfig(BaseModel):
    """Single take-profit level."""

    name: str = "TP1"
    trigger_r: float = 1.0  # R-multiple to trigger this level
    close_fraction: float = 0.25  # fraction of remaining size to close
    move_sl_to_be: bool = False  # whether to move SL to BE after this level


class MultiTPConfig(BaseModel):
    """Multi-level take-profit exit profile for PAPER/LIVE."""

    enabled: bool = False
    levels: list[TPLevelConfig] = Field(
        default_factory=lambda: [
            TPLevelConfig(name="TP1", trigger_r=1.0, close_fraction=0.20, move_sl_to_be=True),
            TPLevelConfig(name="TP2", trigger_r=2.0, close_fraction=0.50, move_sl_to_be=False),
            TPLevelConfig(name="TP3", trigger_r=3.0, close_fraction=1.0, move_sl_to_be=False),
        ]
    )
    be_offset_r: float = 0.05  # BE offset as fraction of initial risk
    be_delay_bars: int = 4  # bars to wait after TP1 before moving to BE
    trailing_enabled: bool = True  # enable trailing stop after TP1
    trailing_swing_window: int = 12  # lookback bars for swing detection
    trailing_buffer_r: float = 0.12  # buffer below swing low (as R fraction)


# ---------------------------------------------------------------------------
# Tier-based sizing (PAPER/LIVE)
# ---------------------------------------------------------------------------
class TierSizingConfig(BaseModel):
    """Score-tier-based position sizing multipliers."""

    enabled: bool = False
    a_plus_mult: float = 1.5  # A+ setups get 1.5× risk
    a_mult: float = 1.0  # A setups get normal risk
    b_mult: float = 0.6  # B setups get 0.6× risk
    observe_mult: float = 0.0  # OBSERVE not traded
    # Tier score boundaries (used when score_tiers.enabled is False)
    a_plus_min_score: float = 80.0
    a_min_score: float = 65.0
    b_min_score: float = 55.0


# ---------------------------------------------------------------------------
# Session-aware entry windows
# ---------------------------------------------------------------------------
class SessionEntryWindowConfig(BaseModel):
    """Single trading session window."""

    name: str = "LONDON"
    utc_start: str = "07:00"  # UTC start
    utc_end: str = "16:00"  # UTC end
    threshold_adjust: float = 0.0  # added to base threshold
    risk_mult: float = 1.0  # risk multiplier for this session
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])


class SessionFilterConfig(BaseModel):
    """Session-based trading filter."""

    enabled: bool = False
    sessions: list[SessionEntryWindowConfig] = Field(
        default_factory=lambda: [
            SessionEntryWindowConfig(
                name="LONDON",
                utc_start="07:00",
                utc_end="11:00",
                threshold_adjust=-3.0,
                risk_mult=1.0,
            ),
            SessionEntryWindowConfig(
                name="NY_OVERLAP",
                utc_start="12:00",
                utc_end="16:00",
                threshold_adjust=-2.0,
                risk_mult=1.0,
            ),
            SessionEntryWindowConfig(
                name="NY_AFTERNOON",
                utc_start="16:00",
                utc_end="20:00",
                threshold_adjust=0.0,
                risk_mult=0.8,
            ),
            SessionEntryWindowConfig(
                name="ASIA",
                utc_start="00:00",
                utc_end="07:00",
                threshold_adjust=4.0,
                risk_mult=0.5,
            ),
        ]
    )
    block_outside_sessions: bool = False  # hard-block trades outside all sessions


# ---------------------------------------------------------------------------
# Compound equity sizing
# ---------------------------------------------------------------------------
class CompoundEquityConfig(BaseModel):
    """Compound (geometric) equity-based sizing."""

    enabled: bool = False
    floor_equity: float = 50.0  # never size below this equity
    cap_equity: float = 0.0  # 0 = no cap (unlimited compound)
    smooth_window: int = 0  # 0 = use raw equity; >0 = EMA smoothing


class BacktestTuningConfig(BaseModel):
    wait_reaction_timeout_bars: int = 8
    wait_mitigation_timeout_bars: int = 12
    wait_hard_block_bars: int = 2
    reaction_timeout_force_enable: bool = True
    wait_timeout_soft_penalty: float = 4.0
    wait_timeout_small_risk_multiplier: float = 0.4
    wait_timeout_soft_grace_bars: int = 2
    max_loss_r_cap: float = 1.0
    tp1_trigger_r: float = 0.7
    tp1_fraction: float = 0.35
    be_offset_r: float = 0.05
    be_delay_bars_after_tp1: int = 2
    tp_target_min_r: float = 2.0
    tp_target_max_r: float = 2.0
    tp_target_a_plus_r: float = 3.0
    tp_gradient_enabled: bool = False
    tp_gradient_tier_high_score: float = 75.0
    tp_gradient_tier_high_r: float = 3.0
    tp_gradient_tier_mid_score: float = 62.0
    tp_gradient_tier_mid_r: float = 2.5
    tp_gradient_tier_low_r: float = 2.0
    tp_profile_mode: str = "strict_tp_price"
    trailing_after_tp1: bool = True
    trailing_swing_window_bars: int = 8
    trailing_buffer_r: float = 0.05
    expected_rr_min: float = 1.15
    expected_rr_lookback_bars: int = 120
    segment_soft_gap_minutes: int = 120
    segment_hard_gap_minutes: int = 600
    penalty_orb_no_retest: float = -4.0
    penalty_orb_confirm_low: float = -3.0
    penalty_scalp_no_displacement: float = -4.0
    penalty_scalp_no_mss: float = -3.0
    penalty_scalp_no_fvg: float = -2.0
    ohlc_only_spread_soft_penalty: float = 3.0
    thresholds_v2_trade: float = 62.0
    thresholds_v2_small_min: float = 58.0
    thresholds_v2_small_max: float = 61.0
    dynamic_spread_ratio_frac: float = 0.9
    dynamic_atr_buffer_mult: float = 1.1
    dynamic_spread_score_penalty: float = 2.0
    dynamic_atr_score_penalty: float = 1.0
    dynamic_assumed_spread_enabled: bool = False
    dynamic_assumed_spread_min_by_symbol: dict[str, float] = Field(default_factory=dict)
    dynamic_assumed_spread_max_by_symbol: dict[str, float] = Field(default_factory=dict)
    assumed_spread_by_symbol: dict[str, float] = Field(default_factory=lambda: {"XAUUSD": 0.2})
    broker_leverage: float = 20.0
    broker_margin_requirement_pct: float = 5.0
    overnight_swap_time_utc: str = "23:00"
    overnight_swap_long_pct: float = -0.016
    overnight_swap_short_pct: float = 0.0076
    spread_limit_points: float | None = None
    no_overnight: bool = False
    rollover_block_minutes_before: int = 30
    rollover_block_minutes_after: int = 30
    force_close_before_rollover_minutes: int = 15
    min_edge_to_cost_ratio: float = 4.0
    fx_conversion_pct: float = 0.0
    max_drawdown_halt_pct: float = 0.0  # 0 = disabled; e.g. 25.0 halts backtest when DD > 25% of equity
    wf_roll_equity_forward: bool = False  # if True, each WF split starts with the previous split's ending equity

    @model_validator(mode="after")
    def validate_values(self) -> BacktestTuningConfig:
        if self.wait_reaction_timeout_bars <= 0:
            raise ValueError("wait_reaction_timeout_bars must be > 0")
        if self.wait_mitigation_timeout_bars <= 0:
            raise ValueError("wait_mitigation_timeout_bars must be > 0")
        if self.wait_hard_block_bars < 0:
            raise ValueError("wait_hard_block_bars must be >= 0")
        if self.wait_hard_block_bars >= self.wait_mitigation_timeout_bars:
            raise ValueError("wait_hard_block_bars must be < wait_mitigation_timeout_bars")
        if self.wait_timeout_soft_penalty < 0:
            raise ValueError("wait_timeout_soft_penalty must be >= 0")
        if not (0 < self.wait_timeout_small_risk_multiplier <= 1.0):
            raise ValueError("wait_timeout_small_risk_multiplier must be in (0,1]")
        if self.wait_timeout_soft_grace_bars < 0:
            raise ValueError("wait_timeout_soft_grace_bars must be >= 0")
        if self.max_loss_r_cap <= 0:
            raise ValueError("max_loss_r_cap must be > 0")
        if self.tp1_trigger_r <= 0:
            raise ValueError("tp1_trigger_r must be > 0")
        if not (0 < self.tp1_fraction < 1):
            raise ValueError("tp1_fraction must be in (0,1)")
        if self.be_offset_r < 0:
            raise ValueError("be_offset_r must be >= 0")
        if self.be_delay_bars_after_tp1 < 0:
            raise ValueError("be_delay_bars_after_tp1 must be >= 0")
        if self.tp_target_min_r <= 0 or self.tp_target_max_r <= 0:
            raise ValueError("tp_target_min_r and tp_target_max_r must be > 0")
        if self.tp_target_min_r > self.tp_target_max_r:
            raise ValueError("tp_target_min_r must be <= tp_target_max_r")
        if self.tp_target_a_plus_r <= 0:
            raise ValueError("tp_target_a_plus_r must be > 0")
        mode = self.tp_profile_mode.strip().lower()
        if mode not in {"strict_tp_price", "strict_total_rr"}:
            raise ValueError("tp_profile_mode must be strict_tp_price or strict_total_rr")
        self.tp_profile_mode = mode
        if self.trailing_swing_window_bars < 2:
            raise ValueError("trailing_swing_window_bars must be >= 2")
        if self.trailing_buffer_r < 0:
            raise ValueError("trailing_buffer_r must be >= 0")
        if self.expected_rr_min <= 0:
            raise ValueError("expected_rr_min must be > 0")
        if self.expected_rr_lookback_bars <= 0:
            raise ValueError("expected_rr_lookback_bars must be > 0")
        if self.segment_soft_gap_minutes <= 0:
            raise ValueError("segment_soft_gap_minutes must be > 0")
        if self.segment_hard_gap_minutes < self.segment_soft_gap_minutes:
            raise ValueError("segment_hard_gap_minutes must be >= segment_soft_gap_minutes")
        if self.ohlc_only_spread_soft_penalty < 0:
            raise ValueError("ohlc_only_spread_soft_penalty must be >= 0")
        if not (0 <= self.thresholds_v2_small_min <= self.thresholds_v2_small_max <= 100):
            raise ValueError("thresholds_v2_small_min/thresholds_v2_small_max must be in [0,100] and min<=max")
        if not (0 < self.thresholds_v2_trade <= 100):
            raise ValueError("thresholds_v2_trade must be in (0,100]")
        if not (0.0 <= self.dynamic_spread_ratio_frac <= 1.0):
            raise ValueError("dynamic_spread_ratio_frac must be in [0,1]")
        if self.dynamic_atr_buffer_mult <= 0:
            raise ValueError("dynamic_atr_buffer_mult must be > 0")
        if self.broker_leverage <= 0:
            raise ValueError("broker_leverage must be > 0")
        if not (0 < self.broker_margin_requirement_pct <= 100):
            raise ValueError("broker_margin_requirement_pct must be in (0,100]")
        if self.spread_limit_points is not None and float(self.spread_limit_points) <= 0:
            raise ValueError("spread_limit_points must be > 0 when provided")
        if self.rollover_block_minutes_before < 0:
            raise ValueError("rollover_block_minutes_before must be >= 0")
        if self.rollover_block_minutes_after < 0:
            raise ValueError("rollover_block_minutes_after must be >= 0")
        if self.force_close_before_rollover_minutes < 0:
            raise ValueError("force_close_before_rollover_minutes must be >= 0")
        if self.min_edge_to_cost_ratio <= 0:
            raise ValueError("min_edge_to_cost_ratio must be > 0")
        if not (0 <= self.fx_conversion_pct <= 100):
            raise ValueError("fx_conversion_pct must be in [0,100]")
        time_raw = str(self.overnight_swap_time_utc).strip()
        parts = time_raw.split(":")
        if len(parts) != 2:
            raise ValueError("overnight_swap_time_utc must use HH:MM format")
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except ValueError as exc:
            raise ValueError("overnight_swap_time_utc must use HH:MM format") from exc
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError("overnight_swap_time_utc must be a valid UTC time")
        self.overnight_swap_time_utc = f"{hh:02d}:{mm:02d}"
        normalized: dict[str, float] = {}
        for key, value in self.assumed_spread_by_symbol.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            spread = float(value)
            if spread < 0:
                raise ValueError("assumed_spread_by_symbol values must be >= 0")
            normalized[symbol] = spread
        self.assumed_spread_by_symbol = normalized
        min_map: dict[str, float] = {}
        for key, value in self.dynamic_assumed_spread_min_by_symbol.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            spread = float(value)
            if spread < 0:
                raise ValueError("dynamic_assumed_spread_min_by_symbol values must be >= 0")
            min_map[symbol] = spread
        max_map: dict[str, float] = {}
        for key, value in self.dynamic_assumed_spread_max_by_symbol.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            spread = float(value)
            if spread < 0:
                raise ValueError("dynamic_assumed_spread_max_by_symbol values must be >= 0")
            max_map[symbol] = spread
        for symbol, min_val in min_map.items():
            max_val = max_map.get(symbol)
            if max_val is not None and max_val < min_val:
                raise ValueError(f"dynamic_assumed_spread_max_by_symbol[{symbol}] must be >= min")
        self.dynamic_assumed_spread_min_by_symbol = min_map
        self.dynamic_assumed_spread_max_by_symbol = max_map
        return self


class OrderflowConfig(BaseModel):
    # FULL mode removed — only LITE is supported. Fields kept for backward compat.
    default_mode: str = "LITE"
    full_symbols: list[str] = Field(default_factory=list)  # deprecated, ignored
    default_window: int = 64
    trigger_bonus_max: float = 10.0
    execution_bonus_max: float = 5.0
    divergence_penalty_min: float = 6.0
    divergence_penalty_max: float = 10.0
    small_soft_gate_confidence: float = 0.75
    small_soft_gate_chop: float = 0.75

    @model_validator(mode="after")
    def validate_values(self) -> OrderflowConfig:
        # Always normalise to LITE — FULL mode is no longer supported.
        self.default_mode = "LITE"
        self.full_symbols = []  # ignored, kept for config file compat
        if self.default_window <= 0:
            raise ValueError("orderflow.default_window must be > 0")
        if self.trigger_bonus_max < 0:
            raise ValueError("orderflow.trigger_bonus_max must be >= 0")
        if self.execution_bonus_max < 0:
            raise ValueError("orderflow.execution_bonus_max must be >= 0")
        if self.divergence_penalty_min < 0 or self.divergence_penalty_max < 0:
            raise ValueError("orderflow divergence penalties must be >= 0")
        if self.divergence_penalty_min > self.divergence_penalty_max:
            raise ValueError("orderflow.divergence_penalty_min must be <= divergence_penalty_max")
        if not (0 <= self.small_soft_gate_confidence <= 1):
            raise ValueError("orderflow.small_soft_gate_confidence must be in [0,1]")
        if not (0 <= self.small_soft_gate_chop <= 1):
            raise ValueError("orderflow.small_soft_gate_chop must be in [0,1]")
        return self


# ---------------------------------------------------------------------------
# Monte Carlo live window
# ---------------------------------------------------------------------------
class MonteCarloLiveWindowConfig(BaseModel):
    enabled: bool = True
    refresh_seconds: float = 1.0
    window_title: str = "IGNACY BOT — Monte Carlo Live"
    show_stats_overlay: bool = True
    max_fps: int = 2
    open_on_start: bool = False
    viewer_mode: str = "terminal"  # "terminal" | "process" | "manual"
    png_path: str = "reports/live/monte_carlo.png"
    json_path: str = "reports/live/monte_carlo.json"

    @model_validator(mode="after")
    def validate_viewer_mode(self) -> MonteCarloLiveWindowConfig:
        mode = self.viewer_mode.strip().lower()
        if mode not in {"process", "manual", "terminal"}:
            raise ValueError("monte_carlo.live_window.viewer_mode must be 'terminal', 'process' or 'manual'")
        self.viewer_mode = mode
        return self


class MCAdaptiveConfig(BaseModel):
    """Settings for the Monte-Carlo adaptive risk model.

    When enabled, the model re-runs MC simulation every *resim_interval*
    trades, computes a health score (0-1), and scales *risk_per_trade*
    accordingly.
    """

    enabled: bool = False
    min_trades: int = 15
    resim_interval: int = 5
    num_simulations: int = 500
    num_simulations_online: int = 250
    chart_interval: int = 10
    ruin_weight: float = 0.35
    dd_weight: float = 0.25
    pf_weight: float = 0.25
    wr_weight: float = 0.15
    full_risk_health: float = 0.70
    min_risk_health: float = 0.35
    floor_multiplier: float = 0.25
    health_ema_alpha: float = 0.25
    max_step_up: float = 0.05
    max_step_down: float = 0.10

    @model_validator(mode="after")
    def _validate_adaptive(self) -> MCAdaptiveConfig:
        if self.min_trades < 5:
            raise ValueError("mc_adaptive.min_trades must be >= 5")
        if self.resim_interval < 1:
            raise ValueError("mc_adaptive.resim_interval must be >= 1")
        if self.num_simulations < 1:
            raise ValueError("mc_adaptive.num_simulations must be >= 1")
        if self.num_simulations_online < 1:
            raise ValueError("mc_adaptive.num_simulations_online must be >= 1")
        if self.chart_interval < 1:
            raise ValueError("mc_adaptive.chart_interval must be >= 1")
        total = self.ruin_weight + self.dd_weight + self.pf_weight + self.wr_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"mc_adaptive weights must sum to 1.0, got {total:.3f}")
        if not (0 < self.full_risk_health <= 1.0):
            raise ValueError("mc_adaptive.full_risk_health must be in (0, 1]")
        if not (0 <= self.min_risk_health < self.full_risk_health):
            raise ValueError("mc_adaptive.min_risk_health must be in [0, full_risk_health)")
        if not (0 < self.floor_multiplier <= 1.0):
            raise ValueError("mc_adaptive.floor_multiplier must be in (0, 1]")
        if not (0.0 < self.health_ema_alpha <= 1.0):
            raise ValueError("mc_adaptive.health_ema_alpha must be in (0, 1]")
        if not (0.0 < self.max_step_up <= 1.0):
            raise ValueError("mc_adaptive.max_step_up must be in (0, 1]")
        if not (0.0 < self.max_step_down <= 1.0):
            raise ValueError("mc_adaptive.max_step_down must be in (0, 1]")
        return self


class MonteCarloConfig(BaseModel):
    enabled: bool = True
    num_simulations: int = 1000
    ruin_dd_threshold: float = 0.50
    sampling_mode: str = "iid_bootstrap"
    block_size: int = 8
    equity_mode_backtest: str = "initial"
    equity_mode_adaptive: str = "current"
    ruin_equity_floor_pct: float | None = None
    ruin_equity_floor_abs: float | None = None
    count_breakeven_as_loss: bool = False
    seed: int | None = None
    max_paths_plotted: int = 200
    live_window: MonteCarloLiveWindowConfig = Field(default_factory=MonteCarloLiveWindowConfig)
    adaptive: MCAdaptiveConfig = Field(default_factory=MCAdaptiveConfig)

    @model_validator(mode="after")
    def _validate_mc_params(self) -> MonteCarloConfig:
        if self.num_simulations < 1:
            raise ValueError("monte_carlo.num_simulations must be >= 1")
        if self.num_simulations > 100_000:
            raise ValueError("monte_carlo.num_simulations must be <= 100 000")
        if self.ruin_dd_threshold <= 0 or self.ruin_dd_threshold > 1:
            raise ValueError("monte_carlo.ruin_dd_threshold must be in (0, 1]")
        self.sampling_mode = str(self.sampling_mode or "iid_bootstrap").strip().lower()
        if self.sampling_mode not in {"iid_bootstrap", "moving_block_bootstrap"}:
            raise ValueError("monte_carlo.sampling_mode must be iid_bootstrap or moving_block_bootstrap")
        if self.block_size < 2:
            raise ValueError("monte_carlo.block_size must be >= 2")
        self.equity_mode_backtest = str(self.equity_mode_backtest or "initial").strip().lower()
        if self.equity_mode_backtest not in {"initial", "current"}:
            raise ValueError("monte_carlo.equity_mode_backtest must be initial or current")
        self.equity_mode_adaptive = str(self.equity_mode_adaptive or "current").strip().lower()
        if self.equity_mode_adaptive not in {"initial", "current"}:
            raise ValueError("monte_carlo.equity_mode_adaptive must be initial or current")
        if self.ruin_equity_floor_pct is not None:
            value_pct = float(self.ruin_equity_floor_pct)
            if value_pct < 0 or value_pct > 1:
                raise ValueError("monte_carlo.ruin_equity_floor_pct must be in [0, 1]")
            self.ruin_equity_floor_pct = value_pct
        if self.ruin_equity_floor_abs is not None:
            value_abs = float(self.ruin_equity_floor_abs)
            if value_abs < 0:
                raise ValueError("monte_carlo.ruin_equity_floor_abs must be >= 0")
            self.ruin_equity_floor_abs = value_abs
        if self.max_paths_plotted < 0:
            raise ValueError("monte_carlo.max_paths_plotted must be >= 0")
        return self


# ---------------------------------------------------------------------------
# Diagnostics / decision trace
# ---------------------------------------------------------------------------
class DiagnosticsConfig(BaseModel):
    decision_trace_enabled: bool = False
    decision_trace_path: str = "logs/decision_trace.jsonl"
    decision_trace_auto_enable_backtest: bool = False
    dashboard_port: int = 8777


class ResearchCapitalConfig(BaseModel):
    equity: float
    currency: str = "USD"

    @model_validator(mode="after")
    def validate_capital(self) -> ResearchCapitalConfig:
        if self.equity <= 0:
            raise ValueError("research.optimize.capitals.equity must be > 0")
        self.currency = str(self.currency or "USD").strip().upper() or "USD"
        return self


class ResearchQualityFilterConfig(BaseModel):
    mode: str = "strict"
    apply_windows: list[str] = Field(default_factory=lambda: ["is", "oos"])
    blocked_anomaly_flags: list[str] = Field(
        default_factory=lambda: ["PF_EXTREME", "PAYOFF_EXTREME", "LOSS_TINY_VS_SPREAD"]
    )
    min_is_trades: int = 120
    min_oos_trades: int = 120
    require_orders_submitted: bool = True
    require_trades_filled: bool = True

    @model_validator(mode="after")
    def validate_quality_filter(self) -> ResearchQualityFilterConfig:
        self.mode = str(self.mode or "strict").strip().lower()
        if self.mode not in {"strict", "off"}:
            raise ValueError("research.optimize.quality_filter.mode must be strict or off")

        normalized_windows: list[str] = []
        seen_windows: set[str] = set()
        for item in self.apply_windows:
            key = str(item).strip().lower()
            if key not in {"is", "oos"}:
                continue
            if key in seen_windows:
                continue
            seen_windows.add(key)
            normalized_windows.append(key)
        if not normalized_windows:
            raise ValueError("research.optimize.quality_filter.apply_windows must contain is and/or oos")
        self.apply_windows = normalized_windows

        normalized_flags: list[str] = []
        seen_flags: set[str] = set()
        for item in self.blocked_anomaly_flags:
            key = str(item).strip().upper()
            if not key or key in seen_flags:
                continue
            seen_flags.add(key)
            normalized_flags.append(key)
        if self.mode == "strict" and not normalized_flags:
            raise ValueError("research.optimize.quality_filter.blocked_anomaly_flags cannot be empty in strict mode")
        self.blocked_anomaly_flags = normalized_flags

        if self.min_is_trades < 0:
            raise ValueError("research.optimize.quality_filter.min_is_trades must be >= 0")
        if self.min_oos_trades < 0:
            raise ValueError("research.optimize.quality_filter.min_oos_trades must be >= 0")
        return self


class ResearchOptimizeConfig(BaseModel):
    enabled: bool = False
    runtime_budget: str = "deep"
    split_ratio_is: float = 0.70
    min_days_is: int = 30
    min_days_oos: int = 30
    objective_mode: str = "risk_adjusted_pnl_dd"
    dd_cap_basis: str = "both"
    max_workers: int = 3
    seed: int = 42
    top_gate_keep: int = 10
    top_final_keep: int = 20
    capitals: list[ResearchCapitalConfig] = Field(default_factory=list)
    capital_run_mode: str = "sequential"
    quality_filter: ResearchQualityFilterConfig = Field(default_factory=ResearchQualityFilterConfig)

    @model_validator(mode="after")
    def validate_optimize(self) -> ResearchOptimizeConfig:
        self.runtime_budget = str(self.runtime_budget or "deep").strip().lower()
        if self.runtime_budget not in {"quick", "medium", "deep"}:
            raise ValueError("research.optimize.runtime_budget must be quick, medium or deep")
        if self.split_ratio_is <= 0.0 or self.split_ratio_is >= 1.0:
            raise ValueError("research.optimize.split_ratio_is must be in (0, 1)")
        if self.min_days_is < 1:
            raise ValueError("research.optimize.min_days_is must be >= 1")
        if self.min_days_oos < 1:
            raise ValueError("research.optimize.min_days_oos must be >= 1")
        self.objective_mode = str(self.objective_mode or "risk_adjusted_pnl_dd").strip().lower()
        if self.objective_mode not in {"pnl_dd_cap", "risk_adjusted_pnl_dd"}:
            raise ValueError("research.optimize.objective_mode must be pnl_dd_cap or risk_adjusted_pnl_dd")
        self.dd_cap_basis = str(self.dd_cap_basis or "both").strip().lower()
        if self.dd_cap_basis not in {"initial", "peak", "both"}:
            raise ValueError("research.optimize.dd_cap_basis must be one of: initial, peak, both")
        if self.max_workers < 1:
            raise ValueError("research.optimize.max_workers must be >= 1")
        if self.max_workers > 3:
            self.max_workers = 3
        if self.top_gate_keep < 1:
            raise ValueError("research.optimize.top_gate_keep must be >= 1")
        if self.top_final_keep < 1:
            raise ValueError("research.optimize.top_final_keep must be >= 1")
        self.capital_run_mode = str(self.capital_run_mode or "sequential").strip().lower()
        if self.capital_run_mode not in {"sequential", "parallel"}:
            raise ValueError("research.optimize.capital_run_mode must be sequential or parallel")
        return self


class ResearchGateSearchSpaceConfig(BaseModel):
    trend_thr: list[float] = Field(
        default_factory=lambda: [0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0025, 0.0030]
    )
    trend_vol_news_thr: list[float] = Field(default_factory=lambda: [0.0006, 0.0010, 0.0015])
    trend_vol_news_pre_post: list[list[int]] = Field(default_factory=lambda: [[15, 15], [30, 30], [45, 45], [60, 60]])
    trend_vol_news_vol_max: list[float] = Field(default_factory=lambda: [0.015, 0.020, 0.025])
    trend_vol_news_max_spread_mult: list[float] = Field(default_factory=lambda: [1.0, 1.5])

    @model_validator(mode="after")
    def validate_gate_space(self) -> ResearchGateSearchSpaceConfig:
        def _normalize_float_list(values: list[float], field_name: str) -> list[float]:
            normalized: list[float] = []
            seen: set[float] = set()
            for item in values:
                value = float(item)
                if value <= 0:
                    raise ValueError(f"research.search_space.gate.{field_name} values must be > 0")
                if value in seen:
                    continue
                seen.add(value)
                normalized.append(value)
            if not normalized:
                raise ValueError(f"research.search_space.gate.{field_name} must contain at least one value")
            return normalized

        self.trend_thr = _normalize_float_list(self.trend_thr, "trend_thr")
        self.trend_vol_news_thr = _normalize_float_list(self.trend_vol_news_thr, "trend_vol_news_thr")
        self.trend_vol_news_vol_max = _normalize_float_list(self.trend_vol_news_vol_max, "trend_vol_news_vol_max")
        self.trend_vol_news_max_spread_mult = _normalize_float_list(
            self.trend_vol_news_max_spread_mult, "trend_vol_news_max_spread_mult"
        )

        normalized_pre_post: list[list[int]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for pair in self.trend_vol_news_pre_post:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("research.search_space.gate.trend_vol_news_pre_post values must be [pre, post]")
            pre = int(pair[0])
            post = int(pair[1])
            if pre < 0 or post < 0:
                raise ValueError("research.search_space.gate.trend_vol_news_pre_post values must be >= 0")
            key = (pre, post)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            normalized_pre_post.append([pre, post])
        if not normalized_pre_post:
            raise ValueError("research.search_space.gate.trend_vol_news_pre_post must contain at least one pair")
        self.trend_vol_news_pre_post = normalized_pre_post
        return self


class ResearchRiskProfileConfig(BaseModel):
    name: str
    risk_per_trade: float
    max_trades_per_day: int
    max_total_risk_pct: float
    daily_stop_pct: float

    @model_validator(mode="after")
    def validate_profile(self) -> ResearchRiskProfileConfig:
        self.name = str(self.name or "").strip().upper()
        if not self.name:
            raise ValueError("research.search_space.risk_profiles.name cannot be empty")
        if self.risk_per_trade <= 0:
            raise ValueError("research.search_space.risk_profiles.risk_per_trade must be > 0")
        if self.max_trades_per_day < 1:
            raise ValueError("research.search_space.risk_profiles.max_trades_per_day must be >= 1")
        if self.max_total_risk_pct <= 0:
            raise ValueError("research.search_space.risk_profiles.max_total_risk_pct must be > 0")
        if self.daily_stop_pct <= 0:
            raise ValueError("research.search_space.risk_profiles.daily_stop_pct must be > 0")
        return self


class ResearchSearchSpaceConfig(BaseModel):
    gate: ResearchGateSearchSpaceConfig = Field(default_factory=ResearchGateSearchSpaceConfig)
    risk_profiles: list[ResearchRiskProfileConfig] = Field(
        default_factory=lambda: [
            ResearchRiskProfileConfig(
                name="RISK_01",
                risk_per_trade=0.003,
                max_trades_per_day=2,
                max_total_risk_pct=0.010,
                daily_stop_pct=0.010,
            ),
            ResearchRiskProfileConfig(
                name="RISK_02",
                risk_per_trade=0.003,
                max_trades_per_day=3,
                max_total_risk_pct=0.010,
                daily_stop_pct=0.010,
            ),
            ResearchRiskProfileConfig(
                name="RISK_03",
                risk_per_trade=0.003,
                max_trades_per_day=4,
                max_total_risk_pct=0.015,
                daily_stop_pct=0.010,
            ),
            ResearchRiskProfileConfig(
                name="RISK_04",
                risk_per_trade=0.005,
                max_trades_per_day=2,
                max_total_risk_pct=0.010,
                daily_stop_pct=0.010,
            ),
            ResearchRiskProfileConfig(
                name="RISK_05",
                risk_per_trade=0.005,
                max_trades_per_day=3,
                max_total_risk_pct=0.015,
                daily_stop_pct=0.015,
            ),
            ResearchRiskProfileConfig(
                name="RISK_06",
                risk_per_trade=0.005,
                max_trades_per_day=4,
                max_total_risk_pct=0.020,
                daily_stop_pct=0.015,
            ),
            ResearchRiskProfileConfig(
                name="RISK_07",
                risk_per_trade=0.0075,
                max_trades_per_day=2,
                max_total_risk_pct=0.015,
                daily_stop_pct=0.010,
            ),
            ResearchRiskProfileConfig(
                name="RISK_08",
                risk_per_trade=0.0075,
                max_trades_per_day=3,
                max_total_risk_pct=0.020,
                daily_stop_pct=0.015,
            ),
            ResearchRiskProfileConfig(
                name="RISK_09",
                risk_per_trade=0.0075,
                max_trades_per_day=4,
                max_total_risk_pct=0.020,
                daily_stop_pct=0.020,
            ),
            ResearchRiskProfileConfig(
                name="RISK_10",
                risk_per_trade=0.010,
                max_trades_per_day=3,
                max_total_risk_pct=0.015,
                daily_stop_pct=0.015,
            ),
            ResearchRiskProfileConfig(
                name="RISK_11",
                risk_per_trade=0.010,
                max_trades_per_day=4,
                max_total_risk_pct=0.020,
                daily_stop_pct=0.020,
            ),
            ResearchRiskProfileConfig(
                name="RISK_12",
                risk_per_trade=0.010,
                max_trades_per_day=6,
                max_total_risk_pct=0.020,
                daily_stop_pct=0.020,
            ),
        ]
    )

    @model_validator(mode="after")
    def validate_search_space(self) -> ResearchSearchSpaceConfig:
        if not self.risk_profiles:
            raise ValueError("research.search_space.risk_profiles must contain at least one profile")
        seen_names: set[str] = set()
        normalized: list[ResearchRiskProfileConfig] = []
        for profile in self.risk_profiles:
            name = str(profile.name).strip().upper()
            if name in seen_names:
                continue
            seen_names.add(name)
            normalized.append(profile)
        self.risk_profiles = normalized
        return self


class ResearchConfig(BaseModel):
    objective_mode: str = "pnl_dd_cap"
    dd_cap_pct: float = 25.0
    dd_cap_basis: str = "both"
    min_trades_oos: int = 120
    symbols: list[str] = Field(default_factory=lambda: ["XAUUSD", "BTCUSD"])
    max_workers: int = 3
    seed: int = 42
    optimize: ResearchOptimizeConfig = Field(default_factory=ResearchOptimizeConfig)
    search_space: ResearchSearchSpaceConfig = Field(default_factory=ResearchSearchSpaceConfig)

    @model_validator(mode="after")
    def validate_research(self) -> ResearchConfig:
        self.objective_mode = str(self.objective_mode or "pnl_dd_cap").strip().lower()
        if self.objective_mode not in {"pnl_dd_cap", "risk_adjusted_pnl_dd"}:
            raise ValueError("research.objective_mode must be pnl_dd_cap or risk_adjusted_pnl_dd")
        if self.dd_cap_pct <= 0:
            raise ValueError("research.dd_cap_pct must be > 0")
        self.dd_cap_basis = str(self.dd_cap_basis or "both").strip().lower()
        if self.dd_cap_basis not in {"initial", "peak", "both"}:
            raise ValueError("research.dd_cap_basis must be one of: initial, peak, both")
        if self.min_trades_oos < 0:
            raise ValueError("research.min_trades_oos must be >= 0")
        normalized_symbols: list[str] = []
        seen: set[str] = set()
        for symbol in self.symbols:
            key = str(symbol).strip().upper()
            if not key or key in seen:
                continue
            normalized_symbols.append(key)
            seen.add(key)
        self.symbols = normalized_symbols
        if self.max_workers < 1:
            raise ValueError("research.max_workers must be >= 1")
        if self.max_workers > 3:
            self.max_workers = 3
        return self


class BacktestRuntimeConfig(BaseModel):
    feature_cache_enabled: bool = True
    feature_cache_dir: str = "runs/cache/features"
    parallel_workers: int = 3
    deterministic: bool = True

    @model_validator(mode="after")
    def validate_runtime(self) -> BacktestRuntimeConfig:
        if self.parallel_workers < 1:
            raise ValueError("backtest_runtime.parallel_workers must be >= 1")
        if self.parallel_workers > 3:
            self.parallel_workers = 3
        cache_dir = str(self.feature_cache_dir or "").strip()
        self.feature_cache_dir = cache_dir or "runs/cache/features"
        return self


class AppConfig(BaseModel):
    timezone: str = "Europe/Warsaw"
    account_currency: str = "USD"
    fx_conversion_fee_rate: float = 0.007
    fx_fee_mode: str = "all_in_rate"
    fx_rate_source: str = "static"
    fx_static_rates: dict[str, float] = Field(default_factory=dict)
    fx_apply_to: list[str] = Field(default_factory=lambda: ["pnl", "swap", "commission"])
    reporting_currency: str = "account"
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    assets: list[AssetConfig] = Field(default_factory=list)
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    swings: SwingConfig = Field(default_factory=SwingConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    displacement: DisplacementConfig = Field(default_factory=DisplacementConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    spread_filter: SpreadFilterConfig = Field(default_factory=SpreadFilterConfig)
    spread_guard: SpreadGuardGateConfig = Field(default_factory=SpreadGuardGateConfig)
    news_gate: NewsGateConfig = Field(default_factory=NewsGateConfig)
    daily_gate: DailyGateConfig = Field(default_factory=DailyGateConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    capital_ramp: CapitalRampConfig = Field(default_factory=CapitalRampConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    market_hours: MarketHoursConfig = Field(default_factory=MarketHoursConfig)
    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    strategy_runtime: StrategyRuntimeConfig = Field(default_factory=StrategyRuntimeConfig)
    strategy_router: StrategyRouterConfig = Field(default_factory=StrategyRouterConfig)
    scalp: ScalpStrategyConfig = Field(default_factory=ScalpStrategyConfig)
    portfolio: PortfolioSupervisorConfig = Field(default_factory=PortfolioSupervisorConfig)
    decision_policy: DecisionPolicyConfig = Field(default_factory=DecisionPolicyConfig)
    backtest_tuning: BacktestTuningConfig = Field(default_factory=BacktestTuningConfig)
    orderflow: OrderflowConfig = Field(default_factory=OrderflowConfig)
    risk_budget: RiskBudgetConfig = Field(default_factory=RiskBudgetConfig)
    score_tiers: ScoreTiersConfig = Field(default_factory=ScoreTiersConfig)
    score_v3: ScoreV3ConfigModel = Field(default_factory=ScoreV3ConfigModel)
    fvg_entry_ladder: FvgEntryLadderConfig = Field(default_factory=FvgEntryLadderConfig)
    exits: ExitsConfig = Field(default_factory=ExitsConfig)
    correlation_v2: CorrelationV2Config = Field(default_factory=CorrelationV2Config)
    paper_costs: PaperCostPydanticConfig = Field(default_factory=PaperCostPydanticConfig)
    micro_loss_defense: MicroLossDefensePydanticConfig = Field(default_factory=MicroLossDefensePydanticConfig)
    adaptive_threshold: AdaptiveThresholdPydanticConfig = Field(default_factory=AdaptiveThresholdPydanticConfig)
    multi_tp: MultiTPConfig = Field(default_factory=MultiTPConfig)
    tier_sizing: TierSizingConfig = Field(default_factory=TierSizingConfig)
    session_filter: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    compound_equity: CompoundEquityConfig = Field(default_factory=CompoundEquityConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    backtest_runtime: BacktestRuntimeConfig = Field(default_factory=BacktestRuntimeConfig)

    @model_validator(mode="after")
    def normalize_assets(self) -> AppConfig:
        self.account_currency = str(self.account_currency or "USD").strip().upper() or "USD"
        if bool(self.capital_ramp.enabled) and self.account_currency != "PLN":
            raise ValueError("capital_ramp.enabled requires account_currency=PLN")
        self.fx_fee_mode = str(self.fx_fee_mode or "all_in_rate").strip().lower()
        if self.fx_fee_mode not in {"all_in_rate"}:
            raise ValueError("fx_fee_mode must be all_in_rate")
        self.fx_rate_source = str(self.fx_rate_source or "static").strip().lower()
        if self.fx_rate_source not in {"static", "provider"}:
            raise ValueError("fx_rate_source must be static or provider")
        if not (0.0 <= float(self.fx_conversion_fee_rate) <= 1.0):
            raise ValueError("fx_conversion_fee_rate must be in [0,1]")
        normalized_rates: dict[str, float] = {}
        for key, value in self.fx_static_rates.items():
            pair = str(key).strip().upper()
            if len(pair) < 6:
                raise ValueError(f"fx_static_rates key '{pair}' must look like USDPLN")
            rate = float(value)
            if rate <= 0:
                raise ValueError(f"fx_static_rates[{pair}] must be > 0")
            normalized_rates[pair] = rate
        self.fx_static_rates = normalized_rates
        valid_apply_to = {"pnl", "swap", "commission"}
        apply_to: list[str] = []
        seen_apply_to: set[str] = set()
        for item in self.fx_apply_to:
            key = str(item).strip().lower()
            if not key:
                continue
            if key not in valid_apply_to:
                raise ValueError(f"fx_apply_to contains unsupported value '{item}'")
            if key in seen_apply_to:
                continue
            seen_apply_to.add(key)
            apply_to.append(key)
        self.fx_apply_to = apply_to
        reporting = str(self.reporting_currency or "account").strip().lower()
        if reporting != "account":
            raise ValueError("reporting_currency currently supports only 'account'")
        self.reporting_currency = reporting

        if not self.assets:
            self.assets = [AssetConfig(**self.instrument.model_dump(), trade_enabled=True)]
        dedup: list[AssetConfig] = []
        seen: set[str] = set()
        for asset in self.assets:
            epic = asset.epic.strip().upper()
            if not epic or epic in seen:
                continue
            asset.epic = epic
            if not str(asset.instrument_currency or "").strip():
                asset.instrument_currency = asset.currency
            seen.add(epic)
            dedup.append(asset)
        self.assets = dedup
        return self


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)
