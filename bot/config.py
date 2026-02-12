from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class InstrumentConfig(BaseModel):
    epic: str = "XAUUSD"
    currency: str = "USD"
    point_size: float = 0.01
    minimal_tick_buffer: float = 0.05
    min_size: float = 0.01
    size_step: float = 0.01


class AssetConfig(InstrumentConfig):
    trade_enabled: bool = True


class TimeframesConfig(BaseModel):
    h1: str = "H1"
    m15: str = "M15"
    m5: str = "M5"


class IndicatorsConfig(BaseModel):
    ema_period_h1: int = 200
    atr_period: int = 14


class SwingConfig(BaseModel):
    fractal_left: int = 2
    fractal_right: int = 2


class SweepConfig(BaseModel):
    lookback_min_hours: int = 2
    lookback_max_hours: int = 8
    threshold_atr_multiplier: float = 0.15

    @model_validator(mode="after")
    def validate_lookback(self) -> "SweepConfig":
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


class NewsGateConfig(BaseModel):
    block_minutes: int = 60


class CorrelationGroupConfig(BaseModel):
    name: str
    epics: list[str]
    max_open_positions: int = 1

    @model_validator(mode="after")
    def normalize(self) -> "CorrelationGroupConfig":
        self.epics = [str(epic).strip().upper() for epic in self.epics if str(epic).strip()]
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be > 0")
        return self


class RiskConfig(BaseModel):
    equity: float = 10000
    risk_per_trade: float = 0.005
    max_trades_per_day: int = 3
    daily_stop_pct: float = 0.015
    max_positions: int = 1
    global_max_positions: int = 3
    max_total_risk_pct: float = 0.015
    cooldown_loss_streak: int = 2
    cooldown_minutes: int = 60
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
    def validate_risk(self) -> "RiskConfig":
        if self.risk_per_trade > 0.005:
            raise ValueError("risk_per_trade cannot exceed 0.005 (0.5%)")
        if self.max_total_risk_pct <= 0:
            raise ValueError("max_total_risk_pct must be > 0")
        if self.global_max_positions <= 0:
            raise ValueError("global_max_positions must be > 0")
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


class StrategyRuntimeConfig(BaseModel):
    allow_neutral_bias: bool = False
    neutral_bias_risk_multiplier: float = 0.5

    @model_validator(mode="after")
    def validate_values(self) -> "StrategyRuntimeConfig":
        if not (0 < self.neutral_bias_risk_multiplier <= 1.0):
            raise ValueError("neutral_bias_risk_multiplier must be in (0, 1]")
        return self


class CalendarConfig(BaseModel):
    provider: str = "dummy"
    dummy_file: str = "news_data/events.json"
    http_timeout_seconds: int = 10
    http_cache_ttl_seconds: int = 300


class WatchlistConfig(BaseModel):
    epics: list[str] = Field(default_factory=list)
    log_quotes: bool = True

    @model_validator(mode="after")
    def normalize_epics(self) -> "WatchlistConfig":
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
    def normalize(self) -> "StrategySymbolMappingConfig":
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
    def validate_thresholds(self) -> "ScalpStrategyConfig":
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
    def validate_values(self) -> "PortfolioSupervisorConfig":
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


class DecisionPolicyConfig(BaseModel):
    trade_score_threshold: float = 65.0
    small_score_min: float = 60.0
    small_score_max: float = 64.99
    trade_risk_multiplier: float = 1.0
    small_risk_multiplier_default: float = 0.45

    @model_validator(mode="after")
    def validate_values(self) -> "DecisionPolicyConfig":
        if not (0 < self.trade_score_threshold <= 100):
            raise ValueError("trade_score_threshold must be in (0,100]")
        if not (0 <= self.small_score_min <= self.small_score_max <= 100):
            raise ValueError("small_score_min/small_score_max must be in [0,100] and min<=max")
        if not (0 < self.trade_risk_multiplier <= 1.0):
            raise ValueError("trade_risk_multiplier must be in (0,1]")
        if not (0 < self.small_risk_multiplier_default <= 1.0):
            raise ValueError("small_risk_multiplier_default must be in (0,1]")
        return self


class BacktestTuningConfig(BaseModel):
    wait_reaction_timeout_bars: int = 15
    wait_mitigation_timeout_bars: int = 30
    reaction_timeout_force_enable: bool = True
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
    assumed_spread_by_symbol: dict[str, float] = Field(default_factory=lambda: {"XAUUSD": 0.2})

    @model_validator(mode="after")
    def validate_values(self) -> "BacktestTuningConfig":
        if self.wait_reaction_timeout_bars <= 0:
            raise ValueError("wait_reaction_timeout_bars must be > 0")
        if self.wait_mitigation_timeout_bars <= 0:
            raise ValueError("wait_mitigation_timeout_bars must be > 0")
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
        return self


class OrderflowConfig(BaseModel):
    default_mode: str = "LITE"
    full_symbols: list[str] = Field(default_factory=lambda: ["BTCUSD"])
    default_window: int = 64
    trigger_bonus_max: float = 10.0
    execution_bonus_max: float = 5.0
    divergence_penalty_min: float = 6.0
    divergence_penalty_max: float = 10.0
    small_soft_gate_confidence: float = 0.75
    small_soft_gate_chop: float = 0.75

    @model_validator(mode="after")
    def validate_values(self) -> "OrderflowConfig":
        self.default_mode = self.default_mode.strip().upper()
        if self.default_mode not in {"LITE", "FULL"}:
            raise ValueError("orderflow.default_mode must be LITE or FULL")
        self.full_symbols = [str(item).strip().upper() for item in self.full_symbols if str(item).strip()]
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


class AppConfig(BaseModel):
    timezone: str = "Europe/Warsaw"
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    assets: list[AssetConfig] = Field(default_factory=list)
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    swings: SwingConfig = Field(default_factory=SwingConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    displacement: DisplacementConfig = Field(default_factory=DisplacementConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    spread_filter: SpreadFilterConfig = Field(default_factory=SpreadFilterConfig)
    news_gate: NewsGateConfig = Field(default_factory=NewsGateConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    strategy_runtime: StrategyRuntimeConfig = Field(default_factory=StrategyRuntimeConfig)
    strategy_router: StrategyRouterConfig = Field(default_factory=StrategyRouterConfig)
    scalp: ScalpStrategyConfig = Field(default_factory=ScalpStrategyConfig)
    portfolio: PortfolioSupervisorConfig = Field(default_factory=PortfolioSupervisorConfig)
    decision_policy: DecisionPolicyConfig = Field(default_factory=DecisionPolicyConfig)
    backtest_tuning: BacktestTuningConfig = Field(default_factory=BacktestTuningConfig)
    orderflow: OrderflowConfig = Field(default_factory=OrderflowConfig)

    @model_validator(mode="after")
    def normalize_assets(self) -> "AppConfig":
        if not self.assets:
            self.assets = [AssetConfig(**self.instrument.model_dump(), trade_enabled=True)]
        dedup: list[AssetConfig] = []
        seen: set[str] = set()
        for asset in self.assets:
            epic = asset.epic.strip().upper()
            if not epic or epic in seen:
                continue
            asset.epic = epic
            seen.add(epic)
            dedup.append(asset)
        self.assets = dedup
        return self


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfig.model_validate(raw)
