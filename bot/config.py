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
    def normalize_currency_fields(self) -> "InstrumentConfig":
        self.currency = str(self.currency).strip().upper() or "USD"
        instrument_ccy = str(self.instrument_currency or "").strip().upper()
        self.instrument_currency = instrument_ccy or self.currency
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
    def validate_values(self) -> "DailyGateConfig":
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
        self.allowed_strategies = [
            str(item).strip().upper()
            for item in self.allowed_strategies
            if str(item).strip()
        ]
        return self


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
    low_equity_mode_enabled: bool = True
    low_equity_threshold: float = 250.0
    low_equity_risk_multiplier: float = 0.35
    low_equity_risk_per_trade_cap: float = 0.02
    low_equity_max_trades_per_day: int = 2
    low_equity_daily_stop_pct: float = 0.01
    low_equity_min_size_fallback_enabled: bool = True
    low_equity_min_size_fallback_max_risk_pct: float = 0.02
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
        if self.risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be > 0")
        if self.risk_per_trade > 1.0:
            raise ValueError("risk_per_trade cannot exceed 1.0 (100.0%)")
        if self.max_total_risk_pct <= 0:
            raise ValueError("max_total_risk_pct must be > 0")
        if self.global_max_positions <= 0:
            raise ValueError("global_max_positions must be > 0")
        if self.low_equity_threshold <= 0:
            raise ValueError("low_equity_threshold must be > 0")
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
    fx_conversion_pct: float = 0.0

    @model_validator(mode="after")
    def validate_values(self) -> "BacktestTuningConfig":
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
    news_gate: NewsGateConfig = Field(default_factory=NewsGateConfig)
    daily_gate: DailyGateConfig = Field(default_factory=DailyGateConfig)
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
        self.account_currency = str(self.account_currency or "USD").strip().upper() or "USD"
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
