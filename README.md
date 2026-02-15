# Capital.com DEMO Multi-Asset Trading Bot (Python 3.11)

## Disclaimer (Very Early Stage)
- This is a very early-stage project.
- This repository is for educational purposes only and is not investment advice.
- Trading and investing involve risk, including possible loss of capital.

This repo contains a DEMO/paper trading bot for Capital.com Open API with:
- ICT + trend-following strategy logic
- LIMIT-only entries at 50% FVG
- SQLite journaling and restart resilience
- per-asset + global risk controls
- multi-asset runtime (trade + observe)
- backtest and walk-forward modes
- dashboard snapshot + Telegram/Discord alerts

## Safety
- DEMO only.
- Use only official HTTP API.

## Install
1. `pip install -r requirements.txt`
2. `copy .env.example .env`
3. Fill `.env` values.

## Run
- Dry-run loop: `python main.py --dry-run`
- Paper (DEMO API orders): `python main.py --paper`
- Pipeline trace as JSON logs: `python main.py --dry-run --state-log json`
- One forced test order: `python main.py --dry-run --test-order`
- Test order short on selected epic: `python main.py --dry-run --test-order --test-side SHORT --test-epic GOLD`

## Backtest
CSV must include columns: `timestamp,open,high,low,close` (M5 candles).
- Backtest: `python main.py --backtest --backtest-data data.csv --backtest-epic GOLD`
- Walk-forward: `python main.py --backtest --walk-forward --wf-splits 4 --backtest-data data.csv --backtest-epic GOLD`
- Auto data loader (parquet folder `data/<source>/<SYMBOL>/<SIDE>/<TF>/YYYY/MM.parquet`):
  `python main.py --backtest --backtest-symbols XAUUSD,EURUSD --backtest-tf 5m --backtest-start 2023-01-01 --backtest-end 2023-12-31 --backtest-price mid`
- Auto loader + optional fetch:
  `python main.py --backtest --backtest-symbols XAUUSD,EURUSD,US100,US500,BTCUSD --backtest-start 2023-01-01 --backtest-end 2023-12-31 --backtest-autofetch`
- ScoreV3 enabled backtest (see config variants):
  `python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_PNL_R83.yaml --initial-equity 100`

## Backtest reports
Detailed report artifacts are generated automatically in backtest mode (unless disabled).

- Enable/disable: `--report` / `--no-report`
- Base output directory: `--report-dir reports/backtest`
- Formats: `--report-formats json,csv,png,html`
- Auto-open HTML after run: `--report-open`

Each report run is saved in its own timestamped folder:
`reports/backtest/{symbol}_{tf}_{start}_{end}_{variant}_{YYYYmmdd-HHMMSS}/`

Artifacts:
- `report.json` (full meta + metrics + extra)
- `summary.json` (headline metrics)
- `trades.csv`
- `equity.csv`
- `charts/equity_curve.png`
- `charts/drawdown.png`
- `charts/pnl_per_trade.png`
- `charts/pnl_hist.png`
- optional `charts/pnl_by_month.png`
- optional `report.html`

## DailyGate (hard gate)
- CLI mode: `--daily-gate off|trend|trend_vol_news`
- A/B run (all three modes in one launch): `--daily-gate-ab`
- Optional grid search for gate params: `--daily-gate-grid-search`
- Optional runtime overrides:
  - `--daily-gate-thr`
  - `--daily-gate-pre-minutes`
  - `--daily-gate-post-minutes`
  - `--daily-gate-vol-max`
  - `--daily-gate-max-spread`

Example A/B on XAUUSD 5m:
`python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config config.variant_B.yaml --initial-equity 100 --daily-gate-ab`

## Currency conversion fee (all-in rate, 0.7%)
Backtest/paper now support explicit account currency conversion with Capital.com-style fee embedded in the FX rate.

Config keys:
```yaml
account_currency: "PLN"
fx_conversion_fee_rate: 0.007
fx_fee_mode: "all_in_rate"
fx_rate_source: "static"
fx_static_rates:
  USDPLN: 4.00
fx_apply_to: ["pnl", "swap", "commission"]
reporting_currency: "account"
```

Per-asset:
```yaml
assets:
  - epic: "XAUUSD"
    instrument_currency: "USD"
```

Notes:
- FX fee is applied only when a conversion is needed (`instrument_currency != account_currency`).
- The fee is modeled as a less favorable all-in rate, not as % of notional on entry/exit.
- `fx_cost_sum` in reports is the explicit conversion drag.
- Backtest trade reports include breakdown fields: `spread_cost`, `slippage_cost`, `commission_cost`, `swap_cost`, `fx_cost`.

## Key .env fields
```env
CAPITAL_BASE_URL=https://demo-api-capital.backend-capital.com/api/v1
CAPITAL_API_KEY=
CAPITAL_IDENTIFIER=
CAPITAL_API_PASSWORD=
CAPITAL_ACCOUNT_ID=

CAPITAL_EPIC=GOLD
CAPITAL_TRADE_EPICS=GOLD,BTCUSD,EURUSD,US100,US500
CAPITAL_WATCH_EPICS=

SQLITE_PATH_TEMPLATE=bot_state_{mode}.db
# SQLITE_PATH=bot_state.db  # auto -> bot_state_dry.db / bot_state_paper.db

NEWS_PROVIDER=dummy
NEWS_HTTP_URL=
NEWS_HTTP_TOKEN=

CAPITAL_RATE_LIMIT_RPS=2.0
CAPITAL_RATE_LIMIT_BURST=5
CAPITAL_REQUEST_MAX_ATTEMPTS=6
CAPITAL_BACKOFF_BASE_SECONDS=0.5
CAPITAL_BACKOFF_MAX_SECONDS=20
CAPITAL_RECONNECT_SHORT_RETRIES=2
CAPITAL_SESSION_REFRESH_MIN_INTERVAL_SECONDS=5

QUOTE_REFRESH_TRADE_SECONDS=30
QUOTE_REFRESH_OBSERVE_SECONDS=60
CANDLE_CLOSE_GRACE_SECONDS=3
CANDLE_RETRY_SECONDS=15
SYNC_PENDING_SECONDS=30
SYNC_POSITIONS_SECONDS=30

DASHBOARD_PATH=runtime_dashboard.json
ALERT_DISCORD_WEBHOOK=
ALERT_TELEGRAM_BOT_TOKEN=
ALERT_TELEGRAM_CHAT_ID=
ALERT_COOLDOWN_SECONDS=30

LOG_LEVEL=INFO
```

## Strategy and risk (current implementation)
- Multi-strategy router per symbol (multiple active strategies are allowed):
  - `SCALP_ICT_PA` (ICT/PA scalping)
  - `ORB_H4_RETEST` (H4 breakout + M5 retest)
  - `TREND_PULLBACK_M15` (trend continuation pullback)
  - `INDEX_EXISTING` (legacy index logic, preserved for `US100`/`US500`)
- **ScoreV3 Enhanced Scoring System** (optional, enabled per config):
  - 35-feature extraction: HTF alignment, FVG quality, trigger confirmations, volatility regime, session/time, entry quality
  - Heuristic scorer (default): improved rule-based scoring with session/volatility awareness
  - ML model support (future): LightGBM/LogisticRegression loaded from disk
  - Quantile-based tier mapping: A+ (top 10%), A (next 25%), B (next 30%), OBSERVE
  - Fill probability adjustment: scores adjusted by entry distance likelihood
  - Shadow observer: simulates outcomes for all candidates (including OBSERVE) to measure missed opportunities
- Global decision policy (V2 legacy or V3 enhanced):
  - `TRADE`: score `>= 65` (V2) or `>= 48` (V3)
  - `SMALL`: score `60-64` (V2) or `38-47.99` (V3)
  - `OBSERVE`: score `< 60` (V2) or `< 38` (V3)
- H1 bias from EMA + BOS.
- Premium/Discount gating.
- M15 sweep + rejection, M5 MSS + displacement + FVG.
- Entry only by LIMIT at FVG midpoint.
- SL behind swept level with ATR buffer.
- TP 2R default, 3R for A+ setup.
- +1R management: SL to BE + 50% partial.
- Daily stop, max trades/day, max positions, global exposure and correlation limits.
- News block window and pending-order cancel in blocked window.

## ScoreV3 Enhanced Scoring System

The ScoreV3 system is an optional enhancement that increases trading throughput 2-3x while maintaining risk management quality. It replaces the legacy V2 scorer with improved feature extraction and more permissive thresholds.

### Features
- **35 Feature Vector**: HTF alignment, FVG quality, trigger confirmations, volatility regime, session/time awareness, entry quality metrics
- **Heuristic Scorer**: Rule-based scoring with session and volatility awareness (max score ~91)
- **ML Model Support**: Future support for LightGBM/LogisticRegression models loaded from disk
- **Quantile Tiers**: A+ (top 10%), A (next 25%), B (next 30%), OBSERVE (bottom 35%)
- **Fill Probability**: Scores adjusted by entry distance likelihood for better execution prediction
- **Shadow Observer**: Simulates outcomes for all candidates (including OBSERVE) to measure missed opportunities

### Configuration
Enable in config variants:
```yaml
score_v3:
  enabled: true
  mode: heuristic  # or 'ml' for future model support
  trade_threshold: 48.0
  small_min: 38.0
  small_max: 47.99
  shadow_observer:
    enabled: true
    output_path: "data/shadow_candidates.jsonl"
```

### Decision Policy (V3)
- `TRADE`: score `>= 48` (vs V2's 65)
- `SMALL`: score `38-47.99` (vs V2's 60-64)
- `OBSERVE`: score `< 38` (vs V2's <60)

### Shadow Observer
Records all signal candidates (including those that would be OBSERVED) and simulates their full trade outcomes. Useful for:
- Measuring missed opportunities
- Validating scoring improvements
- Backtest comparison analysis

### Score Audit Tool
CLI tool for analyzing scoring performance:
```bash
# Generate funnel report
python tools/score_audit.py reports/backtest_dir --funnel

# Compare two backtests
python tools/score_audit.py reports/scorev3_dir --compare reports/baseline_dir

# Score distribution analysis
python tools/score_audit.py reports/backtest_dir --distribution
```

## Low-Equity Protection (micro accounts)
- For very small balances (default threshold: `250` in account currency), risk is auto-tightened.
- Effective risk per trade is reduced and capped (`low_equity_risk_multiplier`, `low_equity_risk_per_trade_cap`).
- Daily stop and max trades/day are tightened in low-equity mode.
- Optional min-size fallback can place `min_size` only when its risk is still within a strict cap (`low_equity_min_size_fallback_max_risk_pct`).

## Portfolio supervisor
- max open positions total: `2`
- max open per symbol: `1`
- daily risk budget: `2R`
- per-symbol cooldown (configurable in `strategy_router.symbols[].cooldown_seconds`)
- top entries per cycle: `portfolio.max_entries_per_cycle` (default `1`)
- if signals collide:
  - ranker sorts by score + confirmation bonus + execution penalty
  - supervisor picks TOP-K under limits/cooldowns

## Multi-asset runtime
- Assets are configured in `config.yaml` under `assets`.
- Each asset has its own state cache and daily stats.
- `trade_enabled: false` means observe-only (quotes and monitoring, no orders).
- State DB is isolated by mode (`bot_state_dry.db` / `bot_state_paper.db` by default).

## Strategy mapping config
`config.yaml`:
```yaml
decision_policy:
  trade_score_threshold: 65
  small_score_min: 60
  small_score_max: 64

strategy_router:
  symbols:
    - symbol: "GOLD"
      strategy: "SCALP_ICT_PA"
      priority: 90
      params:
        quality_gates:
          spread_ratio_max: 0.15
          min_confirm: 2
      risk:
        small_risk_multiplier: 0.45
    - symbol: "GOLD"
      strategy: "TREND_PULLBACK_M15"
      priority: 78
    - symbol: "BTCUSD"
      strategy: "SCALP_ICT_PA"
      priority: 92
    - symbol: "US100"
      strategy: "INDEX_EXISTING"
      priority: 98
      params:
        schedule:
          enabled: true
          windows: ["08:00-22:00"]
    - symbol: "US100"
      strategy: "ORB_H4_RETEST"
      priority: 74
```

## Runtime tuning
- Candle calculations are triggered on closed bars only (M5/M15/H1) with configurable close grace.
- Quote polling cadence is independent for trade vs observe assets.
- Heartbeat logs include top blockers and API retry/429 stats.

## Monitoring
- Dashboard JSON file is updated periodically (`monitoring.dashboard_path`).
- Alerts can be sent to Discord/Telegram (optional env fields).

## Storage
SQLite tables include:
- `journal_trades`
- `orders`
- `positions`
- `daily_stats`
- `spreads`
- `risk_state`

## Tests
Run:
`pytest -q`

Current unit tests cover:
- swing detection
- FVG detection
- MSS detection
- bias + premium/discount gating
- risk limits and news gate
- ScoreV3 feature extraction, scoring engine, shadow observer, and integration (43 additional tests)

## Notes
- Epic names differ by account. For Gold on many DEMO accounts use `GOLD` (not `XAUUSD`).
- If API returns accountId errors, verify DEMO account is active and `CAPITAL_ACCOUNT_ID` matches that account.
