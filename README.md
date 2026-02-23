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
- Keep viewers open after run (optional):
  `python main.py --backtest ... --dashboard --mc-viewer --hold-viewers`

Default behavior:
- `--dashboard` is OFF by default.
- Monte Carlo viewer auto-open is OFF in default configs.
- Backtest process exits automatically after run unless `--hold-viewers` is used.

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

## Research run (PnL with DD cap)
Use `--research-run` to execute an A/B sweep (`off`, `trend`, `trend_vol_news`) and rank results with:
- objective: `pnl_dd_cap`
- hard constraint: `max_drawdown_pct <= dd_cap_pct`
- tie-breakers: lower drawdown, then higher expectancy

CLI:
`python main.py --research-run --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_PNL_R83.yaml --research-symbols XAUUSD,BTCUSD --research-dd-cap 25 --research-dd-cap-basis both --research-workers 3`

Config:
```yaml
research:
  objective_mode: "pnl_dd_cap"
  dd_cap_pct: 25.0
  dd_cap_basis: "both"  # initial | peak | both
  min_trades_oos: 120
  symbols: ["XAUUSD", "BTCUSD"]
  max_workers: 3
  seed: 42

backtest_runtime:
  feature_cache_enabled: true
  feature_cache_dir: "runs/cache/features"
  parallel_workers: 3
  deterministic: true
```

Output:
- `reports/research/<timestamp>/research_summary.json`
- `reports/research/<timestamp>/research_summary.csv`
- `configs/variants/config.variant_RESEARCH_WINNER.yaml` (auto-generated effective config with winning `daily_gate.mode`)

If one symbol has missing data in the requested range, research stores a `partial_result` with:
- `available_symbols`
- `missing_symbols`
- tail of stderr from failed subprocess  
so you can still compare modes on symbols that completed.

## Research optimize (deep PnL+DD pipeline)
Use `--research-optimize` to run a two-stage optimizer:
- Stage A: gate-only sweep on IS (70% split by UTC day)
- Stage B: top gates x risk presets on IS+OOS (30%)
- ranking objective: `risk_adjusted_pnl_dd = oos_total_pnl_net / max(oos_dd_ref_pct, 0.25)`
- hard DD constraint enforced with `dd_cap_basis` (`both` recommended)

CLI:
`python main.py --research-optimize --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_PNL_R83.yaml --research-benchmark-symbols XAUUSD --research-dd-cap 25 --research-dd-cap-basis both --research-workers 3 --research-runtime-budget deep`

Key config:
```yaml
research:
  optimize:
    enabled: false
    runtime_budget: deep
    split_ratio_is: 0.70
    min_days_is: 30
    min_days_oos: 30
    objective_mode: "risk_adjusted_pnl_dd"
    dd_cap_basis: "both"
    max_workers: 3
    seed: 42
    top_gate_keep: 10
    top_final_keep: 20
  search_space:
    gate: ...
    risk_profiles: ...
```

Artifacts:
- `reports/research_opt/<timestamp>/search_space.json`
- `reports/research_opt/<timestamp>/split_info.json`
- `reports/research_opt/<timestamp>/stage_a_gate_is.csv`
- `reports/research_opt/<timestamp>/stage_b_gate_risk_is_oos.csv`
- `reports/research_opt/<timestamp>/top20.json`
- `reports/research_opt/<timestamp>/best.json`
- `reports/research_opt/<timestamp>/checkpoint.json` (resume state)
- `configs/variants/config.variant_RESEARCH_OPT_BEST.yaml` (auto-generated winner config)

## Recommended variants
- Stress/aggressive profile:
  `configs/variants/config.variant_PNL_R83.yaml`
- Stability baseline:
  `configs/variants/config.variant_SAFE_BASE.yaml`

Reference commands:
- R83:
  `python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_PNL_R83.yaml --no-dashboard --no-mc-viewer`
- SAFE baseline:
  `python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_SAFE_BASE.yaml --no-dashboard --no-mc-viewer`

## Capital Ramp 100 -> 1200 PLN
Feature is config-only (`capital_ramp.enabled: true`) and works in live/paper/backtest without changing signal logic.

Policy (fixed in code):
- start equity: `100 PLN`
- monthly topup: `+100 PLN`
- first topup: first day of next month
- topups stop when:
  - model equity reaches `1200 PLN`, or
  - after `31 Dec` of start year
- model equity basis: realized closed PnL only
- backtest multi-year: topups only in first campaign year

Required:
- `account_currency: PLN` when enabled

Starter variant:
- `configs/variants/config.variant_LIVE_CAPITAL_RAMP_100PLN.yaml`

Backtest example:
- `python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2025-02-01 --backtest-tf 5m --backtest-price mid --backtest-data-root data --config configs/variants/config.variant_LIVE_CAPITAL_RAMP_100PLN.yaml --no-dashboard --no-mc-viewer`

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

## Oracle Cloud deploy (paper first, BTC 24/7)
This repo now includes a ready-to-use `deploy/` package for Oracle Free Tier VPS:

- `deploy/systemd/bot-paper-fx.service` (FX/indices 24/5)
- `deploy/systemd/bot-paper-btc.service` (BTC 24/7)
- `deploy/systemd/bot-heartbeat.*` (anti-idle every 5 min)
- `deploy/systemd/bot-watchdog.*` (service + heartbeat checks every 1 min)
- `deploy/systemd/bot-backup.*` (daily backups)
- `deploy/systemd/bot-soak-report.*` (daily ops soak report, 06:05 UTC)
- `deploy/scripts/heartbeat.sh`
- `deploy/scripts/watchdog.sh`
- `deploy/scripts/backup_state.sh`
- `deploy/scripts/restore_state.sh`
- `deploy/scripts/soak_report.sh`
- `deploy/scripts/soak_closeout.sh`
- `deploy/env/paper_fx.env.example`
- `deploy/env/paper_btc.env.example`
- `deploy/logrotate/trading-bot`

Runtime split:
- FX process: `CAPITAL_TRADE_EPICS=GOLD,EURUSD,US100,US500` + DB `bot_state_paper_fx.db`
- BTC process: `CAPITAL_TRADE_EPICS=BTCUSD` + DB `bot_state_paper_btc.db`

### Market hours config
`BTCUSD` can be forced to 24/7 while other symbols stay weekdays only:

```yaml
market_hours:
  default_profile: "WEEKDAYS"
  symbol_profiles:
    BTCUSD: "ALWAYS"
```

Profiles:
- `WEEKDAYS`: Mon-Fri in configured bot timezone
- `ALWAYS`: no weekend block

### Quick setup on Oracle VPS
1. Create runtime dirs:
   `sudo mkdir -p /opt/trading-bot/{state,logs,runtime,backups,deploy/env}`
2. Copy repo to `/opt/trading-bot` and create venv.
3. Copy env templates:
   - `cp deploy/env/paper_fx.env.example deploy/env/paper_fx.env`
   - `cp deploy/env/paper_btc.env.example deploy/env/paper_btc.env`
4. Protect secrets:
   `chmod 600 /opt/trading-bot/deploy/env/*.env`
5. Make scripts executable:
   `chmod +x /opt/trading-bot/deploy/scripts/*.sh`
6. Install systemd units and timers:
   - `sudo cp deploy/systemd/*.service /etc/systemd/system/`
   - `sudo cp deploy/systemd/*.timer /etc/systemd/system/`
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now bot-paper-fx.service`
   - `sudo systemctl enable --now bot-paper-btc.service`
   - `sudo systemctl enable --now bot-heartbeat.timer`
   - `sudo systemctl enable --now bot-watchdog.timer`
   - `sudo systemctl enable --now bot-backup.timer`
   - `sudo systemctl enable --now bot-soak-report.timer`
7. Install logrotate:
   `sudo cp deploy/logrotate/trading-bot /etc/logrotate.d/trading-bot`

### Ops commands (one-shot)
- Healthcheck:
  `python main.py --ops-healthcheck --config config.yaml`
- Backup now:
  `python main.py --ops-backup-now --config config.yaml`
- Restore verify-only:
  `python main.py --ops-restore-verify backups/<timestamp>`

### Ops tools
- Deploy preflight:
  `python tools/deploy_preflight.py --root /opt/trading-bot --config config.yaml`
- Healthcheck tool:
  `python tools/ops_healthcheck.py --root /opt/trading-bot --config config.yaml`
- Daily soak report:
  `python tools/ops_soak_report.py --root /opt/trading-bot --config config.yaml --since-hours 24`
- 7-day closeout (GO/NO-GO):
  `python tools/ops_soak_closeout.py --root /opt/trading-bot --since-days 7 --end-date 2026-03-01 --verify-only-drill-done --full-restore-drill-done --json --output /opt/trading-bot/reports/ops/soak_closeout_2026-03-01.json`
- Watchdog one-shot:
  `python tools/ops_watchdog.py --root /opt/trading-bot --config config.yaml --json`

Runtime artifacts:
- `runtime/heartbeat.json`
- `runtime/watchdog.json`
- `runtime/soak_reports/YYYY-MM-DD.json`
- `backups/<timestamp>/manifest.json`
- `reports/ops/soak_closeout_YYYY-MM-DD.json`

### Security baseline
- SSH keys only, disable password login in `sshd_config`.
- Keep inbound firewall to `22/tcp` only.
- Run services as non-root user (`tradingbot`).
- Store API credentials only in `deploy/env/*.env` with `600` permissions.

### Recommended rollout stages
1. Start only `bot-paper-fx.service`, observe for 48h.
2. Enable `bot-paper-btc.service` and verify weekend cycles.
3. Run both for 7 days and review watchdog/heartbeat alerts.
4. Consider live rollout only after stable paper metrics and no recurring service faults.

### Next 4-6 weeks roadmap
1. Week 3-4: off-host backup (Object Storage/S3-compatible) + monthly restore drill.
2. Week 4-5: ops observability (uptime/restart/backup-age metrics + lightweight dashboard).
3. Week 5-6: live-readiness gate (7 days without critical alerts, backup pass-rate, restart stability).
4. In parallel: deduplicate scoring/gating logic between `main.py` and `bot/backtest/engine.py`.

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

## Decision Trace & Terminal Visualization

### Decision Trace (JSONL)
Every evaluation bar emits a **decision** event and every position open/close emits a **fill** event to `logs/decision_trace.jsonl` (append mode, per-line flush).

Enable via CLI:
```bash
# Backtest with trace
python main.py --backtest --backtest-symbols XAUUSD --backtest-start 2024-01-01 --backtest-end 2024-06-01 \
    --decision-trace logs/decision_trace.jsonl

# Paper mode with trace
python main.py --paper --decision-trace logs/decision_trace.jsonl
```

Or via `config.yaml`:
```yaml
diagnostics:
  decision_trace_enabled: true
  decision_trace_path: "logs/decision_trace.jsonl"
```

**Schema** (one JSON object per line):
| Field           | Type    | Decision | Fill |
|-----------------|---------|:--------:|:----:|
| `type`          | string  | yes      | yes  |
| `ts`            | ISO8601 | yes      | yes  |
| `symbol`        | string  | yes      | yes  |
| `candidates`    | int     | yes      |      |
| `signal`        | string  | yes      |      |
| `score`         | float   | yes      |      |
| `threshold`     | float   | yes      |      |
| `reject_reason` | string  | yes      |      |
| `side`          | string  |          | yes  |
| `pnl`           | float   |          | yes  |
| `equity_after`  | float   |          | yes  |
| `reason_close`  | string  |          | yes  |
| `holding_min`   | float   |          | yes  |

### Live TUI Dashboard
Tails the JSONL file and shows a live Rich TUI with equity sparkline, counters, reject histogram, recent decisions/fills, and alarms.
```bash
# Tail new lines only (best for live/paper)
python -m tools.termviz_live --path logs/decision_trace.jsonl

# Read from start (best after a completed backtest)
python -m tools.termviz_live --path logs/decision_trace.jsonl --from-start

# Custom refresh interval
python -m tools.termviz_live --refresh 0.3
```

### Backtest Replay
Animates a completed backtest from `trades.csv` (and optionally `equity.csv`) as a Rich TUI.
```bash
# Basic replay
python -m tools.termviz_replay --trades runs/latest/trades.csv

# With equity overlay and faster playback
python -m tools.termviz_replay --trades runs/latest/trades.csv --equity runs/latest/equity.csv --speed 5

# Only last 50 trades
python -m tools.termviz_replay --trades runs/latest/trades.csv --last 50
```

### Monte Carlo Live Viewer
Real-time Monte Carlo simulation viewer with two modes:

#### Terminal mode (default) - ASCII fan chart via plotext
A terminal-native viewer that draws equity percentile fan charts directly in a console window using `plotext`. Updates in real time as trades complete.

**Standalone usage:**
```bash
python tools/termviz_mc.py \
    --json reports/live/monte_carlo.json \
    --refresh 1.0
```

The terminal viewer shows:
- **Fan chart** with p5 / p25 / p50 / p75 / p95 equity percentile lines
- **Ruin threshold** horizontal line
- **Stats panel** - P(ruin), health score, equity percentiles, max DD, win rate, profit factor, consecutive loss streak

#### Process mode - matplotlib desktop window
A GUI window that displays the Monte Carlo simulation PNG and refreshes automatically. Uses `matplotlib` TkAgg backend.

**Standalone usage:**
```bash
python tools/monte_carlo_live_viewer.py \
    --png reports/live/monte_carlo.png \
    --json reports/live/monte_carlo.json \
    --refresh 1.0
```

#### Auto-launch with backtest
When `monte_carlo.live_window.enabled` is `true` and `viewer_mode` is `"terminal"` or `"process"`, the viewer window spawns automatically when a backtest starts.

**Config (`config.yaml`):**
```yaml
monte_carlo:
  sampling_mode: "iid_bootstrap"      # or "moving_block_bootstrap"
  block_size: 8                       # used by moving_block_bootstrap
  equity_mode_backtest: "initial"     # initial|current
  equity_mode_adaptive: "current"     # initial|current
  ruin_equity_floor_pct: null         # optional additional ruin trigger
  ruin_equity_floor_abs: null         # optional additional ruin trigger
  count_breakeven_as_loss: false
  adaptive:
    num_simulations_online: 250
    health_ema_alpha: 0.25
    max_step_up: 0.05
    max_step_down: 0.10
  live_window:
    enabled: true
    refresh_seconds: 1.0
    window_title: "IGNACY BOT - Monte Carlo Live"
    show_stats_overlay: true
    max_fps: 2
    open_on_start: true
    viewer_mode: "terminal"  # "terminal" (ASCII) | "process" (matplotlib GUI) | "manual"
    png_path: "reports/live/monte_carlo.png"
    json_path: "reports/live/monte_carlo.json"
```

**Dependencies:** `plotext` (terminal mode), `matplotlib` + `Pillow` (process mode) - all in `requirements.txt`.

The viewer handles missing files gracefully - it shows "Waiting for Monte Carlo data..." until the PNG appears.

MC JSON remains backward-compatible (`prob_ruin`, `equity_end_pXX`, `max_dd_pXX`, `health_score`, `step_percentiles`) and now adds simulation metadata (`sampling_mode`, `block_size`, `equity_mode`, ruin floors, `count_breakeven_as_loss`).

**Window title example:**
```
IGNACY BOT - MC Live | P(ruin>=50%)=12.4% | EqEnd p50=132.5 p5=88.1 p95=210.7 | maxDD p95=43.0%
```

## Notes
- Epic names differ by account. For Gold on many DEMO accounts use `GOLD` (not `XAUUSD`).
- If API returns accountId errors, verify DEMO account is active and `CAPITAL_ACCOUNT_ID` matches that account.
