# Capital.com DEMO Multi-Asset Trading Bot (Python 3.11)

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
- No secrets in repo (`.env` is ignored).
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
- Global decision policy:
  - `TRADE`: score `>= 65`
  - `SMALL`: score `60-64` (enabled, reduced risk)
  - `OBSERVE`: score `< 60`
- H1 bias from EMA + BOS.
- Premium/Discount gating.
- M15 sweep + rejection, M5 MSS + displacement + FVG.
- Entry only by LIMIT at FVG midpoint.
- SL behind swept level with ATR buffer.
- TP 2R default, 3R for A+ setup.
- +1R management: SL to BE + 50% partial.
- Daily stop, max trades/day, max positions, global exposure and correlation limits.
- News block window and pending-order cancel in blocked window.

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

## Notes
- Epic names differ by account. For Gold on many DEMO accounts use `GOLD` (not `XAUUSD`).
- If API returns accountId errors, verify DEMO account is active and `CAPITAL_ACCOUNT_ID` matches that account.
