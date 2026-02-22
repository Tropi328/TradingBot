# Trading Bot Oracle Paper Runbook

## Scope
This runbook covers the Oracle Free Tier paper environment:
- `bot-paper-fx.service` (GOLD, EURUSD, US100, US500)
- `bot-paper-btc.service` (BTCUSD 24/7)
- `bot-heartbeat.timer` + `bot-watchdog.timer`
- Daily backup via `bot-backup.timer`

## 1. Start / Stop / Restart

### Start all services
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now bot-paper-fx.service
sudo systemctl enable --now bot-paper-btc.service
sudo systemctl enable --now bot-heartbeat.timer
sudo systemctl enable --now bot-watchdog.timer
sudo systemctl enable --now bot-backup.timer
```

### Stop all services
```bash
sudo systemctl stop bot-paper-fx.service bot-paper-btc.service
sudo systemctl stop bot-heartbeat.timer bot-watchdog.timer bot-backup.timer
```

### Restart after config change
```bash
sudo systemctl restart bot-paper-fx.service
sudo systemctl restart bot-paper-btc.service
```

## 2. Healthcheck and preflight

### Runtime healthcheck (0/1)
```bash
python main.py --ops-healthcheck --config config.yaml
```

### One-shot backup
```bash
python main.py --ops-backup-now --config config.yaml
```

### Restore verify-only
```bash
python main.py --ops-restore-verify backups/20260222-010203
```

### Deploy preflight
```bash
python tools/deploy_preflight.py --root /opt/trading-bot --config config.yaml
```

## 3. If a service is `inactive` or `failed`
1. Check status:
```bash
systemctl status bot-paper-fx.service --no-pager
systemctl status bot-paper-btc.service --no-pager
```
2. Check logs:
```bash
tail -n 200 /opt/trading-bot/logs/bot-paper-fx.err.log
tail -n 200 /opt/trading-bot/logs/bot-paper-btc.err.log
```
3. Restart services:
```bash
sudo systemctl restart bot-paper-fx.service
sudo systemctl restart bot-paper-btc.service
```
4. Confirm health:
```bash
python main.py --ops-healthcheck --config config.yaml
```

## 4. If heartbeat is stale
1. Check timer and last execution:
```bash
systemctl status bot-heartbeat.timer --no-pager
systemctl status bot-heartbeat.service --no-pager
```
2. Run heartbeat manually:
```bash
sudo systemctl start bot-heartbeat.service
cat /opt/trading-bot/runtime/heartbeat.json
```
3. If network check fails, validate VPS DNS/HTTP egress.

## 5. Backup / Restore

### Manual backup
```bash
sudo systemctl start bot-backup.service
tail -n 100 /opt/trading-bot/logs/backup.log
```

### Full restore after failure
1. Verify backup:
```bash
/opt/trading-bot/deploy/scripts/restore_state.sh --verify-only /opt/trading-bot/backups/<timestamp>
```
2. Stop services:
```bash
sudo systemctl stop bot-paper-fx.service bot-paper-btc.service
```
3. Restore:
```bash
/opt/trading-bot/deploy/scripts/restore_state.sh /opt/trading-bot/backups/<timestamp>
```
4. Confirm integrity:
```bash
python main.py --ops-healthcheck --config config.yaml
```

## 6. Security baseline
- In `sshd_config`: `PasswordAuthentication no`
- Firewall: inbound `22/tcp` only
- Secrets only in `deploy/env/*.env` with `600` permissions
- Services run as non-root user `tradingbot`
