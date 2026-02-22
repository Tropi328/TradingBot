# 10-Min Incident Checklist (Oracle Paper)

## Daily 5-min routine (preventive)
1. `python main.py --ops-healthcheck --config config.yaml`
2. `python main.py --ops-backup-now --config config.yaml` (manual fallback if needed)
3. `python tools/ops_soak_report.py --root /opt/trading-bot --config config.yaml --since-hours 24`
4. Verify new daily artifact exists: `/opt/trading-bot/runtime/soak_reports/YYYY-MM-DD.json`

## 0-2 min: Triage
1. `python main.py --ops-healthcheck --config config.yaml`
2. `systemctl status bot-paper-fx.service bot-paper-btc.service --no-pager`
3. `cat /opt/trading-bot/runtime/watchdog.json`

## 2-5 min: Stabilize
1. If service failed:
   - `sudo systemctl restart bot-paper-fx.service`
   - `sudo systemctl restart bot-paper-btc.service`
2. If heartbeat is stale:
   - `sudo systemctl start bot-heartbeat.service`
3. Confirm Telegram alert status.

## 5-8 min: Data safety
1. Run backup:
   - `sudo systemctl start bot-backup.service`
2. Confirm `manifest.json` and `backup.log`.

## 8-10 min: Recovery decision
1. If issue returns:
   - verify latest backup:
     `python main.py --ops-restore-verify backups/<timestamp>`
2. If verify is OK and runtime is unstable:
   - execute full restore from `deploy/RUNBOOK.md`.

## After incident
1. Record:
   - incident start/end time,
   - root cause,
   - impacted services,
   - whether restart/restore was needed.
2. Add preventive action item to cloud-hardening backlog.

## 7-day closeout reminder
- Daily soak report must stay free from unresolved critical codes:
  - `E_SERVICE_DOWN`
  - `E_HEARTBEAT_STALE`
  - `E_DB_WRITE_FAIL`
  - `E_BACKUP_STALE`
