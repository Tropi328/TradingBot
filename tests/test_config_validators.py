"""Tests for config validation rules added in the hardening pass.

Covers:
- InstrumentConfig: point_size, min_size, size_step > 0
- IndicatorsConfig: ema_period_h1, atr_period >= 1
- RiskConfig: equity > 0, sizing_mode validation, conditional fixed_qty/fixed_notional,
  max_trades_per_day >= 1, daily_stop_pct > 0, max_positions >= 1,
  cooldown_loss_streak >= 0, cooldown_minutes >= 0
- CapitalConfig: rate_limit_rps > 0, rate_limit_burst >= 1, request_max_attempts >= 1,
  backoff_base_seconds > 0, backoff_max_seconds > 0, reconnect_short_retries >= 0,
  session_refresh_min_interval_seconds >= 0
"""
from __future__ import annotations

import pytest

from bot.config import (
    AppConfig,
    CapitalConfig,
    IndicatorsConfig,
    InstrumentConfig,
    OpsConfig,
    RiskConfig,
)


# ── InstrumentConfig ──────────────────────────────────────────────────────

class TestInstrumentConfigValidation:
    def test_defaults_valid(self):
        cfg = InstrumentConfig()
        assert cfg.point_size > 0
        assert cfg.min_size > 0
        assert cfg.size_step > 0

    @pytest.mark.parametrize("field_name,bad_value", [
        ("point_size", 0),
        ("point_size", -1),
        ("min_size", 0),
        ("min_size", -0.01),
        ("size_step", 0),
        ("size_step", -5),
    ])
    def test_rejects_non_positive(self, field_name: str, bad_value: float):
        with pytest.raises(ValueError, match=field_name):
            InstrumentConfig(**{field_name: bad_value})


# ── IndicatorsConfig ─────────────────────────────────────────────────────

class TestIndicatorsConfigValidation:
    def test_defaults_valid(self):
        cfg = IndicatorsConfig()
        assert cfg.ema_period_h1 >= 1
        assert cfg.atr_period >= 1

    @pytest.mark.parametrize("field_name,bad_value", [
        ("ema_period_h1", 0),
        ("ema_period_h1", -5),
        ("atr_period", 0),
        ("atr_period", -1),
    ])
    def test_rejects_invalid_periods(self, field_name: str, bad_value: int):
        with pytest.raises(ValueError, match=field_name):
            IndicatorsConfig(**{field_name: bad_value})


# ── RiskConfig ───────────────────────────────────────────────────────────

class TestRiskConfigValidation:
    def test_defaults_valid(self):
        cfg = RiskConfig()
        assert cfg.equity > 0
        assert cfg.sizing_mode == "risk_pct_equity"

    def test_rejects_zero_equity(self):
        with pytest.raises(ValueError, match="equity"):
            RiskConfig(equity=0)

    def test_rejects_negative_equity(self):
        with pytest.raises(ValueError, match="equity"):
            RiskConfig(equity=-500)

    def test_rejects_invalid_sizing_mode(self):
        with pytest.raises(ValueError, match="sizing_mode"):
            RiskConfig(sizing_mode="invalid_mode")

    def test_fixed_qty_must_be_positive(self):
        with pytest.raises(ValueError, match="fixed_qty"):
            RiskConfig(sizing_mode="fixed_qty", fixed_qty=0)

    def test_fixed_notional_must_be_positive(self):
        with pytest.raises(ValueError, match="fixed_notional"):
            RiskConfig(sizing_mode="fixed_notional", fixed_notional=0)

    def test_fixed_qty_valid(self):
        cfg = RiskConfig(sizing_mode="fixed_qty", fixed_qty=1.5)
        assert cfg.sizing_mode == "fixed_qty"
        assert cfg.fixed_qty == 1.5

    def test_rejects_zero_max_trades(self):
        with pytest.raises(ValueError, match="max_trades_per_day"):
            RiskConfig(max_trades_per_day=0)

    def test_rejects_zero_daily_stop(self):
        with pytest.raises(ValueError, match="daily_stop_pct"):
            RiskConfig(daily_stop_pct=0)

    def test_rejects_zero_max_positions(self):
        with pytest.raises(ValueError, match="max_positions"):
            RiskConfig(max_positions=0)

    def test_rejects_negative_cooldown_streak(self):
        with pytest.raises(ValueError, match="cooldown_loss_streak"):
            RiskConfig(cooldown_loss_streak=-1)

    def test_rejects_negative_cooldown_minutes(self):
        with pytest.raises(ValueError, match="cooldown_minutes"):
            RiskConfig(cooldown_minutes=-1)

    def test_accepts_zero_cooldown(self):
        cfg = RiskConfig(cooldown_loss_streak=0, cooldown_minutes=0)
        assert cfg.cooldown_loss_streak == 0
        assert cfg.cooldown_minutes == 0

    def test_rejects_risk_per_trade_zero(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=0)

    def test_rejects_risk_per_trade_over_1(self):
        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=1.5)

    def test_rejects_invalid_risk_mode(self):
        with pytest.raises(ValueError, match="risk_mode"):
            RiskConfig(risk_mode="dollar")

    def test_accepts_cash_risk_mode(self):
        cfg = RiskConfig(risk_mode="cash")
        assert cfg.risk_mode == "cash"

    def test_min_risk_cash_nonnegative(self):
        with pytest.raises(ValueError, match="min_risk_cash_per_trade"):
            RiskConfig(min_risk_cash_per_trade=-1)

    def test_max_risk_cash_positive(self):
        with pytest.raises(ValueError, match="max_risk_cash_per_trade"):
            RiskConfig(max_risk_cash_per_trade=0)

    def test_min_cannot_exceed_max_risk_cash(self):
        with pytest.raises(ValueError, match="min_risk_cash_per_trade cannot exceed"):
            RiskConfig(min_risk_cash_per_trade=100, max_risk_cash_per_trade=50)


# ── CapitalConfig ────────────────────────────────────────────────────────

class TestCapitalConfigValidation:
    def test_defaults_valid(self):
        cfg = CapitalConfig()
        assert cfg.rate_limit_rps > 0
        assert cfg.rate_limit_burst >= 1

    @pytest.mark.parametrize("field_name,bad_value,match", [
        ("rate_limit_rps", 0, "rate_limit_rps"),
        ("rate_limit_rps", -1, "rate_limit_rps"),
        ("rate_limit_burst", 0, "rate_limit_burst"),
        ("request_max_attempts", 0, "request_max_attempts"),
        ("backoff_base_seconds", 0, "backoff_base_seconds"),
        ("backoff_max_seconds", 0, "backoff_max_seconds"),
        ("reconnect_short_retries", -1, "reconnect_short_retries"),
        ("session_refresh_min_interval_seconds", -1, "session_refresh_min_interval"),
    ])
    def test_rejects_invalid(self, field_name: str, bad_value: float | int, match: str):
        with pytest.raises(ValueError, match=match):
            CapitalConfig(**{field_name: bad_value})

    def test_accepts_zero_reconnect_retries(self):
        cfg = CapitalConfig(reconnect_short_retries=0)
        assert cfg.reconnect_short_retries == 0

    def test_accepts_zero_refresh_interval(self):
        cfg = CapitalConfig(session_refresh_min_interval_seconds=0)
        assert cfg.session_refresh_min_interval_seconds == 0


class TestOpsConfigValidation:
    def test_defaults_valid(self):
        cfg = OpsConfig()
        assert cfg.heartbeat_stale_seconds > 0
        assert cfg.watchdog_interval_seconds > 0
        assert cfg.backup_retention_days >= 0

    def test_rejects_invalid_heartbeat_stale(self):
        with pytest.raises(ValueError, match="ops.heartbeat_stale_seconds"):
            OpsConfig(heartbeat_stale_seconds=0)

    def test_rejects_invalid_watchdog_interval(self):
        with pytest.raises(ValueError, match="ops.watchdog_interval_seconds"):
            OpsConfig(watchdog_interval_seconds=0)

    def test_rejects_invalid_alert_cooldown(self):
        with pytest.raises(ValueError, match="ops.alert_cooldown_seconds"):
            OpsConfig(alert_cooldown_seconds=-1)

    def test_rejects_invalid_backup_retention(self):
        with pytest.raises(ValueError, match="ops.backup_retention_days"):
            OpsConfig(backup_retention_days=-1)


class TestCapitalRampValidation:
    def test_capital_ramp_requires_pln_account_currency(self):
        with pytest.raises(ValueError, match="account_currency=PLN"):
            AppConfig(account_currency="USD", capital_ramp={"enabled": True})

    def test_capital_ramp_accepts_pln_account_currency(self):
        cfg = AppConfig(account_currency="PLN", capital_ramp={"enabled": True})
        assert cfg.capital_ramp.enabled is True
