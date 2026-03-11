"""CLI config helpers, path resolution, CLI overrides — extracted from app_main.py."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bot.cli.args import parse_args as _parse_cli_args
from bot.config import AppConfig, ResearchCapitalConfig


def parse_args() -> argparse.Namespace:
    return _parse_cli_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_epics_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        epic = part.strip().upper()
        if not epic or epic in seen:
            continue
        seen.add(epic)
        out.append(epic)
    return out


def parse_research_capitals(raw: str | None) -> list[dict[str, object]]:
    if raw is None:
        return []
    entries: list[dict[str, object]] = []
    seen: set[tuple[float, str]] = set()
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid capital entry '{item}'. Expected amount:CCY.")
        amount_raw, currency_raw = item.split(":", 1)
        try:
            equity = float(amount_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid capital amount '{amount_raw}' in '{item}'.") from exc
        if equity <= 0:
            raise ValueError(f"Capital amount must be > 0 in '{item}'.")
        currency = str(currency_raw).strip().upper()
        if not currency:
            raise ValueError(f"Missing capital currency in '{item}'.")
        key = (equity, currency)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"equity": equity, "currency": currency})
    return entries


def _capital_dir_name(equity: float, currency: str) -> str:
    if float(equity).is_integer():
        amount = str(int(equity))
    else:
        amount = str(equity).replace(".", "p")
    return f"capital_{amount}_{currency.upper()}"


def _capital_file_tag(equity: float, currency: str) -> str:
    ccy = str(currency).strip().upper()
    if abs(float(equity) - 10000.0) <= 1e-9 and ccy == "USD":
        return "10K_USD"
    if abs(float(equity) - 100.0) <= 1e-9 and ccy == "PLN":
        return "100PLN"
    if float(equity).is_integer():
        amount = str(int(equity))
    else:
        amount = str(equity).replace(".", "p")
    return f"{amount}_{ccy}"


def _resolve_optimizer_capitals(config: AppConfig) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    seen: set[tuple[float, str]] = set()
    for item in config.research.optimize.capitals:
        equity = float(item.equity)
        currency = str(item.currency).strip().upper()
        key = (equity, currency)
        if key in seen:
            continue
        seen.add(key)
        resolved.append({"equity": equity, "currency": currency})
    if not resolved:
        resolved.append({"equity": float(config.risk.equity), "currency": str(config.account_currency).strip().upper()})
    return resolved


def _validate_optimizer_capitals(
    *,
    capitals: list[dict[str, object]],
    fx_static_rates: dict[str, float],
) -> None:
    for item in capitals:
        currency = str(item.get("currency", "USD")).strip().upper()
        if currency != "PLN":
            continue
        if "USDPLN" not in fx_static_rates:
            raise RuntimeError("Optimizer capital currency PLN requires fx_static_rates.USDPLN for conversion model.")


def _apply_cli_overrides(args: argparse.Namespace, config: AppConfig) -> None:
    if args.initial_equity is not None:
        config.risk.equity = float(args.initial_equity)
    if args.research_benchmark_symbols is not None:
        config.research.symbols = parse_epics_csv(args.research_benchmark_symbols)
    if args.research_symbols is not None:
        config.research.symbols = parse_epics_csv(args.research_symbols)
    if args.research_objective is not None:
        config.research.objective_mode = str(args.research_objective).strip().lower()
    if args.research_dd_cap is not None:
        config.research.dd_cap_pct = float(args.research_dd_cap)
    if args.research_dd_cap_basis is not None:
        config.research.dd_cap_basis = str(args.research_dd_cap_basis).strip().lower()
        config.research.optimize.dd_cap_basis = str(args.research_dd_cap_basis).strip().lower()
    if args.research_runtime_budget is not None:
        config.research.optimize.runtime_budget = str(args.research_runtime_budget).strip().lower()
    if args.research_workers is not None:
        workers = max(1, min(3, int(args.research_workers)))
        config.research.max_workers = workers
        config.research.optimize.max_workers = workers
        config.backtest_runtime.parallel_workers = workers
    if args.research_capitals is not None:
        parsed = parse_research_capitals(args.research_capitals)
        config.research.optimize.capitals = [
            ResearchCapitalConfig(equity=float(item["equity"]), currency=str(item["currency"])) for item in parsed
        ]
    if args.research_capital_run_mode is not None:
        config.research.optimize.capital_run_mode = str(args.research_capital_run_mode).strip().lower()
    if args.daily_gate is not None:
        config.daily_gate.mode = str(args.daily_gate).strip().lower()
    if args.daily_gate_thr is not None:
        config.daily_gate.thr = float(args.daily_gate_thr)
    if args.daily_gate_pre_minutes is not None:
        config.daily_gate.pre_minutes = int(args.daily_gate_pre_minutes)
    if args.daily_gate_post_minutes is not None:
        config.daily_gate.post_minutes = int(args.daily_gate_post_minutes)
    if args.daily_gate_vol_max is not None:
        config.daily_gate.vol_max = float(args.daily_gate_vol_max)
    if args.daily_gate_max_spread is not None:
        config.daily_gate.max_spread = float(args.daily_gate_max_spread)
    # Decision-trace CLI override
    if getattr(args, "decision_trace", None) is not None:
        config.diagnostics.decision_trace_enabled = True
        config.diagnostics.decision_trace_path = str(args.decision_trace)

    # Dashboard implies live decision trace stream.
    if getattr(args, "dashboard", False):
        config.diagnostics.decision_trace_enabled = True

    # Optional config-based auto-enable for backtest mode.
    if (
        getattr(args, "backtest", False)
        and bool(config.diagnostics.decision_trace_auto_enable_backtest)
        and not config.diagnostics.decision_trace_enabled
    ):
        config.diagnostics.decision_trace_enabled = True


def _daily_gate_mode(config: AppConfig) -> str:
    return str(config.daily_gate.mode).strip().lower()


def _resolve_runtime_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return root / path


def _resolve_config_path(root: Path, value: str) -> Path:
    direct = _resolve_runtime_path(root, value)
    if direct.exists():
        return direct
    raw = Path(value)
    candidates = [
        root / "configs" / raw,
        root / "configs" / "variants" / raw,
        root / "configs" / raw.name,
        root / "configs" / "variants" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct
