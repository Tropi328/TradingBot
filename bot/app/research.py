"""Research mode execution — extracted from app_main.py."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bot.app.config_helpers import (
    _resolve_config_path,
    _resolve_runtime_path,
    parse_epics_csv,
)
from bot.app.report_helpers import (
    _backtest_symbols,
    _extract_source_reports_from_report_dir,
)
from bot.config import AppConfig, AssetConfig
from bot.research.objective import (
    OBJECTIVE_FAIL_VALUE,
    aggregate_reports,
    objective_rank_key,
)

LOGGER = logging.getLogger("trading_bot")


def _build_research_subprocess_command(
    *,
    root: Path,
    args: argparse.Namespace,
    config_path: Path,
    symbols: list[str],
    gate_mode: str,
    run_report_dir: Path,
    run_auto_reports_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "main.py"),
        "--backtest",
        "--backtest-symbols",
        ",".join(symbols),
        "--backtest-start",
        str(args.backtest_start),
        "--backtest-end",
        str(args.backtest_end),
        "--backtest-tf",
        str(args.backtest_tf),
        "--backtest-price",
        str(args.backtest_price),
        "--backtest-data-root",
        str(args.backtest_data_root),
        "--backtest-spread",
        str(args.backtest_spread),
        "--backtest-slippage-points",
        str(args.backtest_slippage_points),
        "--backtest-slippage-atr-multiplier",
        str(args.backtest_slippage_atr_multiplier),
        "--backtest-variants",
        str(args.backtest_variants),
        "--daily-gate",
        str(gate_mode),
        "--report",
        "--report-formats",
        "json",
        "--report-dir",
        str(run_report_dir),
        "--backtest-reports-dir",
        str(run_auto_reports_dir),
        "--config",
        str(config_path),
        "--no-dashboard",
        "--no-mc-viewer",
    ]
    if str(args.backtest_source_priority or "").strip():
        command.extend(["--backtest-source-priority", str(args.backtest_source_priority)])
    if bool(args.backtest_autofetch):
        command.append("--backtest-autofetch")
    if bool(args.walk_forward):
        command.extend(["--walk-forward", "--wf-splits", str(args.wf_splits)])
    if args.initial_equity is not None:
        command.extend(["--initial-equity", str(args.initial_equity)])
    return command


def _run_single_research_candidate(
    *,
    root: Path,
    args: argparse.Namespace,
    config: AppConfig,
    config_path: Path,
    symbols: list[str],
    gate_mode: str,
    run_dir: Path,
) -> dict[str, object]:
    report_dir = run_dir / "detailed_reports"
    auto_reports_dir = run_dir / "auto_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    auto_reports_dir.mkdir(parents=True, exist_ok=True)

    command = _build_research_subprocess_command(
        root=root,
        args=args,
        config_path=config_path,
        symbols=symbols,
        gate_mode=gate_mode,
        run_report_dir=report_dir,
        run_auto_reports_dir=auto_reports_dir,
    )
    env = os.environ.copy()
    if config.backtest_runtime.deterministic:
        env["PYTHONHASHSEED"] = str(config.research.seed)

    started_at = time.time()
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
    elapsed_seconds = round(time.time() - started_at, 3)

    source_reports = _extract_source_reports_from_report_dir(report_dir)
    if source_reports:
        summary = aggregate_reports(
            source_reports,
            initial_equity=float(config.risk.equity),
            dd_cap_pct=float(config.research.dd_cap_pct),
            dd_cap_basis=str(config.research.dd_cap_basis),
            min_trades_oos=int(config.research.min_trades_oos),
            objective_mode=str(config.research.objective_mode),
        )
    else:
        summary = {
            "reports_count": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_pnl_net": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct_peak": 0.0,
            "max_drawdown_pct_initial": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy": 0.0,
            "expectancy_net": 0.0,
            "avg_r": 0.0,
            "cost_breakdown_net": {},
            "blocked_by_reason": {},
            "oos_pass": False,
            "constraint_dd_cap_pass_peak": False,
            "constraint_dd_cap_pass_initial": False,
            "constraint_dd_cap_pass": False,
            "objective_value": OBJECTIVE_FAIL_VALUE,
        }

    available_symbols = sorted(
        {
            str(item.get("symbol", "")).strip().upper()
            for item in source_reports
            if isinstance(item, dict) and str(item.get("symbol", "")).strip()
        }
    )
    missing_symbols = sorted(set(symbols) - set(available_symbols)) if available_symbols else list(symbols)
    partial_result = bool(result.returncode != 0 and bool(source_reports))
    if partial_result:
        summary = dict(summary)
        summary["partial_result"] = True
        summary["available_symbols"] = available_symbols
        summary["missing_symbols"] = missing_symbols

    return {
        "gate_mode": gate_mode,
        "command": command,
        "returncode": int(result.returncode),
        "partial_result": partial_result,
        "available_symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": "\n".join((result.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((result.stderr or "").splitlines()[-20:]),
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "auto_reports_dir": str(auto_reports_dir),
        "summary": summary,
    }


def _write_research_summary_files(research_dir: Path, payload: dict[str, object]) -> None:
    research_dir.mkdir(parents=True, exist_ok=True)
    summary_path = research_dir / "research_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return
    csv_path = research_dir / "research_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "gate_mode",
                "returncode",
                "partial_result",
                "available_symbols",
                "missing_symbols",
                "trades",
                "total_pnl_net",
                "max_drawdown_pct_peak",
                "max_drawdown_pct_initial",
                "max_drawdown_pct",
                "expectancy_net",
                "objective_value",
                "constraint_dd_cap_pass_peak",
                "constraint_dd_cap_pass_initial",
                "constraint_dd_cap_pass",
                "oos_pass",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(candidates, start=1):
            if not isinstance(item, dict):
                continue
            summary = item.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            writer.writerow(
                {
                    "rank": idx,
                    "gate_mode": item.get("gate_mode"),
                    "returncode": item.get("returncode"),
                    "partial_result": item.get("partial_result", False),
                    "available_symbols": ",".join(str(v) for v in item.get("available_symbols", [])),
                    "missing_symbols": ",".join(str(v) for v in item.get("missing_symbols", [])),
                    "trades": summary.get("trades", 0),
                    "total_pnl_net": summary.get("total_pnl_net", 0.0),
                    "max_drawdown_pct_peak": summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0)),
                    "max_drawdown_pct_initial": summary.get("max_drawdown_pct_initial", 0.0),
                    "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
                    "expectancy_net": summary.get("expectancy_net", summary.get("expectancy", 0.0)),
                    "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
                    "constraint_dd_cap_pass_peak": summary.get("constraint_dd_cap_pass_peak", False),
                    "constraint_dd_cap_pass_initial": summary.get("constraint_dd_cap_pass_initial", False),
                    "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                    "oos_pass": summary.get("oos_pass", False),
                    "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                }
            )


def _write_research_winner_config(
    *,
    root: Path,
    base_config: AppConfig,
    source_config_path: Path,
    research_dir: Path,
    best_candidate: dict[str, object] | None,
) -> Path | None:
    if not isinstance(best_candidate, dict):
        return None
    winner_mode = str(best_candidate.get("gate_mode", "")).strip().lower()
    if winner_mode not in {"off", "trend", "trend_vol_news"}:
        return None

    winner_config = base_config.model_copy(deep=True)
    winner_config.daily_gate.mode = winner_mode

    target_path = root / "configs" / "variants" / "config.variant_RESEARCH_WINNER.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "# Auto-generated by --research-run",
        f"# generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"# source_config: {source_config_path}",
        f"# research_report_dir: {research_dir}",
        f"# winner_gate_mode: {winner_mode}",
        "",
    ]
    payload = winner_config.model_dump()
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    target_path.write_text("\n".join(header_lines) + body, encoding="utf-8")
    return target_path


def run_research_mode(args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path) -> None:
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--research-run requires --backtest-start and --backtest-end")

    symbols = parse_epics_csv(args.research_symbols) if args.research_symbols else list(config.research.symbols)
    if not symbols:
        symbols = _backtest_symbols(args, assets)
    workers = max(1, min(3, int(config.research.max_workers)))
    objective_mode = str(config.research.objective_mode).strip().lower()
    dd_cap_pct = float(config.research.dd_cap_pct)
    dd_cap_basis = str(config.research.dd_cap_basis).strip().lower()
    min_trades_oos = int(config.research.min_trades_oos)

    run_stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    research_dir = _resolve_runtime_path(root, f"reports/research/{run_stamp}")
    config_path = _resolve_config_path(root, str(args.config))
    candidate_modes = ["off", "trend", "trend_vol_news"]

    LOGGER.info(
        "Research run started | objective=%s dd_cap_pct=%.2f dd_cap_basis=%s min_trades_oos=%d symbols=%s workers=%d",
        objective_mode,
        dd_cap_pct,
        dd_cap_basis,
        min_trades_oos,
        ",".join(symbols),
        workers,
    )

    futures = []
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for mode in candidate_modes:
            candidate_dir = research_dir / mode
            futures.append(
                executor.submit(
                    _run_single_research_candidate,
                    root=root,
                    args=args,
                    config=config,
                    config_path=config_path,
                    symbols=symbols,
                    gate_mode=mode,
                    run_dir=candidate_dir,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            summary = result.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            LOGGER.info(
                "Research candidate done | mode=%s rc=%s partial=%s reports=%s pnl_net=%.4f dd_peak=%.4f dd_initial=%.4f objective=%.4f",
                result.get("gate_mode"),
                result.get("returncode"),
                bool(result.get("partial_result", False)),
                int(summary.get("reports_count", 0)),
                float(summary.get("total_pnl_net", 0.0)),
                float(summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0))),
                float(summary.get("max_drawdown_pct_initial", 0.0)),
                float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            )
            if bool(result.get("partial_result", False)):
                LOGGER.warning(
                    "Research partial result | mode=%s available_symbols=%s missing_symbols=%s",
                    result.get("gate_mode"),
                    ",".join(str(item) for item in result.get("available_symbols", [])),
                    ",".join(str(item) for item in result.get("missing_symbols", [])),
                )

    results.sort(key=lambda item: objective_rank_key(item.get("summary", {})))
    best = results[0] if results else None
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "objective_mode": objective_mode,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "min_trades_oos": min_trades_oos,
        "symbols": symbols,
        "max_workers": workers,
        "seed": int(config.research.seed),
        "config": str(config_path),
        "backtest": {
            "start": str(args.backtest_start),
            "end": str(args.backtest_end),
            "timeframe": str(args.backtest_tf),
            "price": str(args.backtest_price),
            "variants": str(args.backtest_variants),
            "initial_equity": float(args.initial_equity if args.initial_equity is not None else config.risk.equity),
        },
        "candidates": results,
        "best": best,
    }
    _write_research_summary_files(research_dir, payload)
    winner_config_path = _write_research_winner_config(
        root=root,
        base_config=config,
        source_config_path=config_path,
        research_dir=research_dir,
        best_candidate=best,
    )

    LOGGER.info("Research summary saved: %s", research_dir)
    if winner_config_path is not None:
        LOGGER.info("Research winner config saved: %s", winner_config_path)
    LOGGER.info("Research ranking (best->worst):")
    LOGGER.info("rank | mode | pnl_net | max_dd_peak | max_dd_initial | objective | dd_cap_pass | oos_pass | trades")
    for idx, item in enumerate(results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        LOGGER.info(
            "%d | %s | %.4f | %.4f | %.4f | %.4f | %s | %s | %s",
            idx,
            item.get("gate_mode"),
            float(summary.get("total_pnl_net", 0.0)),
            float(summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0))),
            float(summary.get("max_drawdown_pct_initial", 0.0)),
            float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            bool(summary.get("constraint_dd_cap_pass", False)),
            bool(summary.get("oos_pass", False)),
            int(summary.get("trades", 0)),
        )
