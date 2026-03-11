"""Research optimizer mode — multi-stage hyperparameter search — extracted from app_main.py."""

from __future__ import annotations

import argparse
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
    _capital_dir_name,
    _capital_file_tag,
    _resolve_config_path,
    _resolve_optimizer_capitals,
    _resolve_runtime_path,
    _validate_optimizer_capitals,
    parse_epics_csv,
)
from bot.app.report_helpers import (
    _backtest_symbols,
    _empty_aggregate_summary,
    _extract_source_reports_from_report_dir,
    _write_csv,
)
from bot.config import AppConfig, AssetConfig
from bot.research.objective import OBJECTIVE_FAIL_VALUE, aggregate_reports
from bot.research.optimizer import (
    build_search_space_payload,
    build_stage_a_gate_candidates,
    build_stage_b_candidates,
    build_stage_b_summary,
    build_time_split,
    evaluate_quality_filter,
    failed_stage_summary,
    get_checkpoint_record,
    load_checkpoint,
    normalize_runtime_budget,
    optimizer_rank_key,
    save_checkpoint,
    upsert_checkpoint_record,
)

LOGGER = logging.getLogger("trading_bot")


def _build_optimizer_candidate_config(
    *,
    base_config: AppConfig,
    gate_mode: str,
    gate_params: dict[str, object],
    risk_profile: dict[str, object] | None,
    capital_equity: float | None = None,
    capital_currency: str | None = None,
) -> dict[str, object]:
    payload = base_config.model_dump()

    daily_gate = payload.setdefault("daily_gate", {})
    if not isinstance(daily_gate, dict):
        daily_gate = {}
        payload["daily_gate"] = daily_gate
    daily_gate["mode"] = str(gate_mode).strip().lower()
    if "thr" in gate_params:
        daily_gate["thr"] = float(gate_params["thr"])
    if "pre_minutes" in gate_params:
        daily_gate["pre_minutes"] = int(gate_params["pre_minutes"])
    if "post_minutes" in gate_params:
        daily_gate["post_minutes"] = int(gate_params["post_minutes"])
    if "vol_max" in gate_params:
        daily_gate["vol_max"] = float(gate_params["vol_max"])
    if "max_spread" in gate_params:
        daily_gate["max_spread"] = float(gate_params["max_spread"])
    if "max_spread_mult" in gate_params:
        base_spread = daily_gate.get("max_spread", base_config.daily_gate.max_spread)
        if base_spread is not None:
            daily_gate["max_spread"] = float(base_spread) * float(gate_params["max_spread_mult"])

    if risk_profile:
        risk = payload.setdefault("risk", {})
        if not isinstance(risk, dict):
            risk = {}
            payload["risk"] = risk
        if "risk_per_trade" in risk_profile:
            risk["risk_per_trade"] = float(risk_profile["risk_per_trade"])
        if "max_trades_per_day" in risk_profile:
            risk["max_trades_per_day"] = int(risk_profile["max_trades_per_day"])
        if "max_total_risk_pct" in risk_profile:
            risk["max_total_risk_pct"] = float(risk_profile["max_total_risk_pct"])
        if "daily_stop_pct" in risk_profile:
            risk["daily_stop_pct"] = float(risk_profile["daily_stop_pct"])

    risk_payload = payload.setdefault("risk", {})
    if not isinstance(risk_payload, dict):
        risk_payload = {}
        payload["risk"] = risk_payload
    if capital_equity is not None:
        risk_payload["equity"] = float(capital_equity)
    if capital_currency is not None and str(capital_currency).strip():
        payload["account_currency"] = str(capital_currency).strip().upper()

    return payload


def _build_backtest_subprocess_command_for_optimizer(
    *,
    root: Path,
    args: argparse.Namespace,
    config_path: Path,
    symbols: list[str],
    start: str,
    end: str,
    run_report_dir: Path,
    run_auto_reports_dir: Path,
    initial_equity_override: float | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "main.py"),
        "--backtest",
        "--backtest-symbols",
        ",".join(symbols),
        "--backtest-start",
        str(start),
        "--backtest-end",
        str(end),
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
    initial_equity = initial_equity_override if initial_equity_override is not None else args.initial_equity
    if initial_equity is not None:
        command.extend(["--initial-equity", str(initial_equity)])
    return command


def _run_optimizer_window_candidate(
    *,
    root: Path,
    args: argparse.Namespace,
    config: AppConfig,
    config_path: Path,
    symbols: list[str],
    start: str,
    end: str,
    run_dir: Path,
    dd_cap_pct: float,
    dd_cap_basis: str,
    min_trades_oos: int,
    objective_mode: str,
    seed: int,
    initial_equity_override: float | None = None,
) -> dict[str, object]:
    report_dir = run_dir / "detailed_reports"
    auto_reports_dir = run_dir / "auto_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    auto_reports_dir.mkdir(parents=True, exist_ok=True)

    command = _build_backtest_subprocess_command_for_optimizer(
        root=root,
        args=args,
        config_path=config_path,
        symbols=symbols,
        start=start,
        end=end,
        run_report_dir=report_dir,
        run_auto_reports_dir=auto_reports_dir,
        initial_equity_override=initial_equity_override,
    )

    env = os.environ.copy()
    if config.backtest_runtime.deterministic:
        env["PYTHONHASHSEED"] = str(seed)

    started_at = time.time()
    result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
    elapsed_seconds = round(time.time() - started_at, 3)

    source_reports = _extract_source_reports_from_report_dir(report_dir)
    if result.returncode != 0 and not source_reports:
        # One retry for transient failures.
        retry = subprocess.run(command, cwd=str(root), capture_output=True, text=True, env=env)
        elapsed_seconds = round(time.time() - started_at, 3)
        result = retry
        source_reports = _extract_source_reports_from_report_dir(report_dir)

    if source_reports:
        summary = aggregate_reports(
            source_reports,
            initial_equity=float(config.risk.equity),
            dd_cap_pct=dd_cap_pct,
            dd_cap_basis=dd_cap_basis,
            min_trades_oos=min_trades_oos,
            objective_mode=objective_mode,
        )
    else:
        summary = _empty_aggregate_summary()

    return {
        "command": command,
        "returncode": int(result.returncode),
        "elapsed_seconds": elapsed_seconds,
        "stdout_tail": "\n".join((result.stdout or "").splitlines()[-20:]),
        "stderr_tail": "\n".join((result.stderr or "").splitlines()[-20:]),
        "report_dir": str(report_dir),
        "auto_reports_dir": str(auto_reports_dir),
        "summary": summary,
    }


def _find_resumable_research_opt_dir(base_dir: Path, *, resume_key: dict[str, object]) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted((path for path in base_dir.iterdir() if path.is_dir()), reverse=True)
    for path in candidates:
        checkpoint_path = path / "checkpoint.json"
        if not checkpoint_path.exists():
            continue
        checkpoint = load_checkpoint(checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if bool(metadata.get("completed", False)):
            continue
        key = metadata.get("resume_key")
        if isinstance(key, dict) and key == resume_key:
            return path
    return None


def _find_resumable_research_opt_root_dir(base_dir: Path, *, resume_key: dict[str, object]) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = sorted((path for path in base_dir.iterdir() if path.is_dir()), reverse=True)
    for path in candidates:
        meta_path = path / "dual_capital_meta.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if bool(payload.get("completed", False)):
            continue
        key = payload.get("resume_key")
        if isinstance(key, dict) and key == resume_key:
            return path
    return None


def _is_qualified_optimizer_winner(summary: dict[str, object]) -> tuple[bool, str]:
    quality_pass = bool(summary.get("quality_pass", False))
    if not quality_pass:
        return False, "QUALITY_FILTER_FAILED"
    if not bool(summary.get("oos_pass", False)):
        return False, "OOS_FAIL"
    if not bool(summary.get("constraint_dd_cap_pass_peak", False)):
        return False, "DD_CAP_PEAK_FAIL"
    if not bool(summary.get("constraint_dd_cap_pass_initial", False)):
        return False, "DD_CAP_INITIAL_FAIL"
    if not bool(summary.get("constraint_dd_cap_pass", False)):
        return False, "DD_CAP_FAIL"
    if float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)) <= OBJECTIVE_FAIL_VALUE:
        return False, "OBJECTIVE_FAIL_VALUE"
    return True, "OK"


def _summarize_stage_progress(
    *,
    stage: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed = max(1e-9, time.time() - started_at)
    throughput_per_min = (completed / elapsed) * 60.0
    remaining = max(0, total - completed)
    eta_minutes = (remaining / throughput_per_min) if throughput_per_min > 1e-9 else 0.0
    LOGGER.info(
        "Research optimize progress | stage=%s done=%d/%d throughput=%.2f cand/min eta=%.2f min",
        stage,
        completed,
        total,
        throughput_per_min,
        eta_minutes,
    )


def _write_optimizer_best_config(
    *,
    root: Path,
    base_config: AppConfig,
    source_config_path: Path,
    report_dir: Path,
    best_record: dict[str, object] | None,
    split_payload: dict[str, object],
    objective_mode: str,
    dd_cap_pct: float,
    dd_cap_basis: str,
    output_filename: str = "config.variant_RESEARCH_OPT_BEST.yaml",
    capital_equity: float | None = None,
    capital_currency: str | None = None,
    quality_filter_mode: str | None = None,
) -> Path | None:
    if not isinstance(best_record, dict):
        return None
    gate_mode = str(best_record.get("gate_mode", "")).strip().lower()
    if gate_mode not in {"off", "trend", "trend_vol_news"}:
        return None
    gate_params = best_record.get("gate_params", {})
    if not isinstance(gate_params, dict):
        gate_params = {}
    risk_profile = best_record.get("risk_profile", {})
    if not isinstance(risk_profile, dict):
        risk_profile = {}

    winner_payload = _build_optimizer_candidate_config(
        base_config=base_config,
        gate_mode=gate_mode,
        gate_params=gate_params,
        risk_profile=risk_profile,
        capital_equity=capital_equity,
        capital_currency=capital_currency,
    )

    target_path = root / "configs" / "variants" / str(output_filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated by --research-optimize",
        f"# generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"# source_config: {source_config_path}",
        f"# research_opt_report_dir: {report_dir}",
        f"# objective_mode: {objective_mode}",
        f"# dd_cap_pct: {dd_cap_pct}",
        f"# dd_cap_basis: {dd_cap_basis}",
        f"# split: {json.dumps(split_payload, ensure_ascii=True)}",
    ]
    if capital_equity is not None:
        header.append(f"# capital_equity: {capital_equity}")
    if capital_currency is not None:
        header.append(f"# capital_currency: {capital_currency}")
    if quality_filter_mode is not None:
        header.append(f"# quality_filter_mode: {quality_filter_mode}")
    header.append("")
    body = yaml.safe_dump(winner_payload, sort_keys=False, allow_unicode=False)
    target_path.write_text("\n".join(header) + body, encoding="utf-8")
    return target_path


def _run_research_optimize_mode_single_capital_legacy(
    args: argparse.Namespace,
    config: AppConfig,
    assets: list[AssetConfig],
    root: Path,
) -> None:
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--research-optimize requires --backtest-start and --backtest-end")

    optimize_cfg = config.research.optimize
    runtime_budget = normalize_runtime_budget(
        args.research_runtime_budget if args.research_runtime_budget is not None else optimize_cfg.runtime_budget
    )
    symbols = (
        parse_epics_csv(args.research_benchmark_symbols)
        if args.research_benchmark_symbols
        else list(config.research.symbols)
    )
    if not symbols:
        symbols = _backtest_symbols(args, assets)
    if not symbols:
        raise RuntimeError("No symbols available for --research-optimize")

    workers = max(1, min(3, int(optimize_cfg.max_workers)))
    dd_cap_pct = float(config.research.dd_cap_pct)
    dd_cap_basis = str(optimize_cfg.dd_cap_basis or config.research.dd_cap_basis).strip().lower()
    min_trades_oos = int(config.research.min_trades_oos)
    objective_mode = str(optimize_cfg.objective_mode).strip().lower()
    seed = int(optimize_cfg.seed)
    config_path = _resolve_config_path(root, str(args.config))

    split = build_time_split(
        backtest_start=str(args.backtest_start),
        backtest_end=str(args.backtest_end),
        split_ratio_is=float(optimize_cfg.split_ratio_is),
        min_days_is=int(optimize_cfg.min_days_is),
        min_days_oos=int(optimize_cfg.min_days_oos),
    )
    split_payload = split.to_dict()

    search_space_payload = config.research.search_space.model_dump()
    gate_space = dict(search_space_payload.get("gate", {}))
    risk_profiles = list(search_space_payload.get("risk_profiles", []))
    stage_a_candidates = build_stage_a_gate_candidates(
        search_space_gate=gate_space,
        runtime_budget=runtime_budget,
    )
    top_gate_keep = min(int(optimize_cfg.top_gate_keep), len(stage_a_candidates))
    top_final_keep = max(1, int(optimize_cfg.top_final_keep))

    resume_key = {
        "config": str(config_path),
        "symbols": symbols,
        "backtest_start": str(args.backtest_start),
        "backtest_end": str(args.backtest_end),
        "timeframe": str(args.backtest_tf),
        "price": str(args.backtest_price),
        "runtime_budget": runtime_budget,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "objective_mode": objective_mode,
    }
    base_dir = _resolve_runtime_path(root, "reports/research_opt")
    resumable = _find_resumable_research_opt_dir(base_dir, resume_key=resume_key)
    if resumable is not None:
        run_dir = resumable
        resumed = True
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / stamp
        resumed = False
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_meta = checkpoint.setdefault("metadata", {})
    if not isinstance(checkpoint_meta, dict):
        checkpoint_meta = {}
        checkpoint["metadata"] = checkpoint_meta
    checkpoint_meta["resume_key"] = resume_key
    checkpoint_meta["runtime_budget"] = runtime_budget
    checkpoint_meta["objective_mode"] = objective_mode
    checkpoint_meta["dd_cap_pct"] = dd_cap_pct
    checkpoint_meta["dd_cap_basis"] = dd_cap_basis
    checkpoint_meta["completed"] = False
    save_checkpoint(checkpoint_path, checkpoint)

    search_space_out = build_search_space_payload(
        runtime_budget=runtime_budget,
        gate_space=gate_space,
        risk_profiles=risk_profiles,
        stage_a_candidates=stage_a_candidates,
    )
    (run_dir / "search_space.json").write_text(
        json.dumps(search_space_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (run_dir / "split_info.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    LOGGER.info(
        "Research optimize started | resumed=%s budget=%s symbols=%s workers=%d stageA=%d",
        resumed,
        runtime_budget,
        ",".join(symbols),
        workers,
        len(stage_a_candidates),
    )

    stage_a_results: list[dict[str, object]] = []
    stage_a_started_at = time.time()
    stage_a_total = len(stage_a_candidates)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[object, tuple[dict[str, object], Path]] = {}
        for candidate in stage_a_candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            cached = get_checkpoint_record(checkpoint, stage="A", candidate_id=candidate_id)
            if cached is not None and cached.get("status") == "done":
                stage_a_results.append(cached)
                continue
            candidate_dir = run_dir / "stage_a" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_config_payload = _build_optimizer_candidate_config(
                base_config=config,
                gate_mode=str(candidate.get("gate_mode", "off")),
                gate_params=dict(candidate.get("gate_params", {})),
                risk_profile=None,
            )
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            candidate_config_path.write_text(
                yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            future = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.is_start,
                end=split.is_end,
                run_dir=candidate_dir / "is",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future] = (candidate, candidate_config_path)

        completed = len(stage_a_results)
        if completed:
            _summarize_stage_progress(
                stage="A", completed=completed, total=stage_a_total, started_at=stage_a_started_at
            )
        for future in as_completed(futures):
            candidate, candidate_config_path = futures[future]
            result = future.result()
            record = {
                "status": "done",
                "stage": "A",
                "candidate_id": candidate.get("candidate_id"),
                "gate_mode": candidate.get("gate_mode"),
                "gate_params": candidate.get("gate_params", {}),
                "risk_profile": None,
                "config_path": str(candidate_config_path),
                "returncode": result.get("returncode"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "summary": result.get("summary", _empty_aggregate_summary()),
                "stdout_tail": result.get("stdout_tail", ""),
                "stderr_tail": result.get("stderr_tail", ""),
                "report_dir": result.get("report_dir"),
            }
            stage_a_results.append(record)
            upsert_checkpoint_record(
                checkpoint, stage="A", candidate_id=str(record.get("candidate_id", "")), record=record
            )
            save_checkpoint(checkpoint_path, checkpoint)
            completed += 1
            if completed % 5 == 0 or completed == stage_a_total:
                _summarize_stage_progress(
                    stage="A", completed=completed, total=stage_a_total, started_at=stage_a_started_at
                )

    stage_a_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", {})))
    top_gate_candidates = [
        {
            "candidate_id": item.get("candidate_id"),
            "gate_mode": item.get("gate_mode"),
            "gate_params": item.get("gate_params", {}),
        }
        for item in stage_a_results[:top_gate_keep]
    ]

    stage_a_csv_rows: list[dict[str, object]] = []
    for rank, item in enumerate(stage_a_results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        stage_a_csv_rows.append(
            {
                "rank": rank,
                "candidate_id": item.get("candidate_id"),
                "gate_mode": item.get("gate_mode"),
                "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                "returncode": item.get("returncode"),
                "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                "trades": summary.get("trades", 0),
                "total_pnl_net": summary.get("total_pnl_net", 0.0),
                "max_drawdown_pct_peak": summary.get("max_drawdown_pct_peak", summary.get("max_drawdown_pct", 0.0)),
                "max_drawdown_pct_initial": summary.get("max_drawdown_pct_initial", 0.0),
                "dd_ref_pct": summary.get("dd_ref_pct", 0.0),
                "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                "oos_pass": summary.get("oos_pass", False),
                "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
            }
        )
    _write_csv(
        run_dir / "stage_a_gate_is.csv",
        fieldnames=[
            "rank",
            "candidate_id",
            "gate_mode",
            "gate_params",
            "returncode",
            "elapsed_seconds",
            "trades",
            "total_pnl_net",
            "max_drawdown_pct_peak",
            "max_drawdown_pct_initial",
            "dd_ref_pct",
            "constraint_dd_cap_pass",
            "oos_pass",
            "objective_value",
        ],
        rows=stage_a_csv_rows,
    )

    stage_b_candidates = build_stage_b_candidates(
        top_gate_candidates=top_gate_candidates,
        risk_profiles=risk_profiles,
        runtime_budget=runtime_budget,
    )
    stage_b_total = len(stage_b_candidates)
    used_risk_profiles = len(
        {str(item.get("risk_profile", {}).get("name", "")) for item in stage_b_candidates if isinstance(item, dict)}
    )
    LOGGER.info(
        "Research optimize stage B | top_gate_keep=%d risk_profiles=%d candidates=%d",
        len(top_gate_candidates),
        used_risk_profiles,
        stage_b_total,
    )

    stage_b_results: list[dict[str, object]] = []
    stage_b_started_at = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[object, tuple[str, dict[str, object], Path]] = {}
        for candidate in stage_b_candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            cached = get_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id)
            if cached is not None and cached.get("status") == "done":
                stage_b_results.append(cached)
                continue

            candidate_dir = run_dir / "stage_b" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_config_payload = _build_optimizer_candidate_config(
                base_config=config,
                gate_mode=str(candidate.get("gate_mode", "off")),
                gate_params=dict(candidate.get("gate_params", {})),
                risk_profile=dict(candidate.get("risk_profile", {})),
            )
            candidate_config_path = candidate_dir / "candidate_config.yaml"
            candidate_config_path.write_text(
                yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )

            future_is = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.is_start,
                end=split.is_end,
                run_dir=candidate_dir / "is",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future_is] = ("is", candidate, candidate_config_path)

            future_oos = executor.submit(
                _run_optimizer_window_candidate,
                root=root,
                args=args,
                config=config,
                config_path=candidate_config_path,
                symbols=symbols,
                start=split.oos_start,
                end=split.oos_end,
                run_dir=candidate_dir / "oos",
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                min_trades_oos=min_trades_oos,
                objective_mode=objective_mode,
                seed=seed,
            )
            futures[future_oos] = ("oos", candidate, candidate_config_path)

        interim: dict[str, dict[str, object]] = {}
        completed = len(stage_b_results)
        if completed:
            _summarize_stage_progress(
                stage="B", completed=completed, total=stage_b_total, started_at=stage_b_started_at
            )

        for future in as_completed(futures):
            window_name, candidate, candidate_config_path = futures[future]
            result = future.result()
            candidate_id = str(candidate.get("candidate_id", ""))
            bucket = interim.setdefault(candidate_id, {"candidate": candidate, "config_path": candidate_config_path})
            bucket[window_name] = result
            if "is" in bucket and "oos" in bucket:
                is_summary = bucket["is"].get("summary", _empty_aggregate_summary())
                oos_summary = bucket["oos"].get("summary", _empty_aggregate_summary())
                if not isinstance(is_summary, dict):
                    is_summary = _empty_aggregate_summary()
                if not isinstance(oos_summary, dict):
                    oos_summary = _empty_aggregate_summary()
                summary = build_stage_b_summary(
                    is_summary=is_summary,
                    oos_summary=oos_summary,
                    dd_cap_pct=dd_cap_pct,
                    dd_cap_basis=dd_cap_basis,
                    min_trades_oos=min_trades_oos,
                    objective_mode=objective_mode,
                )
                combined = {
                    "status": "done",
                    "stage": "B",
                    "candidate_id": candidate.get("candidate_id"),
                    "gate_candidate_id": candidate.get("gate_candidate_id"),
                    "gate_mode": candidate.get("gate_mode"),
                    "gate_params": candidate.get("gate_params", {}),
                    "risk_profile": candidate.get("risk_profile", {}),
                    "risk_profile_name": dict(candidate.get("risk_profile", {})).get("name", ""),
                    "config_path": str(candidate_config_path),
                    "is_returncode": bucket["is"].get("returncode"),
                    "oos_returncode": bucket["oos"].get("returncode"),
                    "elapsed_seconds": float(bucket["is"].get("elapsed_seconds", 0.0))
                    + float(bucket["oos"].get("elapsed_seconds", 0.0)),
                    "is_summary": is_summary,
                    "oos_summary": oos_summary,
                    "summary": summary,
                    "stderr_tail_is": bucket["is"].get("stderr_tail", ""),
                    "stderr_tail_oos": bucket["oos"].get("stderr_tail", ""),
                }
                stage_b_results.append(combined)
                upsert_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id, record=combined)
                save_checkpoint(checkpoint_path, checkpoint)
                interim.pop(candidate_id, None)
                completed += 1
                if completed % 5 == 0 or completed == stage_b_total:
                    _summarize_stage_progress(
                        stage="B", completed=completed, total=stage_b_total, started_at=stage_b_started_at
                    )

    stage_b_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", failed_stage_summary())))
    top_records = stage_b_results[: min(top_final_keep, len(stage_b_results))]
    best_record = top_records[0] if top_records else None

    stage_b_csv_rows: list[dict[str, object]] = []
    for rank, item in enumerate(stage_b_results, start=1):
        summary = item.get("summary", {})
        if not isinstance(summary, dict):
            summary = failed_stage_summary()
        stage_b_csv_rows.append(
            {
                "rank": rank,
                "candidate_id": item.get("candidate_id"),
                "gate_candidate_id": item.get("gate_candidate_id"),
                "gate_mode": item.get("gate_mode"),
                "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                "risk_profile_name": item.get("risk_profile_name"),
                "risk_profile": json.dumps(item.get("risk_profile", {}), ensure_ascii=True, sort_keys=True),
                "is_returncode": item.get("is_returncode"),
                "oos_returncode": item.get("oos_returncode"),
                "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                "is_total_pnl_net": summary.get("is_total_pnl_net", 0.0),
                "is_dd_ref_pct": summary.get("is_dd_ref_pct", 0.0),
                "oos_total_pnl_net": summary.get("oos_total_pnl_net", 0.0),
                "oos_dd_ref_pct": summary.get("oos_dd_ref_pct", 0.0),
                "oos_expectancy_net": summary.get("oos_expectancy_net", 0.0),
                "oos_trades": summary.get("oos_trades", 0),
                "constraint_dd_cap_pass_peak": summary.get("constraint_dd_cap_pass_peak", False),
                "constraint_dd_cap_pass_initial": summary.get("constraint_dd_cap_pass_initial", False),
                "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                "oos_pass": summary.get("oos_pass", False),
                "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
            }
        )
    _write_csv(
        run_dir / "stage_b_gate_risk_is_oos.csv",
        fieldnames=[
            "rank",
            "candidate_id",
            "gate_candidate_id",
            "gate_mode",
            "gate_params",
            "risk_profile_name",
            "risk_profile",
            "is_returncode",
            "oos_returncode",
            "elapsed_seconds",
            "is_total_pnl_net",
            "is_dd_ref_pct",
            "oos_total_pnl_net",
            "oos_dd_ref_pct",
            "oos_expectancy_net",
            "oos_trades",
            "constraint_dd_cap_pass_peak",
            "constraint_dd_cap_pass_initial",
            "constraint_dd_cap_pass",
            "oos_pass",
            "objective_value",
        ],
        rows=stage_b_csv_rows,
    )

    top20_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "runtime_budget": runtime_budget,
        "objective_mode": objective_mode,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "split": split_payload,
        "top": top_records,
    }
    (run_dir / "top20.json").write_text(json.dumps(top20_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "best.json").write_text(
        json.dumps(
            {"best": best_record, "split": split_payload, "objective_mode": objective_mode}, indent=2, ensure_ascii=True
        ),
        encoding="utf-8",
    )

    winner_path = _write_optimizer_best_config(
        root=root,
        base_config=config,
        source_config_path=config_path,
        report_dir=run_dir,
        best_record=best_record,
        split_payload=split_payload,
        objective_mode=objective_mode,
        dd_cap_pct=dd_cap_pct,
        dd_cap_basis=dd_cap_basis,
    )

    checkpoint["metadata"]["completed"] = True
    checkpoint["metadata"]["stage_a_candidates"] = len(stage_a_candidates)
    checkpoint["metadata"]["stage_b_candidates"] = len(stage_b_candidates)
    save_checkpoint(checkpoint_path, checkpoint)

    LOGGER.info("Research optimize summary saved: %s", run_dir)
    if winner_path is not None:
        LOGGER.info("Research optimize winner config saved: %s", winner_path)
    LOGGER.info("Research optimize ranking (best->worst):")
    LOGGER.info("rank | gate | risk | oos_pnl_net | oos_dd_ref | objective | dd_pass | oos_pass | oos_trades")
    for idx, record in enumerate(top_records[:20], start=1):
        summary = record.get("summary", {})
        if not isinstance(summary, dict):
            summary = failed_stage_summary()
        LOGGER.info(
            "%d | %s | %s | %.4f | %.4f | %.4f | %s | %s | %d",
            idx,
            record.get("gate_mode"),
            record.get("risk_profile_name"),
            float(summary.get("oos_total_pnl_net", 0.0)),
            float(summary.get("oos_dd_ref_pct", 0.0)),
            float(summary.get("objective_value", OBJECTIVE_FAIL_VALUE)),
            bool(summary.get("constraint_dd_cap_pass", False)),
            bool(summary.get("oos_pass", False)),
            int(summary.get("oos_trades", 0)),
        )


# Overrides the earlier single-capital implementation with dual-capital orchestration.
def run_research_optimize_mode(
    args: argparse.Namespace, config: AppConfig, assets: list[AssetConfig], root: Path
) -> None:
    if not args.backtest_start or not args.backtest_end:
        raise RuntimeError("--research-optimize requires --backtest-start and --backtest-end")

    optimize_cfg = config.research.optimize
    runtime_budget = normalize_runtime_budget(
        args.research_runtime_budget if args.research_runtime_budget is not None else optimize_cfg.runtime_budget
    )
    symbols = (
        parse_epics_csv(args.research_benchmark_symbols)
        if args.research_benchmark_symbols
        else list(config.research.symbols)
    )
    if not symbols:
        symbols = _backtest_symbols(args, assets)
    if not symbols:
        raise RuntimeError("No symbols available for --research-optimize")

    workers = max(1, min(3, int(optimize_cfg.max_workers)))
    dd_cap_pct = float(config.research.dd_cap_pct)
    dd_cap_basis = str(optimize_cfg.dd_cap_basis or config.research.dd_cap_basis).strip().lower()
    min_trades_oos = int(config.research.min_trades_oos)
    objective_mode = str(optimize_cfg.objective_mode).strip().lower()
    seed = int(optimize_cfg.seed)
    config_path = _resolve_config_path(root, str(args.config))

    split = build_time_split(
        backtest_start=str(args.backtest_start),
        backtest_end=str(args.backtest_end),
        split_ratio_is=float(optimize_cfg.split_ratio_is),
        min_days_is=int(optimize_cfg.min_days_is),
        min_days_oos=int(optimize_cfg.min_days_oos),
    )
    split_payload = split.to_dict()

    search_space_payload = config.research.search_space.model_dump()
    gate_space = dict(search_space_payload.get("gate", {}))
    risk_profiles = list(search_space_payload.get("risk_profiles", []))
    stage_a_candidates = build_stage_a_gate_candidates(
        search_space_gate=gate_space,
        runtime_budget=runtime_budget,
    )
    top_gate_keep = min(int(optimize_cfg.top_gate_keep), len(stage_a_candidates))
    top_final_keep = max(1, int(optimize_cfg.top_final_keep))

    quality_filter = optimize_cfg.quality_filter.model_dump()
    quality_filter_mode = str(quality_filter.get("mode", "strict")).strip().lower()
    quality_filter_windows = [
        str(item).strip().lower()
        for item in quality_filter.get("apply_windows", ["is", "oos"])
        if str(item).strip().lower() in {"is", "oos"}
    ] or ["is", "oos"]

    capitals = _resolve_optimizer_capitals(config)
    _validate_optimizer_capitals(capitals=capitals, fx_static_rates=config.fx_static_rates)
    capital_run_mode = (
        str(
            args.research_capital_run_mode
            if args.research_capital_run_mode is not None
            else optimize_cfg.capital_run_mode
        )
        .strip()
        .lower()
    )
    if capital_run_mode not in {"sequential", "parallel"}:
        capital_run_mode = "sequential"
    if len(capitals) < 2:
        capital_run_mode = "sequential"

    resume_key = {
        "config": str(config_path),
        "symbols": symbols,
        "backtest_start": str(args.backtest_start),
        "backtest_end": str(args.backtest_end),
        "timeframe": str(args.backtest_tf),
        "price": str(args.backtest_price),
        "runtime_budget": runtime_budget,
        "dd_cap_pct": dd_cap_pct,
        "dd_cap_basis": dd_cap_basis,
        "objective_mode": objective_mode,
        "quality_filter": quality_filter,
        "capitals": capitals,
        "capital_run_mode": capital_run_mode,
    }
    base_dir = _resolve_runtime_path(root, "reports/research_opt")
    resumable = _find_resumable_research_opt_root_dir(base_dir, resume_key=resume_key)
    if resumable is not None:
        run_dir = resumable
        resumed = True
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_dir = base_dir / stamp
        resumed = False
    run_dir.mkdir(parents=True, exist_ok=True)

    search_space_out = build_search_space_payload(
        runtime_budget=runtime_budget,
        gate_space=gate_space,
        risk_profiles=risk_profiles,
        stage_a_candidates=stage_a_candidates,
    )
    (run_dir / "search_space.json").write_text(
        json.dumps(search_space_out, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    (run_dir / "split_info.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    (run_dir / "dual_capital_meta.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "resume_key": resume_key,
                "completed": False,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "Research optimize started | resumed=%s budget=%s symbols=%s workers=%d stageA=%d capitals=%s mode=%s",
        resumed,
        runtime_budget,
        ",".join(symbols),
        workers,
        len(stage_a_candidates),
        ",".join(f"{float(item['equity'])}:{str(item['currency']).upper()}" for item in capitals),
        capital_run_mode,
    )

    def _run_single_capital(capital_spec: dict[str, object]) -> dict[str, object]:
        capital_equity = float(capital_spec["equity"])
        capital_currency = str(capital_spec["currency"]).strip().upper()
        capital_dir = run_dir / _capital_dir_name(capital_equity, capital_currency)
        capital_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = capital_dir / "checkpoint.json"
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint_meta = checkpoint.setdefault("metadata", {})
        if not isinstance(checkpoint_meta, dict):
            checkpoint_meta = {}
            checkpoint["metadata"] = checkpoint_meta
        checkpoint_meta["resume_key"] = resume_key
        checkpoint_meta["capital_equity"] = capital_equity
        checkpoint_meta["capital_currency"] = capital_currency
        checkpoint_meta["completed"] = False
        save_checkpoint(checkpoint_path, checkpoint)
        (capital_dir / "search_space.json").write_text(
            json.dumps(search_space_out, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        (capital_dir / "split_info.json").write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )

        capital_config = config.model_copy(deep=True)
        capital_config.risk.equity = capital_equity
        capital_config.account_currency = capital_currency

        stage_a_results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[object, tuple[dict[str, object], Path]] = {}
            for candidate in stage_a_candidates:
                candidate_id = str(candidate.get("candidate_id", ""))
                cached = get_checkpoint_record(checkpoint, stage="A", candidate_id=candidate_id)
                if cached is not None and cached.get("status") == "done":
                    stage_a_results.append(cached)
                    continue
                candidate_dir = capital_dir / "stage_a" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_config_payload = _build_optimizer_candidate_config(
                    base_config=capital_config,
                    gate_mode=str(candidate.get("gate_mode", "off")),
                    gate_params=dict(candidate.get("gate_params", {})),
                    risk_profile=None,
                    capital_equity=capital_equity,
                    capital_currency=capital_currency,
                )
                candidate_config_path = candidate_dir / "candidate_config.yaml"
                candidate_config_path.write_text(
                    yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False), encoding="utf-8"
                )
                future = executor.submit(
                    _run_optimizer_window_candidate,
                    root=root,
                    args=args,
                    config=capital_config,
                    config_path=candidate_config_path,
                    symbols=symbols,
                    start=split.is_start,
                    end=split.is_end,
                    run_dir=candidate_dir / "is",
                    dd_cap_pct=dd_cap_pct,
                    dd_cap_basis=dd_cap_basis,
                    min_trades_oos=min_trades_oos,
                    objective_mode=objective_mode,
                    seed=seed,
                    initial_equity_override=capital_equity,
                )
                futures[future] = (candidate, candidate_config_path)
            for future in as_completed(futures):
                candidate, candidate_config_path = futures[future]
                result = future.result()
                summary = result.get("summary", _empty_aggregate_summary())
                if not isinstance(summary, dict):
                    summary = _empty_aggregate_summary()
                quality = evaluate_quality_filter(
                    is_summary=summary,
                    oos_summary=None,
                    mode=quality_filter_mode,
                    apply_windows=["is"],
                    blocked_anomaly_flags=list(quality_filter.get("blocked_anomaly_flags", [])),
                    min_is_trades=int(quality_filter.get("min_is_trades", min_trades_oos)),
                    min_oos_trades=int(quality_filter.get("min_oos_trades", min_trades_oos)),
                    require_orders_submitted=bool(quality_filter.get("require_orders_submitted", True)),
                    require_trades_filled=bool(quality_filter.get("require_trades_filled", True)),
                )
                summary.update(quality)
                summary["capital_equity"] = capital_equity
                summary["capital_currency"] = capital_currency
                if not bool(summary.get("quality_pass", False)):
                    summary["objective_value"] = OBJECTIVE_FAIL_VALUE
                record = {
                    "status": "done",
                    "stage": "A",
                    "candidate_id": candidate.get("candidate_id"),
                    "gate_mode": candidate.get("gate_mode"),
                    "gate_params": candidate.get("gate_params", {}),
                    "config_path": str(candidate_config_path),
                    "returncode": result.get("returncode"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "summary": summary,
                }
                stage_a_results.append(record)
                upsert_checkpoint_record(
                    checkpoint, stage="A", candidate_id=str(record.get("candidate_id", "")), record=record
                )
                save_checkpoint(checkpoint_path, checkpoint)

        stage_a_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", {})))
        stage_a_csv_rows: list[dict[str, object]] = []
        for rank, item in enumerate(stage_a_results, start=1):
            summary = item.get("summary", {})
            if not isinstance(summary, dict):
                summary = _empty_aggregate_summary()
            stage_a_csv_rows.append(
                {
                    "rank": rank,
                    "capital_equity": capital_equity,
                    "capital_currency": capital_currency,
                    "candidate_id": item.get("candidate_id"),
                    "gate_mode": item.get("gate_mode"),
                    "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                    "returncode": item.get("returncode"),
                    "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                    "is_total_pnl_net": summary.get("total_pnl_net", 0.0),
                    "is_dd_ref_pct": summary.get("dd_ref_pct", 0.0),
                    "is_expectancy_net": summary.get("expectancy_net", summary.get("expectancy", 0.0)),
                    "is_trades": summary.get("trades", 0),
                    "quality_pass": summary.get("quality_pass", False),
                    "quality_reasons": "|".join(summary.get("quality_reasons", [])),
                    "anomaly_flags_is": "|".join(summary.get("anomaly_flags_is", summary.get("anomaly_flags", []))),
                    "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
                }
            )
        _write_csv(
            capital_dir / "stage_a_gate_is.csv",
            fieldnames=list(stage_a_csv_rows[0].keys()) if stage_a_csv_rows else ["rank"],
            rows=stage_a_csv_rows,
        )
        top_gate_candidates = [
            {
                "candidate_id": item.get("candidate_id"),
                "gate_mode": item.get("gate_mode"),
                "gate_params": item.get("gate_params", {}),
            }
            for item in stage_a_results[:top_gate_keep]
        ]
        stage_b_candidates = build_stage_b_candidates(
            top_gate_candidates=top_gate_candidates, risk_profiles=risk_profiles, runtime_budget=runtime_budget
        )

        stage_b_results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[object, tuple[str, dict[str, object], Path]] = {}
            for candidate in stage_b_candidates:
                candidate_id = str(candidate.get("candidate_id", ""))
                cached = get_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id)
                if cached is not None and cached.get("status") == "done":
                    stage_b_results.append(cached)
                    continue
                candidate_dir = capital_dir / "stage_b" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                candidate_config_payload = _build_optimizer_candidate_config(
                    base_config=capital_config,
                    gate_mode=str(candidate.get("gate_mode", "off")),
                    gate_params=dict(candidate.get("gate_params", {})),
                    risk_profile=dict(candidate.get("risk_profile", {})),
                    capital_equity=capital_equity,
                    capital_currency=capital_currency,
                )
                candidate_config_path = candidate_dir / "candidate_config.yaml"
                candidate_config_path.write_text(
                    yaml.safe_dump(candidate_config_payload, sort_keys=False, allow_unicode=False), encoding="utf-8"
                )
                for window_name, start, end in (
                    ("is", split.is_start, split.is_end),
                    ("oos", split.oos_start, split.oos_end),
                ):
                    future = executor.submit(
                        _run_optimizer_window_candidate,
                        root=root,
                        args=args,
                        config=capital_config,
                        config_path=candidate_config_path,
                        symbols=symbols,
                        start=start,
                        end=end,
                        run_dir=candidate_dir / window_name,
                        dd_cap_pct=dd_cap_pct,
                        dd_cap_basis=dd_cap_basis,
                        min_trades_oos=min_trades_oos,
                        objective_mode=objective_mode,
                        seed=seed,
                        initial_equity_override=capital_equity,
                    )
                    futures[future] = (window_name, candidate, candidate_config_path)
            interim: dict[str, dict[str, object]] = {}
            for future in as_completed(futures):
                window_name, candidate, candidate_config_path = futures[future]
                result = future.result()
                candidate_id = str(candidate.get("candidate_id", ""))
                bucket = interim.setdefault(
                    candidate_id, {"candidate": candidate, "config_path": candidate_config_path}
                )
                bucket[window_name] = result
                if "is" in bucket and "oos" in bucket:
                    is_summary = bucket["is"].get("summary", _empty_aggregate_summary())
                    oos_summary = bucket["oos"].get("summary", _empty_aggregate_summary())
                    if not isinstance(is_summary, dict):
                        is_summary = _empty_aggregate_summary()
                    if not isinstance(oos_summary, dict):
                        oos_summary = _empty_aggregate_summary()
                    summary = build_stage_b_summary(
                        is_summary=is_summary,
                        oos_summary=oos_summary,
                        dd_cap_pct=dd_cap_pct,
                        dd_cap_basis=dd_cap_basis,
                        min_trades_oos=min_trades_oos,
                        objective_mode=objective_mode,
                    )
                    quality = evaluate_quality_filter(
                        is_summary=is_summary,
                        oos_summary=oos_summary,
                        mode=quality_filter_mode,
                        apply_windows=quality_filter_windows,
                        blocked_anomaly_flags=list(quality_filter.get("blocked_anomaly_flags", [])),
                        min_is_trades=int(quality_filter.get("min_is_trades", min_trades_oos)),
                        min_oos_trades=int(quality_filter.get("min_oos_trades", min_trades_oos)),
                        require_orders_submitted=bool(quality_filter.get("require_orders_submitted", True)),
                        require_trades_filled=bool(quality_filter.get("require_trades_filled", True)),
                    )
                    summary.update(quality)
                    summary["capital_equity"] = capital_equity
                    summary["capital_currency"] = capital_currency
                    if not bool(summary.get("quality_pass", False)):
                        summary["objective_value"] = OBJECTIVE_FAIL_VALUE
                    combined = {
                        "status": "done",
                        "stage": "B",
                        "candidate_id": candidate.get("candidate_id"),
                        "gate_candidate_id": candidate.get("gate_candidate_id"),
                        "gate_mode": candidate.get("gate_mode"),
                        "gate_params": candidate.get("gate_params", {}),
                        "risk_profile": candidate.get("risk_profile", {}),
                        "risk_profile_name": dict(candidate.get("risk_profile", {})).get("name", ""),
                        "config_path": str(candidate_config_path),
                        "is_returncode": bucket["is"].get("returncode"),
                        "oos_returncode": bucket["oos"].get("returncode"),
                        "elapsed_seconds": float(bucket["is"].get("elapsed_seconds", 0.0))
                        + float(bucket["oos"].get("elapsed_seconds", 0.0)),
                        "is_summary": is_summary,
                        "oos_summary": oos_summary,
                        "summary": summary,
                    }
                    stage_b_results.append(combined)
                    upsert_checkpoint_record(checkpoint, stage="B", candidate_id=candidate_id, record=combined)
                    save_checkpoint(checkpoint_path, checkpoint)
                    interim.pop(candidate_id, None)

        stage_b_results.sort(key=lambda item: optimizer_rank_key(item.get("summary", failed_stage_summary())))
        top_records = stage_b_results[: min(top_final_keep, len(stage_b_results))]
        best_record = top_records[0] if top_records else None
        stage_b_csv_rows: list[dict[str, object]] = []
        for rank, item in enumerate(stage_b_results, start=1):
            summary = item.get("summary", {})
            if not isinstance(summary, dict):
                summary = failed_stage_summary()
            stage_b_csv_rows.append(
                {
                    "rank": rank,
                    "capital_equity": capital_equity,
                    "capital_currency": capital_currency,
                    "candidate_id": item.get("candidate_id"),
                    "gate_candidate_id": item.get("gate_candidate_id"),
                    "gate_mode": item.get("gate_mode"),
                    "gate_params": json.dumps(item.get("gate_params", {}), ensure_ascii=True, sort_keys=True),
                    "risk_profile_name": item.get("risk_profile_name"),
                    "risk_profile": json.dumps(item.get("risk_profile", {}), ensure_ascii=True, sort_keys=True),
                    "is_returncode": item.get("is_returncode"),
                    "oos_returncode": item.get("oos_returncode"),
                    "elapsed_seconds": item.get("elapsed_seconds", 0.0),
                    "oos_total_pnl_net": summary.get("oos_total_pnl_net", 0.0),
                    "oos_dd_ref_pct": summary.get("oos_dd_ref_pct", 0.0),
                    "oos_expectancy_net": summary.get("oos_expectancy_net", 0.0),
                    "oos_trades": summary.get("oos_trades", 0),
                    "quality_pass": summary.get("quality_pass", False),
                    "quality_reasons": "|".join(summary.get("quality_reasons", [])),
                    "anomaly_flags_is": "|".join(summary.get("anomaly_flags_is", [])),
                    "anomaly_flags_oos": "|".join(summary.get("anomaly_flags_oos", [])),
                    "constraint_dd_cap_pass_peak": summary.get("constraint_dd_cap_pass_peak", False),
                    "constraint_dd_cap_pass_initial": summary.get("constraint_dd_cap_pass_initial", False),
                    "constraint_dd_cap_pass": summary.get("constraint_dd_cap_pass", False),
                    "oos_pass": summary.get("oos_pass", False),
                    "objective_value": summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
                }
            )
        _write_csv(
            capital_dir / "stage_b_gate_risk_is_oos.csv",
            fieldnames=list(stage_b_csv_rows[0].keys()) if stage_b_csv_rows else ["rank"],
            rows=stage_b_csv_rows,
        )
        (capital_dir / "top20.json").write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(UTC).isoformat(),
                    "runtime_budget": runtime_budget,
                    "objective_mode": objective_mode,
                    "dd_cap_pct": dd_cap_pct,
                    "dd_cap_basis": dd_cap_basis,
                    "quality_filter": quality_filter,
                    "capital_equity": capital_equity,
                    "capital_currency": capital_currency,
                    "split": split_payload,
                    "top": top_records,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        best_summary = (
            dict(best_record.get("summary", {}))
            if isinstance(best_record, dict) and isinstance(best_record.get("summary", {}), dict)
            else {}
        )
        winner_qualified, winner_reason = _is_qualified_optimizer_winner(best_summary)
        winner_path: Path | None = None
        if winner_qualified:
            winner_path = _write_optimizer_best_config(
                root=root,
                base_config=capital_config,
                source_config_path=config_path,
                report_dir=capital_dir,
                best_record=best_record,
                split_payload=split_payload,
                objective_mode=objective_mode,
                dd_cap_pct=dd_cap_pct,
                dd_cap_basis=dd_cap_basis,
                output_filename=f"config.variant_RESEARCH_OPT_BEST_{_capital_file_tag(capital_equity, capital_currency)}.yaml",
                capital_equity=capital_equity,
                capital_currency=capital_currency,
                quality_filter_mode=quality_filter_mode,
            )
        (capital_dir / "best.json").write_text(
            json.dumps(
                {
                    "best": best_record,
                    "best_summary": best_summary,
                    "winner_qualified": winner_qualified,
                    "winner_reason": winner_reason,
                    "winner_config_path": str(winner_path) if winner_path is not None else None,
                    "split": split_payload,
                    "objective_mode": objective_mode,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        checkpoint["metadata"]["completed"] = True
        checkpoint["metadata"]["winner_qualified"] = winner_qualified
        checkpoint["metadata"]["winner_reason"] = winner_reason
        checkpoint["metadata"]["winner_config_path"] = str(winner_path) if winner_path is not None else None
        save_checkpoint(checkpoint_path, checkpoint)
        return {
            "capital_equity": capital_equity,
            "capital_currency": capital_currency,
            "capital_dir": str(capital_dir),
            "winner_qualified": winner_qualified,
            "winner_reason": winner_reason,
            "winner_config_path": str(winner_path) if winner_path is not None else None,
            "best_summary": best_summary,
            "stage_b_rows": stage_b_csv_rows,
        }

    capital_results: list[dict[str, object]] = []
    if capital_run_mode == "parallel" and len(capitals) > 1:
        with ThreadPoolExecutor(max_workers=min(len(capitals), workers)) as capital_executor:
            futures = [capital_executor.submit(_run_single_capital, item) for item in capitals]
            for future in as_completed(futures):
                capital_results.append(future.result())
    else:
        for item in capitals:
            capital_results.append(_run_single_capital(item))

    capital_results.sort(
        key=lambda item: (str(item.get("capital_currency", "")), float(item.get("capital_equity", 0.0)))
    )
    dual_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for item in capital_results:
        stage_rows = item.get("stage_b_rows", [])
        if isinstance(stage_rows, list):
            dual_rows.extend(stage_rows)
        best_summary = item.get("best_summary", {})
        if not isinstance(best_summary, dict):
            best_summary = {}
        summary_rows.append(
            {
                "capital_equity": item.get("capital_equity"),
                "capital_currency": item.get("capital_currency"),
                "capital_dir": item.get("capital_dir"),
                "winner_qualified": bool(item.get("winner_qualified", False)),
                "winner_reason": item.get("winner_reason"),
                "winner_config_path": item.get("winner_config_path"),
                "best_objective_value": best_summary.get("objective_value", OBJECTIVE_FAIL_VALUE),
                "best_oos_total_pnl_net": best_summary.get("oos_total_pnl_net", 0.0),
                "best_oos_dd_ref_pct": best_summary.get("oos_dd_ref_pct", 0.0),
                "best_quality_pass": best_summary.get("quality_pass", False),
            }
        )
    if dual_rows:
        _write_csv(run_dir / "dual_capital_ranking.csv", fieldnames=list(dual_rows[0].keys()), rows=dual_rows)
    (run_dir / "dual_capital_summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "runtime_budget": runtime_budget,
                "objective_mode": objective_mode,
                "dd_cap_pct": dd_cap_pct,
                "dd_cap_basis": dd_cap_basis,
                "quality_filter": quality_filter,
                "split": split_payload,
                "capitals": summary_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "winner_qualification.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "all_capitals_qualified": all(bool(item.get("winner_qualified", False)) for item in capital_results),
                "capitals": summary_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "dual_capital_meta.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "resume_key": resume_key,
                "completed": True,
                "capitals_done": summary_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Research optimize dual-capital summary saved: %s", run_dir / "dual_capital_summary.json")
