from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot.backtest.data_provider import AutoDataLoader
from bot.backtest.engine import BacktestVariant, run_backtest_multi_strategy
from bot.backtest.runner import BacktestRunner
from bot.config import AppConfig, load_config


@dataclass(slots=True)
class SweepRow:
    spread_limit_points: float
    min_risk_cash_per_trade: float
    no_overnight: bool
    min_edge_to_cost_ratio: float
    total_pnl_net: float
    equity_end: float
    max_dd_net: float
    profit_factor_net: float
    trades: int
    wins: int
    losses: int
    spread_cost_sum: float
    swap_cost_sum: float
    slippage_cost_sum: float
    commission_cost_sum: float
    avg_spread_points: float
    median_spread_points: float
    p90_spread_points: float
    forced_closes_count: int
    min_size_overrides_count: int
    rejected_size_too_small: int
    rejected_spread_too_wide: int
    rejected_edge_too_small: int
    rejected_risk_after_rounding_too_high: int
    score: float
    eligible: bool


def _parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif key in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Unsupported boolean value '{item}'")
    if not out:
        raise ValueError("Empty boolean list")
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        out.append(float(value))
    if not out:
        raise ValueError("Empty float list")
    return out


def _resolve_asset(config: AppConfig, symbol: str):
    symbol_norm = str(symbol).strip().upper()
    for asset in config.assets:
        if asset.epic.upper() == symbol_norm:
            return asset
    aliases = {symbol_norm}
    if symbol_norm == "XAUUSD":
        aliases.add("GOLD")
    elif symbol_norm == "GOLD":
        aliases.add("XAUUSD")
    for asset in config.assets:
        if asset.epic.upper() in aliases:
            return asset
    if config.assets:
        return config.assets[0]
    raise ValueError(f"Asset '{symbol}' not found in config.assets")


def _assumed_spread(config: AppConfig, symbol: str, frame: pd.DataFrame) -> float:
    if "spread" in frame.columns:
        spread = pd.to_numeric(frame["spread"], errors="coerce").dropna()
        if not spread.empty:
            return float(max(0.0, spread.median()))
    spread_map = config.backtest_tuning.assumed_spread_by_symbol
    return float(spread_map.get(symbol.upper(), 0.2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep micro-account XAUUSD parameters and rank results.")
    parser.add_argument("--config", default="configs/variants/config.variant_100PLN_BASE.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--price-mode", default="mid", choices=["mid", "bid", "ask"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-02-01")
    parser.add_argument("--variant", default="W0")
    parser.add_argument("--lambda-dd", type=float, default=0.3)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-profit-factor", type=float, default=1.05)

    parser.add_argument("--spread-limits", default="2,3,4,5,6,8,10")
    parser.add_argument("--min-risk-cash", default="0.10,0.20,0.30,0.50,0.80,1.00")
    parser.add_argument("--no-overnight", default="true,false")
    parser.add_argument("--edge-ratios", default="3,4,5,6")

    parser.add_argument("--out-dir", default="reports/sweeps")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    symbol = str(args.symbol).strip().upper()
    timeframe = str(args.timeframe).strip().lower()

    start_dt = datetime.fromisoformat(str(args.start)).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(str(args.end)).replace(tzinfo=timezone.utc)

    spread_limits = _parse_float_list(args.spread_limits)
    min_risk_cash_values = _parse_float_list(args.min_risk_cash)
    no_overnight_values = _parse_bool_list(args.no_overnight)
    edge_ratios = _parse_float_list(args.edge_ratios)

    data_loader = AutoDataLoader(args.data_root)
    loaded = data_loader.load_symbol_data(
        symbol=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        price_mode=args.price_mode,
    )
    candles = BacktestRunner._frame_to_candles(loaded.frame)
    if not candles:
        raise RuntimeError("No candles loaded for requested sweep range")

    assumed_spread = _assumed_spread(config, symbol, loaded.frame)
    asset_template = _resolve_asset(config, symbol)

    variant = BacktestVariant(code=str(args.variant).strip().upper())
    rows: list[SweepRow] = []

    for spread_limit in spread_limits:
        for min_risk_cash in min_risk_cash_values:
            for no_overnight in no_overnight_values:
                for edge_ratio in edge_ratios:
                    cfg = config.model_copy(deep=True)
                    cfg.risk.min_risk_cash_per_trade = float(min_risk_cash)
                    cfg.backtest_tuning.spread_limit_points = float(spread_limit)
                    cfg.backtest_tuning.no_overnight = bool(no_overnight)
                    cfg.backtest_tuning.min_edge_to_cost_ratio = float(edge_ratio)

                    asset = _resolve_asset(cfg, asset_template.epic)
                    report = run_backtest_multi_strategy(
                        config=cfg,
                        asset=asset,
                        candles_m5=candles,
                        assumed_spread=assumed_spread,
                        variant=variant,
                        data_context={
                            **(loaded.diagnostics or {}),
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "price_mode_requested": args.price_mode,
                        },
                    )
                    score = float(report.total_pnl_net) - (float(args.lambda_dd) * float(report.max_drawdown_net))
                    eligible = bool(report.trades >= int(args.min_trades) and report.profit_factor_net >= float(args.min_profit_factor))
                    rejected = report.rejected_by_reason or {}
                    rows.append(
                        SweepRow(
                            spread_limit_points=float(spread_limit),
                            min_risk_cash_per_trade=float(min_risk_cash),
                            no_overnight=bool(no_overnight),
                            min_edge_to_cost_ratio=float(edge_ratio),
                            total_pnl_net=float(report.total_pnl_net),
                            equity_end=float(report.equity_end),
                            max_dd_net=float(report.max_drawdown_net),
                            profit_factor_net=float(report.profit_factor_net),
                            trades=int(report.trades),
                            wins=int(report.wins),
                            losses=int(report.losses),
                            spread_cost_sum=float(report.spread_cost_sum),
                            swap_cost_sum=float(report.swap_cost_sum),
                            slippage_cost_sum=float(report.slippage_cost_sum),
                            commission_cost_sum=float(report.commission_cost_sum),
                            avg_spread_points=float(report.avg_spread_points),
                            median_spread_points=float(report.median_spread_points),
                            p90_spread_points=float(report.p90_spread_points),
                            forced_closes_count=int(report.forced_closes_count),
                            min_size_overrides_count=int(report.min_size_overrides_count),
                            rejected_size_too_small=int(rejected.get("SIZE_TOO_SMALL", 0)),
                            rejected_spread_too_wide=int(rejected.get("SPREAD_TOO_WIDE", 0)),
                            rejected_edge_too_small=int(rejected.get("EDGE_TOO_SMALL", 0)),
                            rejected_risk_after_rounding_too_high=int(rejected.get("RISK_AFTER_ROUNDING_TOO_HIGH", 0)),
                            score=float(score),
                            eligible=eligible,
                        )
                    )

    ranked = sorted(
        rows,
        key=lambda row: (int(row.eligible), row.score, row.total_pnl_net, row.equity_end),
        reverse=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = f"sweep_xauusd_micro_{stamp}"
    csv_path = out_dir / f"{base_name}.csv"
    json_path = out_dir / f"{base_name}.json"

    fieldnames = list(SweepRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow(asdict(row))

    payload = {
        "meta": {
            "symbol": symbol,
            "timeframe": timeframe,
            "price_mode": args.price_mode,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "variant": variant.code,
            "grid": {
                "spread_limits": spread_limits,
                "min_risk_cash": min_risk_cash_values,
                "no_overnight": no_overnight_values,
                "edge_ratios": edge_ratios,
            },
            "constraints": {
                "min_trades": int(args.min_trades),
                "min_profit_factor": float(args.min_profit_factor),
                "lambda_dd": float(args.lambda_dd),
            },
            "assumed_spread": float(assumed_spread),
            "rows": len(ranked),
        },
        "top10": [asdict(item) for item in ranked[:10]],
        "all": [asdict(item) for item in ranked],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Sweep complete: {len(ranked)} runs")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("TOP 10:")
    for idx, row in enumerate(ranked[:10], start=1):
        print(
            f"{idx:02d}. eligible={row.eligible} score={row.score:.4f} pnl={row.total_pnl_net:.4f} "
            f"eq={row.equity_end:.4f} dd={row.max_dd_net:.4f} pf={row.profit_factor_net:.4f} "
            f"trades={row.trades} spread_lim={row.spread_limit_points} min_risk_cash={row.min_risk_cash_per_trade} "
            f"no_overnight={row.no_overnight} edge={row.min_edge_to_cost_ratio}"
        )


if __name__ == "__main__":
    main()
