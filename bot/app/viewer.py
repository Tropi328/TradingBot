"""Dashboard, Monte Carlo viewer, and related helpers — extracted from app_main.py."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bot.app.config_helpers import _daily_gate_mode, _resolve_runtime_path
from bot.backtest.monte_carlo import run_monte_carlo_simulation
from bot.config import AppConfig

LOGGER = logging.getLogger("trading_bot")


@dataclass(slots=True)
class ViewerManager:
    dashboard_server: object | None = None
    mc_viewer_proc: subprocess.Popen | None = None
    mc_viewer_stderr_fh: object | None = None


_viewer_manager = ViewerManager()
_dashboard_server = None
_mc_viewer_proc: subprocess.Popen | None = None
_mc_viewer_stderr_fh = None


def _load_legacy_viewer_state() -> None:
    if _dashboard_server is not None:
        _viewer_manager.dashboard_server = _dashboard_server
    if _mc_viewer_proc is not None:
        _viewer_manager.mc_viewer_proc = _mc_viewer_proc
    if _mc_viewer_stderr_fh is not None:
        _viewer_manager.mc_viewer_stderr_fh = _mc_viewer_stderr_fh


def _store_legacy_viewer_state() -> None:
    global _dashboard_server, _mc_viewer_proc, _mc_viewer_stderr_fh
    _dashboard_server = _viewer_manager.dashboard_server
    _mc_viewer_proc = _viewer_manager.mc_viewer_proc
    _mc_viewer_stderr_fh = _viewer_manager.mc_viewer_stderr_fh


def _maybe_start_dashboard(args: argparse.Namespace, config: AppConfig) -> None:
    """Start the web dashboard in the background (non-blocking).

    Called *before* the backtest runs so the user can watch data live.
    The JSONL file doesn't need to exist yet – the reader thread will
    wait for it to appear.
    """
    _load_legacy_viewer_state()
    if not getattr(args, "dashboard", False):
        return
    if not config.diagnostics.decision_trace_enabled:
        return
    trace_path = Path(config.diagnostics.decision_trace_path)
    try:
        from tools.termviz_web import start_dashboard

        LOGGER.info("Starting decision-trace dashboard on http://localhost:%d ...", config.diagnostics.dashboard_port)
        _viewer_manager.dashboard_server = start_dashboard(
            path=trace_path,
            port=config.diagnostics.dashboard_port,
            open_browser=True,
            blocking=False,  # returns immediately
        )
        _store_legacy_viewer_state()
    except (ImportError, OSError):
        LOGGER.exception("Could not start dashboard")


def _maybe_block_dashboard(*, hold_open: bool = False) -> None:
    """Optionally keep process alive so dashboard/MC viewer stay open.

    Called *after* backtest. When hold_open is False, viewers are closed and
    function returns immediately. When hold_open is True, process blocks until
    dashboard/viewer are closed or interrupted.
    """
    _load_legacy_viewer_state()
    has_dashboard = _viewer_manager.dashboard_server is not None
    has_mc = _viewer_manager.mc_viewer_proc is not None and _viewer_manager.mc_viewer_proc.poll() is None

    if not has_dashboard and not has_mc:
        return

    if not hold_open:
        if _viewer_manager.dashboard_server is not None:
            _viewer_manager.dashboard_server.shutdown()
            _viewer_manager.dashboard_server = None
        _maybe_stop_mc_viewer()
        _store_legacy_viewer_state()
        return

    parts: list[str] = []
    if has_dashboard:
        parts.append("dashboard")
    if has_mc:
        parts.append("MC viewer")
    running_label = " + ".join(parts)

    print(f"\n  Backtest complete – {running_label} still running.")
    print("  Press Ctrl+C to stop.\n")
    try:
        while True:
            # If MC viewer was closed by the user (window closed), note it
            if _viewer_manager.mc_viewer_proc is not None and _viewer_manager.mc_viewer_proc.poll() is not None:
                _viewer_manager.mc_viewer_proc = None
            # If dashboard is gone and MC viewer is gone, stop blocking
            if _viewer_manager.dashboard_server is None and (
                _viewer_manager.mc_viewer_proc is None or _viewer_manager.mc_viewer_proc.poll() is not None
            ):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if _viewer_manager.dashboard_server is not None:
            _viewer_manager.dashboard_server.shutdown()
            _viewer_manager.dashboard_server = None
        _maybe_stop_mc_viewer()
        _store_legacy_viewer_state()
        print("Stopped.")


# ---------------------------------------------------------------------------
# Monte Carlo live viewer auto-launch
# ---------------------------------------------------------------------------


def _maybe_start_mc_viewer(config: AppConfig, root: Path, cli_override: bool | None = None) -> None:
    """Spawn the Monte Carlo live viewer as a child process (non-blocking).

    Starts when ``monte_carlo.live_window.enabled`` is True and
    ``viewer_mode`` is ``"process"`` or ``"terminal"`` — or when forced
    via *cli_override*.  Failures are logged and swallowed.
    """
    _load_legacy_viewer_state()
    lw = config.monte_carlo.live_window

    # CLI --mc-viewer / --no-mc-viewer override config
    if cli_override is not None:
        if not cli_override:
            return
        # cli_override True → force-start regardless of config flags
    else:
        if not lw.enabled or lw.viewer_mode not in ("process", "terminal") or not lw.open_on_start:
            return

    png_path = root / lw.png_path
    json_path = root / lw.json_path
    # Ensure the parent directory exists so the bot can later write files
    png_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Terminal mode: launch the plotext-based terminal viewer ----------
    if lw.viewer_mode == "terminal":
        viewer_script = root / "tools" / "termviz_mc.py"
        if not viewer_script.exists():
            LOGGER.warning("MC terminal viewer script not found: %s — skipping", viewer_script)
            return
        cmd = [
            sys.executable,
            str(viewer_script),
            "--json",
            str(json_path),
            "--refresh",
            str(lw.refresh_seconds),
            "--title",
            lw.window_title,
        ]
        mc_log = root / "logs" / "mc_viewer.log"
        mc_log.parent.mkdir(parents=True, exist_ok=True)
        stderr_fh = None
        try:
            stderr_fh = open(mc_log, "w", encoding="utf-8")  # noqa: SIM115
            # CREATE_NEW_CONSOLE on Windows gives the viewer its own terminal window
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_CONSOLE
            _viewer_manager.mc_viewer_proc = subprocess.Popen(
                cmd,
                stderr=stderr_fh,
                creationflags=creation_flags,
            )
            _viewer_manager.mc_viewer_stderr_fh = stderr_fh
            LOGGER.info("MC terminal viewer started (PID %s)", _viewer_manager.mc_viewer_proc.pid)
            time.sleep(0.5)
            if _viewer_manager.mc_viewer_proc.poll() is not None:
                rc = _viewer_manager.mc_viewer_proc.returncode
                _viewer_manager.mc_viewer_proc = None
                LOGGER.warning(
                    "MC terminal viewer exited immediately (code %s) — check %s",
                    rc,
                    mc_log,
                )
        except Exception:
            LOGGER.exception("Could not start MC terminal viewer")
            if stderr_fh is not None and _viewer_manager.mc_viewer_proc is None:
                try:
                    stderr_fh.close()
                except OSError:
                    pass
                _viewer_manager.mc_viewer_stderr_fh = None
        _store_legacy_viewer_state()
        return

    # -- Process mode: launch the matplotlib PNG viewer -------------------
    viewer_script = root / "tools" / "monte_carlo_live_viewer.py"
    if not viewer_script.exists():
        LOGGER.warning("MC viewer script not found: %s — skipping auto-launch", viewer_script)
        return

    cmd = [
        sys.executable,
        str(viewer_script),
        "--png",
        str(png_path),
        "--json",
        str(json_path),
        "--refresh",
        str(lw.refresh_seconds),
        "--title",
        lw.window_title,
        "--max-fps",
        str(lw.max_fps),
    ]
    if not lw.show_stats_overlay:
        cmd.append("--no-overlay")

    # Log stderr to file so viewer errors are visible for debugging
    mc_log = root / "logs" / "mc_viewer.log"
    mc_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_fh = None
    try:
        stderr_fh = open(mc_log, "w", encoding="utf-8")  # noqa: SIM115
        _viewer_manager.mc_viewer_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
        )
        _viewer_manager.mc_viewer_stderr_fh = stderr_fh  # store for cleanup
        LOGGER.info("Monte Carlo live viewer started (PID %s)", _viewer_manager.mc_viewer_proc.pid)
        # Brief health-check: give the process a moment to die on import errors
        time.sleep(0.5)
        if _viewer_manager.mc_viewer_proc.poll() is not None:
            rc = _viewer_manager.mc_viewer_proc.returncode
            _viewer_manager.mc_viewer_proc = None
            LOGGER.warning(
                "MC viewer exited immediately (code %s) — check %s for details",
                rc,
                mc_log,
            )
    except Exception:
        LOGGER.exception("Could not start Monte Carlo live viewer")
        # Close stderr_fh if Popen failed so we don't leak the handle
        if stderr_fh is not None and _viewer_manager.mc_viewer_proc is None:
            try:
                stderr_fh.close()
            except OSError:
                pass
            _viewer_manager.mc_viewer_stderr_fh = None
    _store_legacy_viewer_state()


def _maybe_stop_mc_viewer() -> None:
    """Terminate the MC viewer child process if it is still running."""
    if _viewer_manager.mc_viewer_proc is None:
        _store_legacy_viewer_state()
        return
    try:
        _viewer_manager.mc_viewer_proc.terminate()
        _viewer_manager.mc_viewer_proc.wait(timeout=3)
    except Exception:
        LOGGER.debug("MC viewer process cleanup failed", exc_info=True)
    _viewer_manager.mc_viewer_proc = None
    # Close the stderr log file handle
    if _viewer_manager.mc_viewer_stderr_fh is not None:
        try:
            _viewer_manager.mc_viewer_stderr_fh.close()
        except OSError:
            pass
        _viewer_manager.mc_viewer_stderr_fh = None
    _store_legacy_viewer_state()


def _resolve_monte_carlo_starting_equity(
    *,
    mode: str,
    configured_initial_equity: float,
    initial_equity: float | None,
    current_equity: float | None,
    fallback_current_equity: float | None,
) -> float:
    mode_norm = str(mode or "initial").strip().lower()
    if mode_norm not in {"initial", "current"}:
        mode_norm = "initial"

    configured = float(configured_initial_equity) if configured_initial_equity > 0 else 1.0
    initial = configured
    if initial_equity is not None:
        try:
            initial_candidate = float(initial_equity)
        except (TypeError, ValueError):
            initial_candidate = None
        if initial_candidate is not None and initial_candidate > 0:
            initial = initial_candidate

    if mode_norm == "current":
        for candidate in (current_equity, fallback_current_equity):
            if candidate is None:
                continue
            try:
                current_val = float(candidate)
            except (TypeError, ValueError):
                continue
            if current_val > 0:
                return current_val

    return initial


def _run_monte_carlo_for_payloads(
    config: AppConfig,
    root: Path,
    gate_payloads: dict[str, dict[str, object]],
) -> None:
    """Extract trade PnLs from gate_payloads and run Monte Carlo simulation.

    When multiple gate modes are present (e.g. off / trend / trend_vol_news),
    uses only the **primary** configured mode — mixing trades from different
    strategy variants would produce meaningless MC results.
    Falls back to the first available mode if the configured mode is absent.
    """
    mc = config.monte_carlo
    if not mc.enabled:
        return

    # Pick the single gate mode to simulate: configured mode first,
    # else the first mode with actual reports.
    primary_mode = _daily_gate_mode(config)
    if primary_mode not in gate_payloads:
        primary_mode = next(iter(gate_payloads), None)
    if primary_mode is None:
        LOGGER.info("Monte Carlo: no gate payloads — skipping simulation.")
        return

    payload = gate_payloads[primary_mode]
    reports_map = payload.get("reports")
    if not isinstance(reports_map, dict):
        LOGGER.info("Monte Carlo: no reports in gate mode %s — skipping.", primary_mode)
        return

    # Collect PnLs from selected gate mode only
    pnls: list[float] = []
    for _sym, report in reports_map.items():
        if not isinstance(report, dict):
            continue
        # Walk-forward reports nest under 'aggregate'
        src = report.get("aggregate", report)
        # Prefer pre-computed _trade_pnls list (raw trade PnLs)
        trade_pnls = src.get("_trade_pnls") or report.get("_trade_pnls")
        if trade_pnls:
            pnls.extend(float(v) for v in trade_pnls)
            continue
        # Fallback: try to extract from serialised trade_log dicts
        trade_log = src.get("trade_log", [])
        for trade in trade_log:
            if isinstance(trade, dict):
                pnl_val = trade.get("pnl")
                if pnl_val is not None:
                    pnls.append(float(pnl_val))

    if not pnls:
        LOGGER.info("Monte Carlo: no trades in mode %s — skipping simulation.", primary_mode)
        return

    # Extract both initial and current equity from reports, then choose
    # starting_equity via a single mode resolver.
    initial_equity_report: float | None = None
    current_equity_report: float | None = None
    for _sym, report in reports_map.items():
        if not isinstance(report, dict):
            continue
        src = report.get("aggregate", report)
        if initial_equity_report is None:
            _init = src.get("initial_equity")
            if _init is None:
                _init = report.get("initial_equity")
            try:
                _init_f = float(_init)
            except (TypeError, ValueError):
                _init_f = None
            if _init_f is not None and _init_f > 0:
                initial_equity_report = _init_f
        if current_equity_report is None:
            _eq = src.get("equity_end")
            if _eq is None:
                _eq = report.get("equity_end")
            try:
                _eq_f = float(_eq)
            except (TypeError, ValueError):
                _eq_f = None
            if _eq_f is not None and _eq_f > 0:
                current_equity_report = _eq_f
    configured_initial = float(config.risk.equity)
    fallback_current = configured_initial + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=configured_initial,
        initial_equity=initial_equity_report,
        current_equity=current_equity_report,
        fallback_current_equity=fallback_current,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")


def _run_monte_carlo_for_trades(
    config: AppConfig,
    root: Path,
    trade_log: list[object],
) -> None:
    """Run Monte Carlo directly from a list of BacktestTrade objects."""
    mc = config.monte_carlo
    if not mc.enabled:
        return

    pnls: list[float] = []
    for trade in trade_log:
        pnl_val = getattr(trade, "pnl", None)
        if pnl_val is not None:
            pnls.append(float(pnl_val))

    if not pnls:
        LOGGER.info("Monte Carlo: no trades found — skipping simulation.")
        return

    configured_initial = float(config.risk.equity)
    current_equity = configured_initial + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=configured_initial,
        initial_equity=configured_initial,
        current_equity=current_equity,
        fallback_current_equity=current_equity,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")


def _run_monte_carlo_for_batch(
    config: AppConfig,
    root: Path,
    out_root: Path,
    summary: dict[str, object],
) -> None:
    """Run Monte Carlo on the combined batch-backtest trades (from parquet)."""
    mc = config.monte_carlo
    if not mc.enabled:
        return

    parquet_path = out_root / "all" / "combined_trades.parquet"
    if not parquet_path.exists():
        LOGGER.info("Monte Carlo: no combined_trades.parquet — skipping.")
        return

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(str(parquet_path), columns=["pnl"])
        pnls = [float(v) for v in table.column("pnl").to_pylist() if v is not None]
    except Exception:
        LOGGER.exception("Monte Carlo: failed to read batch trade PnLs from parquet")
        return

    if not pnls:
        LOGGER.info("Monte Carlo: no trades in batch results — skipping.")
        return

    combined_metrics = summary.get("combined_metrics", {})
    if not isinstance(combined_metrics, dict):
        combined_metrics = {}
    try:
        initial_equity = float(combined_metrics.get("initial_equity", config.risk.equity))
    except (TypeError, ValueError):
        initial_equity = float(config.risk.equity)
    try:
        current_equity = float(combined_metrics.get("equity_end"))
    except (TypeError, ValueError):
        current_equity = None
    fallback_current = initial_equity + float(sum(pnls))
    starting_equity = _resolve_monte_carlo_starting_equity(
        mode=mc.equity_mode_backtest,
        configured_initial_equity=float(config.risk.equity),
        initial_equity=initial_equity,
        current_equity=current_equity,
        fallback_current_equity=fallback_current,
    )

    png_path = _resolve_runtime_path(root, mc.live_window.png_path)
    json_path = _resolve_runtime_path(root, mc.live_window.json_path)

    try:
        run_monte_carlo_simulation(
            trade_pnls=pnls,
            starting_equity=starting_equity,
            png_path=png_path,
            json_path=json_path,
            num_simulations=mc.num_simulations,
            ruin_dd_threshold=mc.ruin_dd_threshold,
            seed=mc.seed,
            max_paths_plotted=mc.max_paths_plotted,
            sampling_mode=mc.sampling_mode,
            block_size=mc.block_size,
            equity_mode=mc.equity_mode_backtest,
            ruin_equity_floor_pct=mc.ruin_equity_floor_pct,
            ruin_equity_floor_abs=mc.ruin_equity_floor_abs,
            count_breakeven_as_loss=mc.count_breakeven_as_loss,
        )
    except Exception:
        LOGGER.exception("Monte Carlo simulation failed")
