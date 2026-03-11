from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from pathlib import Path

from bot.config import AppConfig, AssetConfig


def run_backtest(
    args: Namespace,
    config: AppConfig,
    root: Path,
    *,
    handler: Callable[[Namespace, AppConfig, Path], None],
) -> None:
    handler(args, config, root)


def run_worker(
    args: Namespace,
    config: AppConfig,
    assets: list[AssetConfig],
    root: Path,
    *,
    handler: Callable[[Namespace, AppConfig, list[AssetConfig], Path], None],
) -> None:
    handler(args, config, assets, root)
