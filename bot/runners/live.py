from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Callable

from bot.config import AppConfig, AssetConfig


def run(
    args: Namespace,
    config: AppConfig,
    assets: list[AssetConfig],
    root: Path,
    *,
    handler: Callable[[Namespace, AppConfig, list[AssetConfig], Path], None],
) -> None:
    handler(args, config, assets, root)
