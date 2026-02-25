from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Callable

from bot.config import AppConfig


def run(
    args: Namespace,
    config: AppConfig,
    root: Path,
    *,
    handler: Callable[[Namespace, AppConfig, Path], None],
) -> None:
    handler(args, config, root)
