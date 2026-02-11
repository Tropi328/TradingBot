from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DashboardWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write(self, payload: dict[str, Any]) -> None:
        snapshot = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self.path)
