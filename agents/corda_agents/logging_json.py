"""JSONL event logger — one event per line. Schema is intentionally loose."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        self._f.write(json.dumps(record, default=str) + "\n")
