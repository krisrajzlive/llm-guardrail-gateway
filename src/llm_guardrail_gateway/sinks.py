"""Event sinks: where tracing and eval events go. A sink is any ``Callable[[dict], None]``."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

Sink = Callable[[dict[str, Any]], None]


class ListSink:
    """Collects events in memory (tests, offline evals)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]


def stdout_sink(event: dict[str, Any]) -> None:
    """One JSON object per line; on Lambda this lands in CloudWatch Logs."""
    sys.stdout.write(json.dumps(event, default=str) + "\n")
    sys.stdout.flush()


class JsonlSink:
    """Appends events to a local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")


def fan_out(*sinks: Sink) -> Sink:
    def _emit(event: dict[str, Any]) -> None:
        for sink in sinks:
            sink(event)

    return _emit
