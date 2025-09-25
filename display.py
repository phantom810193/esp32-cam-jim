"""Text display helpers for CLI/LCD simulations."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable, List


def render_messages(
    messages: Iterable[str],
    *,
    log_path: Path | str = "text_test.log",
    enable_print: bool = True,
) -> List[str]:
    """Render up to five messages and persist structured metrics."""

    queued = [str(message) for message in messages]
    start = time.perf_counter()
    displayed: List[str] = []
    for message in queued[:5]:
        displayed.append(message)
        if enable_print:
            print(message, file=sys.stdout)
    duration = time.perf_counter() - start
    payload = {
        "messages": displayed,
        "count": len(displayed),
        "duration_seconds": duration,
        "status": "ok" if len(displayed) >= min(len(queued), 5) else "partial",
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return displayed


__all__ = ["render_messages"]
