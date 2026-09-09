"""One JSONL reader. Blank lines are skipped everywhere; a malformed line
raises unless lenient=True (append-only caches that may hold a torn last
line), in which case it is skipped."""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path, lenient: bool = False) -> list[dict]:
    out = []
    p = Path(path)
    if lenient and not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if not lenient:
                raise
    return out
