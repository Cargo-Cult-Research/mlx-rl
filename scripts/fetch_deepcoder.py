#!/usr/bin/env python3
# lifecycle: one-off (archive when the deepcoder task is retired)
"""Fetch + normalize the DeepCoder-Preview-Dataset for the deepcoder task.

Downloads agentica-org/DeepCoder-Preview-Dataset (taco, primeintellect,
lcbv5 — the curated-for-RL union of TACO-verified, SYNTHETIC-1 and
pre-cutoff LiveCodeBench; every problem ships tests verified against a
reference solution) and writes one normalized, self-contained line per
problem to data/deepcoder/{train,test}.jsonl.gz:

    {"id": "<cfg>-<idx>", "source": cfg, "split": ..., "problem": ...,
     "starter_code": ..., "tests": [{"input": ..., "output": ...}, ...]}

v1 keeps ONLY stdin/stdout-judged problems — one judge, no ambiguity.
Function-call-judged problems (taco fn_name, lcbv5 functional) are counted
and reported, not silently dropped. Oversized test cases (>200 KB) are
skipped; problems left with no usable case are dropped and counted. Tests
are thinned to <=16 per problem, evenly spaced (grading runs all stored
cases, first failure short-circuits).

Needs `datasets` (installed in .venv ad hoc; not a runtime dependency —
the task module reads the .jsonl.gz with stdlib only).

Run:  .venv/bin/python scripts/fetch_deepcoder.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

NAME = "agentica-org/DeepCoder-Preview-Dataset"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "deepcoder"
MAX_CASE_BYTES = 200_000
MAX_TESTS = 16
CONFIGS = {  # cfg -> {hf split -> our split}
    "taco": {"train": "train"},
    "primeintellect": {"train": "train"},
    "lcbv5": {"train": "train", "test": "test"},
}


def norm_tests(cfg: str, row: dict, stats: Counter) -> list[dict] | None:
    """-> [{'input','output'}] for stdin/stdout problems, None to drop."""
    raw = row["tests"]
    tests = json.loads(raw) if isinstance(raw, str) else raw
    cases = []
    if isinstance(tests, dict):  # taco: {"inputs": [...], "outputs": [...]}
        if tests.get("fn_name"):
            stats[f"{cfg}:drop-fncall"] += 1
            return None
        pairs = zip(tests.get("inputs", []), tests.get("outputs", []))
        cases = [{"input": i, "output": o} for i, o in pairs]
    elif isinstance(tests, list):  # primeintellect / lcbv5
        for t in tests:
            typ = t.get("type") or t.get("testtype")
            if typ in ("stdin_stdout", "stdin"):
                cases.append({"input": t["input"], "output": t["output"]})
            else:
                stats[f"{cfg}:case-nonstdin"] += 1
        if not cases and tests:
            stats[f"{cfg}:drop-fncall"] += 1
            return None
    ok = []
    for c in cases:
        if not isinstance(c["input"], str) or not isinstance(c["output"], str):
            stats[f"{cfg}:case-nonstr"] += 1
            continue
        if len(c["input"]) + len(c["output"]) > MAX_CASE_BYTES:
            stats[f"{cfg}:case-oversize"] += 1
            continue
        ok.append(c)
    if not ok:
        stats[f"{cfg}:drop-no-usable-case"] += 1
        return None
    if len(ok) > MAX_TESTS:  # evenly spaced thinning, deterministic
        step = len(ok) / MAX_TESTS
        ok = [ok[int(i * step)] for i in range(MAX_TESTS)]
        stats[f"{cfg}:thinned"] += 1
    return ok


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    seen: set[str] = set()
    writers = {}
    try:
        for cfg, splits in CONFIGS.items():
            for hf_split, split in splits.items():
                ds = load_dataset(NAME, cfg, split=hf_split)
                for idx, row in enumerate(ds):
                    problem = (row.get("problem") or "").strip()
                    if len(problem) < 40:
                        stats[f"{cfg}:drop-empty-problem"] += 1
                        continue
                    h = hashlib.sha1(problem.encode()).hexdigest()
                    if h in seen:
                        stats[f"{cfg}:drop-dup"] += 1
                        continue
                    tests = norm_tests(cfg, row, stats)
                    if tests is None:
                        continue
                    seen.add(h)
                    if split not in writers:
                        writers[split] = gzip.open(
                            OUT_DIR / f"{split}.jsonl.gz", "wt")
                    writers[split].write(json.dumps({
                        "id": f"{cfg}-{hf_split}-{idx}",
                        "source": cfg,
                        "split": split,
                        "problem": problem,
                        "starter_code": (row.get("starter_code") or "").strip(),
                        "tests": tests,
                    }) + "\n")
                    stats[f"{cfg}:kept-{split}"] += 1
                print(f"{cfg}/{hf_split} done", flush=True)
    finally:
        for w in writers.values():
            w.close()
    print("\n=== stats ===")
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    report = OUT_DIR / "FETCH-REPORT.txt"
    report.write_text("\n".join(f"{k}: {stats[k]}" for k in sorted(stats)) + "\n")
    print(f"\nwritten: {sorted(p.name for p in OUT_DIR.iterdir())}")


if __name__ == "__main__":
    main()
