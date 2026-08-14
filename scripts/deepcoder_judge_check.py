#!/usr/bin/env python3
# lifecycle: one-off (archive when the deepcoder judge is validated)
"""Judge-integrity check: run the dataset's own reference solutions through
DeepCoderTask.reward. DeepCoder verified every problem's tests against a
reference solution with THEIR judge; a reference solution failing OURS marks
a judge mismatch (output normalization, multi-answer problems), which would
otherwise masquerade as model difficulty in the sweep labels.

Samples N problems per source config, takes the first parseable Python
solution per problem. Reports pass rate per source + failure examples.

Run:  .venv/bin/python scripts/deepcoder_judge_check.py [N-per-source]
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasets import load_dataset  # noqa: E402

from mlx_rl.tasks.deepcoder import DeepCoderTask  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NAME = "agentica-org/DeepCoder-Preview-Dataset"


def solutions_of(row) -> list[str]:
    raw = row.get("solutions")
    if not raw:
        return []
    sols = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(sols, dict):  # some taco rows: {"language": [...], "solution": [...]}
        sols = sols.get("solution", [])
    out = []
    for s in sols:
        if isinstance(s, str) and s.strip():
            # some solutions arrive fenced; unwrap the first python block
            if "```" in s:
                import re
                m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", s, re.DOTALL)
                s = m.group(1) if m else s
            out.append(s.strip())
    return out


def main() -> None:
    task = DeepCoderTask()
    by_id = {}
    for split_rows in (task._train, task._eval):
        for r in split_rows:
            by_id[r["id"]] = r
    rng = random.Random(7)

    stats = Counter()
    fails = defaultdict(list)
    for cfg, hf_split in (("taco", "train"), ("primeintellect", "train"),
                          ("lcbv5", "train")):
        ds = load_dataset(NAME, cfg, split=hf_split)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        tested = 0
        for idx in idxs:
            if tested >= N:
                break
            rid = f"{cfg}-{hf_split}-{idx}"
            if rid not in by_id:  # dropped at fetch (fncall etc.)
                continue
            sols = solutions_of(ds[idx])
            if not sols:
                stats[f"{cfg}:no-solution"] += 1
                continue
            ex = task._example(by_id[rid])
            # wrap as a completion so the extractor path is exercised too
            r = task.reward(ex, f"```python\n{sols[0]}\n```")
            tested += 1
            key = f"{cfg}:{'pass' if r.total >= 1.0 else 'FAIL'}"
            stats[key] += 1
            if r.total < 1.0 and len(fails[cfg]) < 3:
                fails[cfg].append(rid)
        print(f"{cfg}: {stats[f'{cfg}:pass']}/{tested} reference solutions pass "
              f"(no-solution skips: {stats[f'{cfg}:no-solution']})", flush=True)
    print("\nfailing ids for inspection:", dict(fails))


if __name__ == "__main__":
    main()
