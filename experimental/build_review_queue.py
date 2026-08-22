"""Assemble a blind human-review queue from transfer-eval episode files.

Stratified sample across arms and regimes; the arm is kept in the record
but NOT shown on the review page (blind). Each item carries what the judge
saw (question, visible reply), the tool trace, the judge's verdict as the
reward parts encode it, the gold, and the reward — so a human can say
whether the judge (and the grade) got it right.

    uv run python experimental/build_review_queue.py --n 100 --out runs/human-review/queue.jsonl \
        runs/arxiv-transfer-20260817b/episodes.jsonl runs/arxiv-transfer-mt3-arm2-20260817/episodes.jsonl ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def _kind(parts: dict) -> str:
    if parts.get("no_reply"):
        return "no_reply"
    if parts.get("answered"):
        return "answer"
    if parts.get("denial"):
        return "denial"
    if parts.get("abstain"):
        return "abstain"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/human-review/queue.jsonl")
    a = ap.parse_args()
    rows = []
    for f in a.files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["_src"] = f
            rows.append(d)
    # stratify: (arm, regime) buckets, round-robin
    buckets = defaultdict(list)
    for d in rows:
        buckets[(d["arm"], d["meta"].get("regime"))].append(d)
    rng = random.Random(a.seed)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    # weight the hard slices: fictional/known/uncertain first
    order = sorted(keys, key=lambda k: {"fictional": 0, "known": 1, "uncertain": 2}.get(k[1], 3))
    picked, i = [], 0
    while len(picked) < a.n and any(buckets[k] for k in keys):
        k = order[i % len(order)]
        i += 1
        if buckets[k]:
            picked.append(buckets[k].pop())
    rng.shuffle(picked)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in picked:
            m = d["meta"]
            rid = hashlib.sha1((d["_src"] + d["arm"] + m.get("question", "") + d.get("visible", "")).encode()).hexdigest()[:12]
            f.write(json.dumps({
                "id": rid, "arm": d["arm"], "src": d["_src"],
                "question": m.get("question"), "regime": m.get("regime"), "band": m.get("band"),
                "turn": m.get("turn", 0), "today": m.get("today"), "gold": m.get("aliases"),
                "title": m.get("title"), "published": m.get("published"),
                "visible": d.get("visible", ""),
                "tool_calls": [{"query": (c.get("args") or {}).get("query") or (c.get("args") or {}).get("url"),
                                "hits": c.get("hits"), "found": c.get("found_target"),
                                "fallback": c.get("fallback"), "capped": c.get("capped")}
                               for c in d.get("tool_calls", [])],
                "judge_kind": _kind(d.get("parts", {})), "correct": d.get("parts", {}).get("correct"),
                "reward": d.get("reward"), "finish": d.get("finish"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} items -> {out}; by regime:",
          {r: sum(1 for d in picked if d['meta'].get('regime') == r) for r in ('fictional', 'known', 'uncertain', 'post')},
          "by arm:", {arm: sum(1 for d in picked if d['arm'] == arm) for arm in sorted({d['arm'] for d in picked})})


if __name__ == "__main__":
    main()
