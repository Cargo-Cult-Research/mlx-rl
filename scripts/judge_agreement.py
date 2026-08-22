"""Can a local model replace the Opus judge on the narrow classification?

Replays items the Opus judge has ALREADY graded (its cache is the label set,
so this costs no Opus tokens) through a local judge and measures agreement.

Yardstick: on 1774 shared items Opus and Sonnet agree on kind 0.964 and on the
exact extracted string only 0.745 — so a local judge should be measured against
~0.96, and exact string identity is the wrong bar for the value field, since
correctness is decided downstream by a mechanical alias match.

    uv run python scripts/judge_agreement.py --model ~/models/mlx/Qwen3-4B-4bit --n 300
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_rl.judge_local import LocalClaimJudge, LocalJudge  # noqa: E402


def norm(v: str | None) -> str:
    return (v or "").strip().lower().rstrip(".").replace("’", "'")


def loose_match(a: str | None, b: str | None) -> bool:
    """Value agreement the way the reward actually uses it: the mechanical
    alias check is substring-ish, so 'Dag Hammarskjold' vs 'Dag Hammarskjold
    (Sweden)' is the same decision."""
    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and (x == y or x in y or y in x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="runs/judge/honesty-cache.jsonl")
    ap.add_argument("--model", default="~/models/mlx/Qwen3.6-27B-4bit")
    ap.add_argument("--n", type=int, default=300, help="items per kind (stratified)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--claim", action="store_true", help="grade the claims judge instead")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.cache) if l.strip()]
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    rng = random.Random(a.seed)
    sample = []
    for kind, items in sorted(by_kind.items()):
        rng.shuffle(items)
        sample += items[:a.n]          # stratified: rare kinds are the interesting ones
    rng.shuffle(sample)
    print(f"{len(sample)} items: " + ", ".join(f"{k}={min(len(v), a.n)}" for k, v in sorted(by_kind.items())))

    model_path = str(Path(a.model).expanduser())
    cls = LocalClaimJudge if a.claim else LocalJudge
    judge = cls(cache_path=f"runs/judge/local-{Path(model_path).name}-{'claim' if a.claim else 'kind'}.jsonl",
                model_path=model_path, max_items=a.batch)
    items = [{"question": r["question"], "reply": r["reply"]} for r in sample]
    t0 = time.time()
    got = judge.verdicts(items)
    wall = time.time() - t0

    n = len(sample)
    agree = sum(1 for r, g in zip(sample, got) if r["kind"] == g["kind"])
    cm = collections.Counter((r["kind"], g["kind"]) for r, g in zip(sample, got) if r["kind"] != g["kind"])
    per = collections.defaultdict(lambda: [0, 0])
    for r, g in zip(sample, got):
        per[r["kind"]][1] += 1
        per[r["kind"]][0] += r["kind"] == g["kind"]
    print(f"\nmodel: {Path(model_path).name}   {wall:.0f}s for {n} items "
          f"({judge.calls} calls, {n / max(wall, 1):.1f} items/s)")
    print(f"kind agreement with Opus: {agree}/{n} = {agree / n:.3f}   (opus-vs-sonnet yardstick 0.964)")
    for k, (ok, tot) in sorted(per.items()):
        print(f"   {k:16s} {ok}/{tot} = {ok / tot:.3f}")
    if cm:
        print("   disagreements (opus -> local):", dict(cm))
    both = [(r, g) for r, g in zip(sample, got)
            if r["kind"] == g["kind"] == getattr(judge, "VALUE_KIND", "answer")]
    if both:
        exact = sum(1 for r, g in both if norm(r["value"]) == norm(g["value"]))
        loose = sum(1 for r, g in both if loose_match(r["value"], g["value"]))
        print(f"   extracted value, both '{judge.VALUE_KIND}': exact {exact}/{len(both)} = {exact / len(both):.3f}"
              f"   loose {loose}/{len(both)} = {loose / len(both):.3f}   (opus-vs-sonnet exact 0.745)")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": model_path, "n": n, "agreement": agree / n, "wall_s": wall,
             "per_kind": {k: v[0] / v[1] for k, v in per.items()},
             "disagreements": {f"{x}->{y}": c for (x, y), c in cm.items()}}, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
