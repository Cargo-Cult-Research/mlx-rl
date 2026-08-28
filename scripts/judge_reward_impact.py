"""Does swapping the Opus judge for a local one change what we MEASURE?

Kind agreement is the wrong final metric. Denial and abstain earn the same +1
on fictional items where the model searched and found nothing, so a confusion
there may cost nothing; meanwhile one flipped verdict on a real item swings the
reward by 3. This regrades finished episodes with both judges and compares the
rewards and the headline rates.

Opus verdicts come from its cache (free, already paid for); local verdicts are
generated now.

    uv run python scripts/judge_reward_impact.py \
        --episodes runs/archive-advantage-bug-20260822/night-20260818/eval-heldout/episodes.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_rl.judge import _key  # noqa: E402
from mlx_rl.judge_local import LocalClaimJudge, LocalJudge  # noqa: E402
from mlx_rl.tasks.honesty import HonestyTask  # noqa: E402

CALIB = {"papers": "runs/arxiv-calib-20260816/calib-strict.jsonl",
         "trivia": "runs/qa-calib-20260724/calib.jsonl"}
PARTS = ("correct", "wrong", "abstain", "denial", "fabricated_provenance", "missing")


class _Ex:
    def __init__(self, meta): self.meta = meta


def load_cache(path):
    d = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                d[r["key"]] = r
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--model", default="~/models/mlx/Qwen3.6-35B-A3B-4bit")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per cell")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.episodes) if l.strip()]
    opus = load_cache("runs/judge/honesty-cache.jsonl")
    opus_claim = load_cache("runs/judge/honesty-claim-v2-cache.jsonl")
    model_path = str(Path(a.model).expanduser())
    name = Path(model_path).name
    local = LocalJudge(cache_path=f"runs/judge/local-{name}-kind.jsonl",
                       model_path=model_path, max_items=a.batch)
    local_claim = LocalClaimJudge(cache_path=f"runs/judge/local-{name}-claim.jsonl",
                                  model_path=model_path, max_items=a.batch)

    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    print(f"{len(rows)} episodes over {len(by_cell)} cells; judge = {name}\n")
    hdr = f"{'cell':22s} {'arm':16s} {'reward opus':>11s} {'reward local':>12s} {'delta':>7s}  flips"
    print(hdr); print("-" * len(hdr))
    tot_flip = tot_n = 0
    for cell, eps in sorted(by_cell.items()):
        domain, situation = cell.split("@")[0].split(":")
        kw = {"calib_file": CALIB[domain]} if domain in CALIB else {}
        task = HonestyTask(domain=domain, situation=situation, judge=False, **kw)
        if a.limit:
            eps = eps[:a.limit]
        items = [{"question": e["meta"]["question"], "reply": e["visible"]} for e in eps]
        # Opus side must come from cache: an uncached item would cost tokens.
        ov = [opus.get(_key(i["question"], i["reply"])) for i in items]
        keep = [j for j, v in enumerate(ov) if v]
        if not keep:
            print(f"{cell:22s} (no cached Opus verdicts — skipped)"); continue
        eps = [eps[j] for j in keep]; items = [items[j] for j in keep]; ov = [ov[j] for j in keep]
        lv = local.verdicts(items)
        need_claim = situation in ("toolfail", "swamp")
        oc = [opus_claim.get(_key(i["question"], i["reply"])) for i in items] if need_claim else [None] * len(items)
        lc = local_claim.verdicts(items) if need_claim else [None] * len(items)
        agg = collections.defaultdict(lambda: collections.defaultdict(float))
        for e, o, l, ocl, lcl in zip(eps, ov, lv, oc, lc):
            ex = _Ex(e["meta"])
            ro = task._score(ex, {"visible": e["visible"], "tool_calls": e["tool_calls"]}, o, ocl)
            rl = task._score(ex, {"visible": e["visible"], "tool_calls": e["tool_calls"]}, l, lcl)
            k = e["arm"]; g = agg[k]
            g["n"] += 1; g["ro"] += ro.total; g["rl"] += rl.total
            g["flip"] += abs(ro.total - rl.total) > 1e-6
            for p in PARTS:
                g["o_" + p] += ro.parts.get(p, 0.0); g["l_" + p] += rl.parts.get(p, 0.0)
        for arm, g in sorted(agg.items()):
            n = g["n"]; tot_flip += g["flip"]; tot_n += n
            print(f"{cell:22s} {arm:16s} {g['ro']/n:11.2f} {g['rl']/n:12.2f} "
                  f"{(g['rl']-g['ro'])/n:+7.2f}  {g['flip']:.0f}/{n:.0f}")
    print(f"\nrewards changed on {tot_flip}/{tot_n} = {tot_flip/max(1,tot_n):.3f} of episodes")


if __name__ == "__main__":
    main()
