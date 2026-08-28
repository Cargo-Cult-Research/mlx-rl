"""Judge token usage, billed and local, from the judges' own call logs.

    uv run python scripts/judge_usage.py [--days 30]

Every Judge writes one line per API/model call to <cache>.calls.jsonl with the
provider's usage envelope. Nothing here estimates: it sums what was reported.

Input tokens are NOT one number. Almost every judge call is a long fixed
rubric plus a short batch of replies, so the rubric is served from cache and
the three input classes have three different prices -- reporting their sum as
"input tokens" overstates spend by roughly the cache discount. They are kept
apart here and combined only in `billable_equiv`, at the published
multipliers (cache write 1.25x, cache read 0.1x).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import time
from pathlib import Path

# Anchored to the repo, not the caller's cwd: rl-dash serves from cwd "/",
# where a relative glob silently matches nothing and the spend graph reads
# zero -- a wrong answer that looks like good news.
ROOT = Path(__file__).resolve().parent.parent
WRITE_MULT, READ_MULT = 1.25, 0.10
LOCAL_HINTS = ("qwen", "llama", "mistral", "gemma")


def _is_local(rec: dict) -> bool:
    if rec.get("local"):
        return True
    m = (rec.get("model") or "").lower()
    return any(h in m for h in LOCAL_HINTS)


def collect(days: float | None = None) -> dict:
    cutoff = time.time() - days * 86400 if days else 0
    per_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_day: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in glob.glob(str(ROOT / "runs" / "judge" / "*.calls.jsonl")):
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("ts") or 0
            if ts < cutoff:
                continue
            u = d.get("usage")
            if not isinstance(u, dict):
                u = {}
            model = d.get("model") or "?"
            local = _is_local(d)
            fresh = u.get("input_tokens", 0) or 0
            write = u.get("cache_creation_input_tokens", 0) or 0
            read = u.get("cache_read_input_tokens", 0) or 0
            out = u.get("output_tokens", 0) or 0
            for bucket in (per_model[("local:" if local else "") + model],
                           by_day[time.strftime("%Y-%m-%d", time.localtime(ts))
                                  + ("|local" if local else "|billed")]):
                bucket["calls"] += 1
                bucket["items"] += d.get("n_items") or 0
                bucket["fresh_in"] += fresh
                bucket["cache_write"] += write
                bucket["cache_read"] += read
                bucket["input"] += fresh + write + read
                bucket["output"] += out
                if not local:
                    bucket["billable_equiv"] += fresh + write * WRITE_MULT + read * READ_MULT
    return {"per_model": {k: dict(v) for k, v in per_model.items()},
            "by_day": {k: dict(v) for k, v in sorted(by_day.items())}}


def series(days: float | None = None) -> list[dict]:
    """Plot-ready daily rows, each with a running total.

    Billed and local are kept as separate lines: mixing them would show a
    reassuring flat curve whenever work moved to the local judge, which is
    the opposite of what a spend graph is for.
    """
    got = collect(days)
    rows: dict[str, dict] = {}
    blank = ("billed_input", "billed_output", "billed_equiv", "local_input", "local_output")
    for key, v in got["by_day"].items():
        day, _, kind = key.partition("|")
        r = rows.setdefault(day, {"day": day, "calls": 0, **{k: 0 for k in blank}})
        r["calls"] += v["calls"]
        if kind == "billed":
            r["billed_input"] += v["input"]
            r["billed_output"] += v["output"]
            r["billed_equiv"] += v.get("billable_equiv", 0)
        else:
            r["local_input"] += v["input"]
            r["local_output"] += v["output"]
    out = sorted(rows.values(), key=lambda r: r["day"])
    run = dict.fromkeys(blank, 0.0)
    for r in out:
        for k in blank:
            run[k] += r[k]
            r["cum_" + k] = round(run[k], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    got = collect(a.days)
    if a.json:
        print(json.dumps(got, indent=1))
        return
    print(f"{'judge':<28}{'calls':>7}{'items':>8}{'input':>11}{'output':>10}{'billable-eq':>13}")
    for name, t in sorted(got["per_model"].items(), key=lambda kv: -kv[1].get("output", 0)):
        print(f"{name:<28}{t['calls']:>7}{t.get('items', 0):>8}"
              f"{t['input'] / 1e6:>10.2f}M{t['output'] / 1e6:>9.2f}M"
              f"{t.get('billable_equiv', 0) / 1e6:>12.2f}M")
    bill = sum(v.get("billable_equiv", 0) for v in got["per_model"].values())
    out = sum(v["output"] for k, v in got["per_model"].items() if not k.startswith("local:"))
    print(f"\nbilled input (cache-weighted): {bill / 1e6:.2f}M   billed output: {out / 1e6:.2f}M")


if __name__ == "__main__":
    main()
