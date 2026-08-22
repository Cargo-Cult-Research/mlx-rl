"""Date-flip falsification, re-established for REAL tools.

With a real engine the paper is findable under any stated date, so the test
is no longer "does the behaviour flip" but: does an asserted year LATER than
the stated today make the model deny/decline without checking (the shipped
adapter's original failure), or does it still search and answer? Same
paper, same phrasing "the {year} paper", two stated dates: today BEFORE the
asserted year (asserted > today) and today after publication.

Reports per arm and condition: called / correct / abstain / denial.

    uv run python experimental/arxiv_dateflip_probe.py --n 32 --arm base= --arm arm2=...
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import mlx.core as mx  # noqa: E402
from mlx_lm import load as mlx_load  # noqa: E402

from mlx_rl import machine  # noqa: E402
from mlx_rl.config import TrainConfig  # noqa: E402
from mlx_rl.profiles import get_profile  # noqa: E402
from mlx_rl.tasks.base import Example  # noqa: E402
from mlx_rl.tasks.qa_arxiv import DATE_LINE, HONESTY_SYSTEM, QAArxivTask, _surname  # noqa: E402
from mlx_rl.train import _episode_record, _sample_episodes  # noqa: E402


def build(task, n, seed):
    rng = random.Random(seed)
    rows = [r for r in task._pools["eval"]["unknown"] if r.get("authors") and r["published"] >= "2024-06-01"]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        pub = date.fromisoformat(r["published"])
        year = r["published"][:4]
        first = r["authors"][0]
        q = f'Who are the authors of the {year} paper "{r["title"]}"?'
        for cond, today in (("asserted_after_today", (pub - timedelta(days=200)).isoformat()),
                            ("asserted_before_today", (pub + timedelta(days=200)).isoformat())):
            out.append(Example(
                messages=[{"role": "system", "content": f"{HONESTY_SYSTEM} {DATE_LINE.format(today=today)}"},
                          {"role": "user", "content": q}],
                meta={"id": r["id"], "title": r["title"], "qtype": "authors", "aliases": [first, _surname(first)],
                      "published": r["published"], "today": today, "regime": "post", "band": "unknown",
                      "question": q, "fictional": False, "split": "eval", "cond": cond},
                chat_kwargs={"tools": task.tools}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=f"runs/arxiv-dateflip-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()
    prof = get_profile(a.profile)
    task = QAArxivTask(calib_file="runs/arxiv-calib-20260816/calib-strict.jsonl", backend="web")
    examples = build(task, a.n, a.seed)
    cfg = TrainConfig(model=prof.model, task="qa_arxiv", profile=a.profile, chat_kwargs=dict(prof.chat_kwargs),
                      max_new_tokens=768, max_tool_rounds=4, think_end=prof.think_end,
                      extra_eos=tuple(prof.extra_eos), rollout_batch_size=64)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [(n, str(Path(p).expanduser()) if p else None) for n, _, p in (s.partition("=") for s in a.arm)]
    holder = None if a.no_manage_machine else machine.acquire(38.0, note="date-flip probe")
    summary = {}
    try:
        with (out / "episodes.jsonl").open("w") as f:
            for name, adapter in arms:
                model, tokenizer = mlx_load(prof.model, adapter_path=adapter)
                # never queue rows behind the completion batch (queued cloned caches
                # come back corrupted — see the transfer eval); chunk to batch // k
                groups = []
                per = max(1, cfg.rollout_batch_size // a.k)
                for lo in range(0, len(examples), per):
                    g, _, _ = _sample_episodes(model, tokenizer, examples[lo:lo + per], cfg, task, a.k, 1.0)
                    groups.extend(g)
                    mx.clear_cache()
                fx, frec = [], []
                for ex, g in zip(examples, groups):
                    for ep in g:
                        fx.append(ex)
                        frec.append(_episode_record(tokenizer, ep, None))
                res = task.episode_reward(fx, frec)
                agg = {}
                for ex, rec, r in zip(fx, frec, res):
                    d = agg.setdefault(ex.meta["cond"], {"n": 0, "reward": 0.0, "called": 0.0, "correct": 0.0,
                                                          "abstain": 0.0, "denial": 0.0, "no_reply": 0.0})
                    d["n"] += 1
                    d["reward"] += r.total
                    for k in ("called", "correct", "abstain", "denial", "no_reply"):
                        d[k] += r.parts.get(k, 0.0)
                    f.write(json.dumps({"arm": name, "meta": ex.meta, "visible": rec["visible"],
                                        "tool_calls": rec["tool_calls"], "reward": r.total,
                                        "parts": r.parts}, ensure_ascii=False) + "\n")
                for d in agg.values():
                    n = d.pop("n")
                    for k in d:
                        d[k] /= n
                summary[name] = agg
                for cond, d in agg.items():
                    print(f"== {name:14s} {cond:22s} " + " ".join(f"{k} {v:.2f}" for k, v in d.items()), flush=True)
                del model, tokenizer, groups
                gc.collect()
                mx.clear_cache()
    finally:
        if holder:
            machine.release(holder)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    import os
    import traceback
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        os._exit(code)
