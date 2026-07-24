"""Per-problem difficulty calibration of the DeepScaleR math corpus.

Requested 2026-07-15, after the think-length probe showed the math task is
bimodal (half the eval problems never close thinking at 16k) and the paper
comparison showed why: SAGE-RL trains at 8,192 tokens on models whose MATH-500
baseline is ~4.9k think — while we trained at cap 4096 on uniform DeepScaleR
(AIME/Olympiad-heavy, measured median >=16k). The corpus carries no usable
difficulty label (the `solution` field is empty for the median row), so
difficulty must be measured against OUR model.

This probe samples N problems from the math task's TRAIN pool (never the
held-out eval rows), runs k sampled rollouts each at the paper's 8,192 training
budget, and records per problem: exact think lengths, censoring, pass@k, and
static pathology flags ([asy] figures, multi-blank fill-ins, prose figure
refs). The output is a reusable label file: downstream, a `math_graded` task
variant filters the train pool to the band where GRPO gets variance (thinks
~1.5-6k, pass@k neither 0 nor k) and drops black-hole/figure/multi-blank rows.

Output: runs/<out>/config.json + calib.jsonl (one row per problem) +
summary.json + a band x pass table on stdout.

Usage (memory lease taken by the wrapper; coexists with the host server):
    .venv/bin/python scripts/math_calibrate.py --no-manage-machine \
        --out runs/math-calib-YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from mlx_lm import load as mlx_load

from mlx_rl import machine
from mlx_rl.config import TrainConfig
from mlx_rl.engine import rollout_groups
from mlx_rl.profiles import get_profile
from mlx_rl.tasks import get_task
from mlx_rl.train import _completion_text, _think_close_marker, _visible_reply
from mlx_rl.rollout import encode_prompt

_MULTIBLANK = re.compile(r"_{2,}|\bis\s{3,}|\s{4,}(?:kilometers|and\b)")
_FIGURE = re.compile(
    r"\b(as shown|figure|diagram|graph below|in the picture)\b", re.I)


def flags(problem: str) -> dict:
    return {
        "asy": "[asy]" in problem,
        "multiblank": len(_MULTIBLANK.findall(problem)) >= 2,
        "figure": bool(_FIGURE.search(problem)) and "[asy]" not in problem,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8192,
                    help="the paper's RL training budget")
    ap.add_argument("--out",
                    default=f"runs/math-calib-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    chat_kwargs = {**prof.chat_kwargs, **prof.think_chat_kwargs}
    cfg = TrainConfig(
        model=prof.model, profile=prof.name, think_end=prof.think_end,
        chat_kwargs=chat_kwargs, extra_eos=tuple(prof.extra_eos),
    )
    (out / "config.json").write_text(json.dumps({
        "math_calibrate": True, "model": prof.model, "profile": prof.name,
        "n": a.n, "k": a.k, "seed": a.seed, "budget": a.budget,
        "max_new_tokens": a.budget, "task": "math",
        "chat_kwargs": chat_kwargs,
    }, indent=2) + "\n")

    task = get_task("math")  # split seed 12345 => stable train-pool indices
    rng = random.Random(a.seed)
    idxs = rng.sample(range(len(task._train)), a.n)
    picks = [(i, task._train[i]) for i in idxs]
    examples = [task._example(row) for _, row in picks]

    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(38.0, note="math corpus calibration probe")
    try:
        model, tokenizer = mlx_load(prof.model)
        think_close = _think_close_marker(tokenizer, cfg, task)
        tk = {**getattr(task, "chat_template_kwargs", {}), **chat_kwargs}
        prompts = [encode_prompt(tokenizer, ex.messages, **tk)
                   for ex in examples]

        t0 = time.time()
        groups, _ = rollout_groups(
            model, tokenizer, prompts, a.k, a.budget, 1.0,
            extra_eos=tuple(cfg.extra_eos))
        print(f"sampled: {a.n}x{a.k} in {time.time()-t0:.0f}s", flush=True)
    finally:
        machine.release(holder)

    with (out / "calib.jsonl").open("w") as f:
        for (idx, row), ex, group in zip(picks, examples, groups):
            results = []
            for comp in group:
                text = _completion_text(tokenizer, comp)
                visible, _ = _visible_reply(text, think_close)
                res = task.reward(ex, visible)
                try:
                    tl = comp.tokens.index(cfg.think_end) + 1
                except ValueError:
                    tl = None  # never closed: censored at len
                results.append({
                    "reward": res.total, "think_len": tl,
                    "censored": tl is None, "len": len(comp.tokens),
                    "finish": comp.finish_reason,
                })
            f.write(json.dumps({
                "train_idx": idx, "answer": row["answer"],
                "problem": row["problem"], "flags": flags(row["problem"]),
                "results": results,
            }) + "\n")

    # band x pass table: band = min CLOSED think across the k attempts
    bands = [(0, 1000, "<1k"), (1000, 2000, "1-2k"), (2000, 4000, "2-4k"),
             (4000, 6000, "4-6k"), (6000, 10**9, "6k+")]
    table: dict[tuple, int] = {}
    n_censored_all = 0
    for rec in map(json.loads, (out / "calib.jsonl").read_text().splitlines()):
        closed = [r["think_len"] for r in rec["results"]
                  if r["think_len"] is not None]
        passes = sum(r["reward"] for r in rec["results"])
        if not closed:
            n_censored_all += 1
            continue
        tl = min(closed)
        band = next(lbl for lo, hi, lbl in bands if lo <= tl < hi)
        table[(band, int(passes))] = table.get((band, int(passes)), 0) + 1

    print(f"\nnever-closed (all {a.k} censored @ {a.budget}): "
          f"{n_censored_all}/{a.n} ({n_censored_all/a.n:.0%})")
    hdr = "  ".join(f"pass{p}" for p in range(a.k + 1))
    print(f"{'band':>6}  {hdr}")
    for _, _, lbl in bands:
        cells = "  ".join(f"{table.get((lbl, p), 0):>5}"
                          for p in range(a.k + 1))
        print(f"{lbl:>6}  {cells}")
    (out / "summary.json").write_text(json.dumps({
        "never_closed": n_censored_all,
        "band_pass": {f"{b}|{p}": v for (b, p), v in sorted(table.items())},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
