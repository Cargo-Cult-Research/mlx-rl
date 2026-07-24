"""How long does the model WANT to think, and does SAGE compress it?

Motivation: a cap-doubling run refuted the truncation hypothesis (at cap
4096 the model grew to fill the room and the learning plateau did not move),
which re-centers the question on the think-length distribution itself. The
earlier oracle probe can't answer it: its budget bound (4096, and SAGE's
step budget censored thinking at exactly 3073 = 64x48+1), its mixture
predates the math task, and rollout_groups never populated think_len for
non-SAGE completions.

This script measures the base model on the trainer's held-out eval stream
(task_kwargs below, seed+100000 => the same held-out problems evaluate()
scores), at a non-binding budget (default 8192), under:

  greedy<B>    temp 0, k=1  — the deployment decode
  sampled<B>   temp 1, k=4  — what vanilla GRPO rollouts see
  sage<B>      m=2 tr=0.5   — the paper's decoder, step budget raised so only
                              the token budget can bind

think_len is exact for every condition: index of the profile's think_end
token (248069 for qwen36) in the completion. Unclosed thinking is CENSORED at
the completion length (a lower bound) and counted separately — quantiles are
reported over closed+censored together, so read them alongside censored%.

The decision rule: if SAGE takes the median from 20k->10k
we are delusional; 6k->3k maybe close; 3k->2k already there.

Output: runs/<out>/config.json + samples.jsonl (dashboard-browsable) +
summary.json + a distribution table on stdout.

Usage:
    .venv/bin/python scripts/think_length.py \
        --out runs/think-length-YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from mlx_lm import load as mlx_load

from mlx_rl import machine
from mlx_rl.config import TrainConfig
from mlx_rl.engine import rollout_groups, sage_completion
from mlx_rl.profiles import get_profile
from mlx_rl.tasks import get_task
from mlx_rl.train import (
    _completion_text,
    _step_delim_ids,
    _think_close_marker,
    _visible_reply,
)
from mlx_rl.rollout import encode_prompt

# the reference mixture the validation runs trained on
TASK_KWARGS = {"weights": {"math": 0.35, "code": 0.35, "arithmetic": 0.3}}


def quantiles(xs: list[int]) -> dict:
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "p25": q(.25), "med": q(.50), "p75": q(.75),
            "p90": q(.90), "max": s[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--n", type=int, default=32,
                    help="held-out problems (32 = the v8 eval set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--sampled-k", type=int, default=4)
    ap.add_argument("--sage-m", type=int, default=2)
    ap.add_argument("--sage-tr", type=float, default=0.5)
    ap.add_argument("--answer-reserve", type=int, default=256)
    ap.add_argument("--tasks", default=None,
                    help="comma list: keep only these tasks (the stream is "
                         "sampled in full first, so problem identity matches "
                         "unfiltered runs)")
    ap.add_argument("--conds", default="greedy,sampled,sage",
                    help="comma subset of greedy,sampled,sage")
    ap.add_argument("--out",
                    default=f"runs/think-length-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()
    want = set(a.conds.split(","))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    chat_kwargs = {**prof.chat_kwargs, **prof.think_chat_kwargs}
    cfg = TrainConfig(
        model=prof.model, profile=prof.name, think_end=prof.think_end,
        chat_kwargs=chat_kwargs, extra_eos=tuple(prof.extra_eos),
    )
    # step budget must exceed the token budget so only tokens bind
    step_tokens = 48
    max_steps = (a.budget - a.answer_reserve) // step_tokens + 2
    (out / "config.json").write_text(json.dumps({
        "think_length": True, "model": prof.model, "profile": prof.name,
        "n": a.n, "seed": a.seed, "budget": a.budget,
        "max_new_tokens": a.budget,  # dashboard cap refline
        "sampled_k": a.sampled_k, "sage_m": a.sage_m, "sage_tr": a.sage_tr,
        "sage_max_reasoning_steps": max_steps,
        "answer_reserve": a.answer_reserve, "task": "mixture",
        "task_kwargs": TASK_KWARGS, "chat_kwargs": chat_kwargs,
    }, indent=2) + "\n")

    task = get_task("mixture", **TASK_KWARGS)
    rng = random.Random(a.seed + 100_000)  # == train.evaluate's stream
    esample = getattr(task, "eval_sample", task.sample)
    examples = [esample(rng) for _ in range(a.n)]
    if a.tasks:
        keep = set(a.tasks.split(","))
        examples = [ex for ex in examples
                    if (getattr(ex, "meta", None) or {}).get("_task") in keep]
        a.n = len(examples)
        print(f"task filter {sorted(keep)}: {a.n} problems", flush=True)

    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(38.0, note="think-length probe (decode-only)")
    try:
        model, tokenizer = mlx_load(prof.model)
        think_close = _think_close_marker(tokenizer, cfg, task)
        eos = set(getattr(tokenizer, "eos_token_ids", None)
                  or [tokenizer.eos_token_id]) | set(cfg.extra_eos)
        step_delim = _step_delim_ids(tokenizer)
        tk = {**getattr(task, "chat_template_kwargs", {}), **chat_kwargs}
        prompts = [encode_prompt(tokenizer, ex.messages, **tk)
                   for ex in examples]

        def grade(comp):
            text = _completion_text(tokenizer, comp)
            visible, closed = _visible_reply(text, think_close)
            res = task.reward(examples[grade.i], visible)
            # exact token-level think length for EVERY condition
            try:
                tl = comp.tokens.index(cfg.think_end) + 1
            except ValueError:
                tl = None  # never closed: censored at len
            return {
                "reward": res.total,
                "parts": {**res.parts, "think_closed": float(closed)},
                "len": len(comp.tokens), "think_len": tl,
                "censored": tl is None,
                "finish": comp.finish_reason, "text": text,
            }

        rows = [[] for _ in range(a.n)]

        def batched(tag, group_size, temperature):
            t0 = time.time()
            groups, _ = rollout_groups(
                model, tokenizer, prompts, group_size, a.budget, temperature,
                extra_eos=tuple(cfg.extra_eos))
            for i, group in enumerate(groups):
                grade.i = i
                for comp in group:
                    rows[i].append({"cond": tag, **grade(comp)})
            print(f"{tag}: {a.n}x{group_size} in {time.time()-t0:.0f}s",
                  flush=True)

        if "greedy" in want:
            batched(f"greedy{a.budget}", 1, 0.0)
        if "sampled" in want:
            batched(f"sampled{a.budget}", a.sampled_k, 1.0)

        t0 = time.time()
        for i, prompt in enumerate(prompts if "sage" in want else []):
            grade.i = i
            comp = sage_completion(
                model, list(prompt), cfg.think_end, eos=eos,
                step_delim=step_delim, m=a.sage_m, tr=a.sage_tr,
                max_new_tokens=a.budget, max_reasoning_steps=max_steps,
                max_step_tokens=256, step_tokens=step_tokens,
                think_temperature=1.0, answer_temperature=1.0,
                answer_reserve=a.answer_reserve)
            rows[i].append({"cond": f"sage{a.budget}", **grade(comp)})
            r = rows[i][-1]
            print(f"sage {i+1}/{a.n}: reward {r['reward']:.2f} "
                  f"think {r['think_len']} len {r['len']} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if "sage" in want:
            print(f"sage{a.budget}: {a.n}x1 in {time.time()-t0:.0f}s",
                  flush=True)
    finally:
        machine.release(holder)

    with (out / "samples.jsonl").open("w") as f:
        for i, (ex, comps) in enumerate(zip(examples, rows)):
            f.write(json.dumps({
                "step": i + 1, "ts": round(time.time(), 1),
                "meta": getattr(ex, "meta", None)
                        or {"_task": type(ex).__name__},
                "completions": [{"sage": c["cond"].startswith("sage"), **c}
                                for c in comps],
            }) + "\n")

    # distributions: cond x (ALL + per task); censored (never-closed) think
    # lengths enter at the completion length as a lower bound
    conds = [f"{c}{a.budget}" for c in ("greedy", "sampled", "sage")
             if c in want]
    cells: dict[tuple, list] = {}
    for ex, comps in zip(examples, rows):
        t = (getattr(ex, "meta", None) or {}).get("_task", "?")
        for c in comps:
            for key in ((c["cond"], "ALL"), (c["cond"], t)):
                cells.setdefault(key, []).append(c)

    summary = {}
    print(f"\n{'cond':>12} {'task':>11} {'n':>4} {'med':>6} {'p25':>6} "
          f"{'p75':>6} {'p90':>6} {'max':>6} {'cens%':>6} {'acc':>6}")
    for cond in conds:
        for taskname in ("ALL", "arithmetic", "code", "math"):
            cs = cells.get((cond, taskname))
            if not cs:
                continue
            tl = [c["think_len"] if c["think_len"] is not None else c["len"]
                  for c in cs]
            d = quantiles(tl)
            d["censored"] = round(sum(c["censored"] for c in cs) / len(cs), 4)
            d["acc"] = round(sum(c["reward"] for c in cs) / len(cs), 4)
            d["capped"] = round(
                sum(c["finish"] == "length" for c in cs) / len(cs), 4)
            summary.setdefault(cond, {})[taskname] = d
            print(f"{cond:>12} {taskname:>11} {d['n']:>4} {d['med']:>6} "
                  f"{d['p25']:>6} {d['p75']:>6} {d['p90']:>6} {d['max']:>6} "
                  f"{d['censored']:>6.0%} {d['acc']:>6.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
