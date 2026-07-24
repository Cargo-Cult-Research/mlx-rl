"""The SAGE oracle: is there a there there, before any RL?

Honesty check requested 2026-07-12. The training runs so far decoded under a
BINDING budget (1024: the confidence gate never fired on arithmetic/code —
we force-closed the model's thinking early, then trained on the result,
which can make the model WORSE, not better). This script measures the base
model (no adapter) on the SAME held-out problem stream the A/B eval uses
(seed+100000), at a non-binding 4096 budget, under four decode conditions:

  greedy1024   temp 0, cap 1024 — quantifies what the old cap cost us
  greedy4096   temp 0, cap 4096 — the deployment default, unconstrained
  sampled4096  temp 1, k=4      — what vanilla GRPO rollouts see
  sage4096     m=2 tr=0.5       — the paper's decoder, unconstrained

Grading is identical to training (visible reply after the final think-close;
unclosed think = 0). If sage4096 does not beat sampled/greedy here, RL that
distills SAGE rollouts has nothing to distill — stop and rethink before
burning more compute.

Output: runs/<out>/config.json + samples.jsonl (dashboard-browsable: one
"step" per problem, completions = all conditions) + summary.json + a table
on stdout.

Usage:
    .venv/bin/python scripts/oracle_sage.py --out runs/oracle-sage-YYYYMMDD
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
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

TASK_KWARGS = {
    "weights": {"code": 0.5, "arithmetic": 0.25, "toolformat": 0.25},
    "arithmetic": {"n_operands": 5, "format_reward": 0.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--n", type=int, default=32, help="held-out problems "
                    "(32 = the A/B eval set, same seed stream)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--sampled-k", type=int, default=4)
    ap.add_argument("--sage-m", type=int, default=2)
    ap.add_argument("--sage-tr", type=float, default=0.5)
    ap.add_argument("--answer-reserve", type=int, default=256)
    ap.add_argument("--out", default=f"runs/oracle-sage-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    # thinking ON — the whole question is about the think phase
    chat_kwargs = {**prof.chat_kwargs, **prof.think_chat_kwargs}
    cfg = TrainConfig(  # minimal cfg so the train.py grading helpers apply
        model=prof.model, profile=prof.name, think_end=prof.think_end,
        chat_kwargs=chat_kwargs, extra_eos=tuple(prof.extra_eos),
    )
    (out / "config.json").write_text(json.dumps({
        "oracle": True, "model": prof.model, "profile": prof.name,
        "n": a.n, "seed": a.seed, "budget": a.budget,
        "max_new_tokens": a.budget,  # dashboard cap refline
        "sampled_k": a.sampled_k, "sage_m": a.sage_m, "sage_tr": a.sage_tr,
        "answer_reserve": a.answer_reserve, "task": "mixture",
        "task_kwargs": TASK_KWARGS, "chat_kwargs": chat_kwargs,
    }, indent=2) + "\n")

    task = get_task("mixture", **TASK_KWARGS)
    rng = random.Random(a.seed + 100_000)  # == train.evaluate's stream
    esample = getattr(task, "eval_sample", task.sample)
    examples = [esample(rng) for _ in range(a.n)]

    holder = None
    if not a.no_manage_machine:
        # generation only: weights + KV, no backward
        holder = machine.acquire(38.0, note="SAGE oracle (decode-only, no RL)")
    try:
        model, tokenizer = mlx_load(prof.model)
        think_close = _think_close_marker(tokenizer, cfg, task)
        eos = set(getattr(tokenizer, "eos_token_ids", None)
                  or [tokenizer.eos_token_id]) | set(cfg.extra_eos)
        step_delim = _step_delim_ids(tokenizer)
        tk = {**getattr(task, "chat_template_kwargs", {}), **chat_kwargs}
        prompts = [encode_prompt(tokenizer, ex.messages, **tk) for ex in examples]

        def grade(comp):
            text = _completion_text(tokenizer, comp)
            visible, closed = _visible_reply(text, think_close)
            res = task.reward(examples[grade.i], visible)
            return {
                "reward": res.total, "parts": {**res.parts,
                                               "think_closed": float(closed)},
                "len": len(comp.tokens), "think_len": comp.think_len,
                "finish": comp.finish_reason, "text": text,
            }

        rows = [[] for _ in range(a.n)]  # per-problem completion dicts

        def batched(tag, group_size, temperature, cap):
            t0 = time.time()
            groups, _ = rollout_groups(
                model, tokenizer, prompts, group_size, cap, temperature,
                extra_eos=tuple(cfg.extra_eos))
            for i, group in enumerate(groups):
                grade.i = i
                for comp in group:
                    rows[i].append({"cond": tag, **grade(comp)})
            print(f"{tag}: {a.n}x{group_size} in {time.time()-t0:.0f}s",
                  flush=True)

        batched("greedy1024", 1, 0.0, 1024)
        batched(f"greedy{a.budget}", 1, 0.0, a.budget)
        batched(f"sampled{a.budget}", a.sampled_k, 1.0, a.budget)

        t0 = time.time()
        for i, prompt in enumerate(prompts):
            grade.i = i
            comp = sage_completion(
                model, list(prompt), cfg.think_end, eos=eos,
                step_delim=step_delim, m=a.sage_m, tr=a.sage_tr,
                max_new_tokens=a.budget, max_reasoning_steps=64,
                max_step_tokens=256, step_tokens=48,
                think_temperature=1.0, answer_temperature=1.0,
                answer_reserve=a.answer_reserve)
            rows[i].append({"cond": f"sage{a.budget}", **grade(comp)})
            r = rows[i][-1]
            print(f"sage {i+1}/{a.n}: reward {r['reward']:.2f} "
                  f"think {r['think_len']} len {r['len']} "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)
        print(f"sage{a.budget}: {a.n}x1 in {time.time()-t0:.0f}s", flush=True)
    finally:
        machine.release(holder)

    # dashboard-browsable dump: one "step" per problem
    with (out / "samples.jsonl").open("w") as f:
        for i, (ex, comps) in enumerate(zip(examples, rows)):
            f.write(json.dumps({
                "step": i + 1, "ts": round(time.time(), 1),
                "meta": getattr(ex, "meta", None) or {"_task": type(ex).__name__},
                "completions": [{"sage": c["cond"].startswith("sage"), **c}
                                for c in comps],
            }) + "\n")

    # summary: per condition, overall + the SAGE natural-termination rate.
    # The think phase is bounded by BOTH the token budget and the reasoning-
    # step budget (max_reasoning_steps × step_tokens = 64×48 = 3072 here);
    # a think_len pinned at the step bound is a forced close, not the gate.
    think_budget = min(a.budget - a.answer_reserve, 64 * 48)
    summary = {}
    conds = ["greedy1024", f"greedy{a.budget}", f"sampled{a.budget}", f"sage{a.budget}"]
    for cond in conds:
        cs = [c for comps in rows for c in comps if c["cond"] == cond]
        n = len(cs)
        summary[cond] = {
            "n": n,
            "acc": round(sum(c["reward"] for c in cs) / n, 4),
            "think_closed": round(
                sum(c["parts"]["think_closed"] for c in cs) / n, 4),
            "mean_len": round(sum(c["len"] for c in cs) / n, 1),
            "capped": round(
                sum(c["finish"] == "length" for c in cs) / n, 4),
        }
    sage_cs = [c for comps in rows for c in comps if c["cond"] == f"sage{a.budget}"]
    summary[f"sage{a.budget}"]["mean_think_len"] = round(
        sum(c["think_len"] or 0 for c in sage_cs) / len(sage_cs), 1)
    summary[f"sage{a.budget}"]["gate_fired_naturally"] = round(
        sum((c["think_len"] or 0) < think_budget for c in sage_cs)
        / len(sage_cs), 4)
    # per-task accuracy — a flat task (same acc in every condition) is a
    # capability/format problem, not a decoding one
    per_task: dict[str, dict[str, list]] = {}
    for ex, comps in zip(examples, rows):
        t = (getattr(ex, "meta", None) or {}).get("_task", "?")
        for c in comps:
            per_task.setdefault(c["cond"], {}).setdefault(t, []).append(
                c["reward"])
    for cond, tasks in per_task.items():
        summary[cond]["per_task"] = {
            t: round(sum(v) / len(v), 4) for t, v in sorted(tasks.items())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'cond':>12} {'acc':>6} {'closed':>7} {'len':>7} {'capped':>7}")
    for cond in conds:
        s = summary[cond]
        print(f"{cond:>12} {s['acc']:>6.3f} {s['think_closed']:>7.2f} "
              f"{s['mean_len']:>7.0f} {s['capped']:>7.2f}")
    s = summary[f"sage{a.budget}"]
    print(f"sage: mean think {s['mean_think_len']:.0f}, gate fired naturally "
          f"on {s['gate_fired_naturally']:.0%}")


if __name__ == "__main__":
    main()
