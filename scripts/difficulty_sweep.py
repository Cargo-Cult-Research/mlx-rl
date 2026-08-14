#!/usr/bin/env python3
# lifecycle: core
"""Base-model pass@k difficulty sweep over a task's FULL dataset.

Rollouts are the expensive resource on this machine; LoRA gradients are not.
This script spends them once, up front: k sampled completions per problem at
the RL operating point (training temperature, training cap), graded by the
task's own reward — yielding a per-problem difficulty label n_pass ∈ [0, k]
that every future run can reuse for curriculum slicing (train on the mixed
band, step up/down a notch as the policy improves) without ever re-measuring.

Requires the task to expose all_examples() (every problem once, split-tagged).
Output: one JSON line per problem, append-only and resumable — rerunning
skips (task_id, temperature) pairs already present in --out.

Grading runs the task's reward, which for code tasks EXECUTES model output in
a subprocess — same non-sandboxed caveat as training (see tasks/code.py).

Run:  .venv/bin/python scripts/difficulty_sweep.py --task code --k 5 \
          --temperature 1.0 --max-new-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_lm import load as mlx_load

from mlx_rl import machine
from mlx_rl.engine import rollout_groups
from mlx_rl.memory import assert_fits
from mlx_rl.rollout import encode_prompt
from mlx_rl import tasks as task_mod

DEFAULT_MODEL = str(Path.home() / "models/mlx/Qwen3.6-35B-A3B-4bit")
PASS_EPS = 0.999  # reward >= this counts as a pass (binary-reward tasks)


def load_done(out: Path, temp: float) -> set:
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from a killed run; will be redone
            if row.get("temperature") == temp:
                done.add(row["task_id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="code")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-prompts", type=int, default=10,
                    help="problems per rollout batch (rows = this * k)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None,
                    help="default runs/sweeps/<task>-pass@<k>.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only")
    ap.add_argument("--save-texts", action="store_true",
                    help="store full completion texts (large; MBPP-scale only)")
    ap.add_argument("--required-gb", type=float, default=38.0)
    ap.add_argument("--no-manage-machine", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else (
        root / "runs" / "sweeps" / f"{args.task}-pass@{args.k}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    task = task_mod.get_task(args.task)
    if not hasattr(task, "all_examples"):
        sys.exit(f"task '{args.task}' has no all_examples() — add one first")
    examples = task.all_examples()
    if args.limit:
        examples = examples[: args.limit]
    done = load_done(out, args.temperature)
    todo = [e for e in examples if e.meta["task_id"] not in done]
    print(f"sweep: {args.task} pass@{args.k} T={args.temperature} "
          f"cap={args.max_new_tokens} — {len(todo)}/{len(examples)} to do "
          f"({len(done)} already in {out.name})", flush=True)
    if not todo:
        return

    holder = None
    if not args.no_manage_machine:
        holder = machine.acquire(
            args.required_gb, wait_s=0, block="experiments",
            note=f"difficulty sweep {args.task} pass@{args.k} T={args.temperature}")
    try:
        assert_fits(args.required_gb)
        model, tokenizer = mlx_load(args.model)
        chat_kwargs = dict(getattr(task, "chat_template_kwargs", {}) or {})
        graded = ThreadPoolExecutor(max_workers=8)
        t0, done_n, tok_total = time.time(), 0, 0

        for i in range(0, len(todo), args.batch_prompts):
            chunk = todo[i : i + args.batch_prompts]
            prompts = [
                encode_prompt(tokenizer, e.messages,
                              enable_thinking=True, **chat_kwargs)
                for e in chunk
            ]
            groups, stats = rollout_groups(
                model, tokenizer, prompts,
                group_size=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                completion_batch_size=args.batch_prompts * args.k,
            )
            texts = [[tokenizer.decode(c.tokens) for c in g] for g in groups]
            rewards = list(graded.map(
                lambda ec: [task.reward(ec[0], t).total for t in ec[1]],
                zip(chunk, texts)))
            with out.open("a") as fh:
                for e, g, txts, rs in zip(chunk, groups, texts, rewards):
                    row = {
                        "task": args.task,
                        "task_id": e.meta["task_id"],
                        "split": e.meta.get("split"),
                        "temperature": args.temperature,
                        "k": args.k,
                        "n_pass": sum(r >= PASS_EPS for r in rs),
                        "rewards": [round(r, 3) for r in rs],
                        "lens": [len(c.tokens) for c in g],
                        "finishes": [c.finish_reason for c in g],
                        "cap": args.max_new_tokens,
                        "model": Path(args.model).name,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    if args.save_texts:
                        row["texts"] = txts
                    fh.write(json.dumps(row) + "\n")
            done_n += len(chunk)
            tok_total += stats.generation_tokens
            el = time.time() - t0
            eta = el / done_n * (len(todo) - done_n)
            npass = [sum(r >= PASS_EPS for r in rs) for rs in rewards]
            print(f"[{done_n}/{len(todo)}] n_pass {npass}  "
                  f"decode {stats.generation_tps:.0f} t/s  "
                  f"peak {stats.peak_memory:.1f} GB  "
                  f"elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)
        graded.shutdown()
    finally:
        machine.release(holder)
    print(f"sweep complete: {out}", flush=True)


if __name__ == "__main__":
    main()
