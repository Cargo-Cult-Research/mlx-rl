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
import os
import re
import subprocess
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

# Swap guard: a healthy sweep at batch-prompts 10 runs at ~0 swap; the
# 2026-08-13 batch-100 probe oversubscribed unified memory, thrashed 25 GB of
# swap, and the killed process's kernel-exit teardown took the whole box down.
# Abort LOUDLY (journal + Telegram) long before that point.
SWAP_ABORT_GB = 10.0


def swap_used_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                         capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def fail_loud(msg: str) -> None:
    """Journal + best-effort Telegram — a guard abort must reach the human."""
    print(f"SWEEP GUARD ABORT: {msg}", flush=True)
    note = os.path.expanduser("~/code/housekeeping/note.sh")
    if os.path.exists(note):
        subprocess.run(["bash", note, f"difficulty sweep GUARD ABORT: {msg}"])
    env_path = os.path.expanduser("~/code/housekeeping/.env")
    try:
        env = dict(
            line.strip().split("=", 1)
            for line in open(env_path)
            if "=" in line and not line.startswith("#"))
        subprocess.run(
            ["curl", "-s", "-m", "10",
             f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendMessage",
             "-d", f"chat_id={env['TELEGRAM_USER_ID']}",
             "--data-urlencode", f"text=⛔ difficulty sweep aborted: {msg}"],
            capture_output=True)
    except Exception as e:  # noqa: BLE001 — guard must not crash on notify
        print(f"(telegram notify failed: {e})", flush=True)


def load_done(out: Path, temp: float, model: str) -> set:
    """Task ids already swept at this temperature BY THIS MODEL.

    The model is part of the key, not just the temperature: a two-model duel
    that points both legs at one --out would otherwise have the second leg
    read the first leg's rows as its own, report "0/200 to do", and exit
    having generated nothing. Separate --out files per model are still the
    recommended layout; this makes the shared-file case fail safe instead of
    silently producing a half-empty comparison."""
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from a killed run; will be redone
            # Rows written before 2026-08-14 carry no "model" key; treat them
            # as belonging to whoever is asking, preserving old resume behaviour.
            if row.get("temperature") == temp and row.get("model", model) == model:
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
    ap.add_argument("--sample", type=int, default=0,
                    help="seeded random N-problem subset (pilots; spans "
                         "sources, unlike --limit's file-order prefix)")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--save-texts", action="store_true",
                    help="store full completion texts (large; MBPP-scale only)")
    ap.add_argument("--required-gb", type=float, default=38.0)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="quantize rollout KV caches (8 = half footprint); "
                         "default fp16. Labels change slightly — see the "
                         "kv8 fidelity A/B before using for a full sweep")
    ap.add_argument("--max-problems", type=int, default=0,
                    help="process at most N not-yet-done problems, then exit. "
                         "Lets two models alternate over the same seeded list "
                         "in aligned chunks (resume skips what each already "
                         "did), so a duel reports pairs from the first chunk "
                         "instead of after the whole first leg")
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
    if args.sample:
        import random
        examples = random.Random(args.sample_seed).sample(
            examples, min(args.sample, len(examples)))
    done = load_done(out, args.temperature, Path(args.model).name)
    todo = [e for e in examples if e.meta["task_id"] not in done]
    remaining = len(todo)
    if args.max_problems:
        todo = todo[: args.max_problems]
    chunked = f" [chunk of {len(todo)}; {remaining - len(todo)} left after]" \
        if args.max_problems and remaining > len(todo) else ""
    print(f"sweep: {args.task} pass@{args.k} T={args.temperature} "
          f"cap={args.max_new_tokens} model={Path(args.model).name} — "
          f"{len(todo)}/{len(examples)} to do "
          f"({len(done)} already in {out.name}){chunked}", flush=True)
    if not todo:
        return

    holder = None
    if not args.no_manage_machine:
        holder = machine.acquire(
            args.required_gb, wait_s=0, block="experiments",
            note=f"difficulty sweep {args.task} pass@{args.k} T={args.temperature}")
    try:
        assert_fits(args.required_gb)
        if (s := swap_used_gb()) > SWAP_ABORT_GB / 2:
            fail_loud(f"swap already {s:.1f} GB before model load — "
                      f"machine not in a fit state, refusing to start")
            sys.exit(1)
        t_load = time.time()
        model, tokenizer = mlx_load(args.model)
        # Logged because duels swap models every batch and trade load time for
        # rollout parallelism — the trade is only sound while this stays small
        # relative to a batch (measure, don't assume).
        print(f"model loaded in {time.time() - t_load:.1f}s: "
              f"{Path(args.model).name}", flush=True)
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
            t_batch = time.time()
            groups, stats = rollout_groups(
                model, tokenizer, prompts,
                group_size=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                completion_batch_size=args.batch_prompts * args.k,
                kv_bits=args.kv_bits,
            )
            batch_wall = time.time() - t_batch
            # Sequences in a batch decode in lockstep and the batch ends when
            # the longest finishes, so a problem's own wall time is well
            # defined as (its longest sample) x (seconds per step) — see
            # per_problem_s() in duel_report.py. Recording the raw terms
            # rather than a derived number keeps the batch context auditable.
            batch_max_len = max(max(len(c.tokens) for c in g) for g in groups)
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
                        "kv_bits": args.kv_bits,
                        "model": Path(args.model).name,
                        "batch_wall_s": round(batch_wall, 2),
                        "batch_size": len(chunk),
                        "batch_max_len": batch_max_len,
                        "gen_tps": round(stats.generation_tps, 1),
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
            swap = swap_used_gb()
            print(f"[{done_n}/{len(todo)}] n_pass {npass}  "
                  f"decode {stats.generation_tps:.0f} t/s  "
                  f"peak {stats.peak_memory:.1f} GB  swap {swap:.1f} GB  "
                  f"elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)
            if swap > SWAP_ABORT_GB:
                fail_loud(
                    f"swap {swap:.1f} GB > {SWAP_ABORT_GB} GB after batch "
                    f"{done_n}/{len(todo)} ({args.task} T={args.temperature}) "
                    f"— stopping before the box thrashes; rows are durable, "
                    f"rerun resumes here")
                sys.exit(2)
        graded.shutdown()
    finally:
        machine.release(holder)
    print(f"sweep complete: {out}", flush=True)


if __name__ == "__main__":
    main()
