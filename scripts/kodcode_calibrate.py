"""Difficulty calibration probe for the kodcode task's train pool.

Measures the BASE model's per-problem pass@k on a sample of the KodCode
train pool: k completions at temperature 1.0 (the training distribution),
graded by the task's sandboxed pytest reward. Writes calib.jsonl
({"question_id", "subset", "pass_rate", "n"}) plus a band histogram:

    saturated  pass_rate == 1    (group gives no GRPO signal)
    learnable  0 < pass_rate < 1 (active groups live here)
    unsolved   pass_rate == 0    (no signal either — too hard for now)

The headline number is the expected ACTIVE-GROUP FRACTION at the training
group size: the share of uniformly-drawn problems whose groups would carry
gradient. If it's low, filter the train pool to the learnable band before
burning a long run.

Grading is threaded (each grade is a sandboxed subprocess; the GIL is
released while waiting), generation is batched via rollout_groups.

Usage:
    .venv/bin/python scripts/kodcode_calibrate.py --n 300 --k 8 \
        --out runs/kodcode-calib-YYYYMMDD
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlx_lm import load as mlx_load

from mlx_rl.engine import rollout_groups
from mlx_rl.rollout import encode_prompt
from mlx_rl.tasks import get_task
from mlx_rl.train import _completion_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=300, help="problems to probe")
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--batch-problems", type=int, default=16,
                    help="problems per rollout_groups call")
    ap.add_argument("--difficulties", default="easy")
    ap.add_argument("--grade-workers", type=int, default=8)
    ap.add_argument("--out",
                    default=f"runs/kodcode-calib-{time.strftime('%Y%m%d')}")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "kodcode_calibrate": True, "model": a.model, "n": a.n, "k": a.k,
        "seed": a.seed, "max_new_tokens": a.max_new_tokens,
        "difficulties": a.difficulties,
    }, indent=2) + "\n")

    task = get_task("kodcode", difficulties=a.difficulties)
    rng = random.Random(a.seed)
    rows = rng.sample(task._train, min(a.n, len(task._train)))

    model, tokenizer = mlx_load(a.model)
    t0 = time.time()
    # Pipelined: grading futures are submitted and NOT awaited inside the
    # loop, so the sandboxed pytest subprocesses run on CPU cores while the
    # next batch generates on the GPU. Blocking per batch would idle the
    # graders during generation (~2/3 of wall time) and the GPU during
    # grading.
    pending: list[tuple[dict, list]] = []
    with ThreadPoolExecutor(a.grade_workers) as pool:
        for lo in range(0, len(rows), a.batch_problems):
            chunk = rows[lo:lo + a.batch_problems]
            examples = [task._train_example(r) for r in chunk]
            prompts = [encode_prompt(tokenizer, ex.messages)
                       for ex in examples]
            groups, _ = rollout_groups(
                model, tokenizer, prompts, a.k, a.max_new_tokens, 1.0)
            for row, ex, group in zip(chunk, examples, groups):
                futs = [pool.submit(task.reward, ex,
                                    _completion_text(tokenizer, comp))
                        for comp in group]
                pending.append((row, futs))
            done = min(lo + a.batch_problems, len(rows))
            print(f"generated {done}/{len(rows)} problems "
                  f"({done * a.k / (time.time() - t0):.1f} compl/s)",
                  flush=True)
        with (out / "calib.jsonl").open("w") as f:
            for row, futs in pending:
                h = sum(int(fut.result().total == 1.0) for fut in futs)
                f.write(json.dumps({
                    "question_id": row["question_id"],
                    "subset": row["subset"],
                    "pass_rate": h / a.k, "n": a.k,
                }) + "\n")
        print(f"graded {len(pending)} problems "
              f"(total {time.time() - t0:.0f}s)", flush=True)

    recs = [json.loads(l) for l in (out / "calib.jsonl").read_text().splitlines()]
    bands = {"saturated": [r for r in recs if r["pass_rate"] == 1.0],
             "learnable": [r for r in recs if 0 < r["pass_rate"] < 1.0],
             "unsolved": [r for r in recs if r["pass_rate"] == 0.0]}
    print(f"\n{len(recs)} problems x {a.k} samples "
          f"(difficulties={a.difficulties})")
    for name, rs in bands.items():
        print(f"  {name:9s} {len(rs):5d}  ({len(rs) / len(recs):.0%})")
    mean_rate = sum(r["pass_rate"] for r in recs) / len(recs)
    active = len(bands["learnable"]) / len(recs)
    print(f"  mean pass@1 (t=1.0): {mean_rate:.3f}")
    print(f"  expected active-group fraction @ group_size={a.k}: {active:.0%}")
    per_subset: dict[str, list] = {}
    for r in recs:
        per_subset.setdefault(r["subset"], []).append(r["pass_rate"])
    print("  by subset (mean pass rate / n):")
    for s, rates in sorted(per_subset.items()):
        print(f"    {s:15s} {sum(rates) / len(rates):.2f} / {len(rates)}")
    (out / "summary.json").write_text(json.dumps(
        {name: len(rs) for name, rs in bands.items()}
        | {"mean_pass_rate": mean_rate, "active_fraction": active},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
