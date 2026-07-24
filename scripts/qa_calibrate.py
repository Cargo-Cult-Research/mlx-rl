"""Knowledge calibration probe for the qa_abstain task.

Measures the BASE model's per-question knowledge on the qa_abstain train
pool: k sampled completions per question under a forced-answer prompt (no
abstain option — we want p(correct-if-answering), the quantity the
answer/abstain policy must be calibrated against). Writes calib.jsonl
({"qid", "question", "pass_rate", "n", "samples"}) consumable by the task's
`calib_file` kwarg, plus a band histogram:

    known      pass_rate >= 0.8   (model reliably knows it)
    uncertain  0 < pass_rate < 0.8  (where calibration signal lives)
    unknown    pass_rate == 0     (model reliably doesn't)

GRPO groups drawn uniformly from an unprobed pool are dominated by
known/unknown questions whose groups have no decision variance — the
curriculum (`band_mix`) built on this file is what keeps gradient flowing.

Usage:
    .venv/bin/python scripts/qa_calibrate.py --n 2000 --k 8 \
        --out runs/qa-calib-YYYYMMDD
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from mlx_lm import load as mlx_load

from mlx_rl import machine
from mlx_rl.engine import rollout_groups
from mlx_rl.profiles import get_profile
from mlx_rl.rollout import encode_prompt
from mlx_rl.tasks import get_task
from mlx_rl.tasks.qa_abstain import _band, grade, parse_reply
from mlx_rl.train import _completion_text

FORCED_PROMPT = (
    "Answer this question with a short factual answer, wrapped in answer "
    "tags like <answer>Paris</answer>. Give your best guess even if you "
    "are not sure.\n\nQuestion: {q}"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=2000, help="questions to probe")
    ap.add_argument("--k", type=int, default=8, help="samples per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-questions", type=int, default=32,
                    help="questions per rollout_groups call")
    ap.add_argument("--out", default=f"runs/qa-calib-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    (out / "config.json").write_text(json.dumps({
        "qa_calibrate": True, "model": prof.model, "profile": prof.name,
        "dataset": a.dataset, "n": a.n, "k": a.k, "seed": a.seed,
        "max_new_tokens": a.max_new_tokens, "chat_kwargs": prof.chat_kwargs,
    }, indent=2) + "\n")

    task = get_task("qa_abstain", dataset=a.dataset)  # split seed 12345
    rng = random.Random(a.seed)
    rows = rng.sample(task._train, min(a.n, len(task._train)))

    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(38.0, note="qa_abstain calibration probe")
    try:
        model, tokenizer = mlx_load(prof.model)
        t0 = time.time()
        with (out / "calib.jsonl").open("w") as f:
            for lo in range(0, len(rows), a.batch_questions):
                chunk = rows[lo:lo + a.batch_questions]
                prompts = [encode_prompt(
                    tokenizer,
                    [{"role": "user",
                      "content": FORCED_PROMPT.format(q=r["question"])}],
                    **prof.chat_kwargs) for r in chunk]
                groups, _ = rollout_groups(
                    model, tokenizer, prompts, a.k, a.max_new_tokens, 1.0,
                    extra_eos=tuple(prof.extra_eos))
                for row, group in zip(chunk, groups):
                    samples = []
                    for comp in group:
                        kind, val = parse_reply(_completion_text(tokenizer, comp))
                        ok = kind == "answer" and grade(val, row["aliases"])
                        samples.append({"kind": kind, "value": val,
                                        "correct": ok})
                    rate = sum(s["correct"] for s in samples) / len(samples)
                    f.write(json.dumps({
                        "qid": row["qid"], "question": row["question"],
                        "aliases": row["aliases"][:8],
                        "pass_rate": rate, "n": len(samples),
                        "samples": samples,
                    }) + "\n")
                done = min(lo + a.batch_questions, len(rows))
                print(f"{done}/{len(rows)} questions "
                      f"({done * a.k / (time.time() - t0):.1f} compl/s)",
                      flush=True)
    finally:
        machine.release(holder)

    # Band histogram + raw examples per band (numbers hide bugs; show data).
    recs = [json.loads(l) for l in (out / "calib.jsonl").read_text().splitlines()]
    bands: dict[str, list] = {"known": [], "uncertain": [], "unknown": []}
    for r in recs:
        bands[_band(r["pass_rate"])].append(r)
    print(f"\n{a.dataset}: {len(recs)} questions x {a.k} samples")
    for name, rs in bands.items():
        print(f"  {name:9s} {len(rs):5d}  ({len(rs) / len(recs):.0%})")
        for r in rs[:2]:
            got = [s["value"] for s in r["samples"]][:3]
            print(f"    e.g. [{r['pass_rate']:.2f}] {r['question'][:70]}"
                  f" -> {got}")
    summary = {name: len(rs) for name, rs in bands.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
