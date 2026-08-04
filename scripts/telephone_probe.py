"""Day-0 probe for the telephone task: what can the channel carry BEFORE RL?

lifecycle: one-off (archive when the telephone arc concludes)

Measures, with no training:
1. bind sanity — listener decodes the full answer sent in the clear
2. natural-language ceilings — code = quality word only / root only / full
   answer / null: the partial-credit plateau RL must beat with k=1
3. prior-mining capacity — brute-force search over a random sample of single
   vocab tokens: for each of the 48 labels, the best-scoring token found and
   its p_correct. If random search finds high-p tokens, a code EXISTS in the
   frozen listener's prior and RL's job is merely to find it on-policy; if
   not, k=1 may be a dead channel and the game needs k=2.

Usage:
  .venv/bin/python scripts/telephone_probe.py --profile tiny --vocab-sample 512
  .venv/bin/python scripts/telephone_probe.py --profile qwen36 --vocab-sample 256
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from mlx_lm import load

from mlx_rl.profiles import PROFILES
from mlx_rl.tasks.telephone import LABELS, QUALITIES, ROOTS, TelephoneTask, _softmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="tiny", choices=sorted(PROFILES))
    ap.add_argument("--vocab-sample", type=int, default=512)
    ap.add_argument("--score-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    prof = PROFILES[args.profile]
    print(f"loading {prof.model} ...")
    if prof.vlm:
        from mlx_lm.utils import load_tokenizer
        from mlx_vlm import load as vlm_load

        from mlx_rl.models import VLMTextPolicy

        inner, _ = vlm_load(prof.model)
        model, tokenizer = VLMTextPolicy(inner), load_tokenizer(
            __import__("pathlib").Path(prof.model))
    else:
        model, tokenizer = load(prof.model)
    task = TelephoneTask(score_chunk=args.score_chunk)
    task.bind_model(model, tokenizer)  # prints the clear-text sanity line

    results: dict = {"profile": args.profile, "chance": 1 / len(LABELS)}

    # -- 2. natural-language ceilings -------------------------------------
    def sweep(name: str, codes: list[str]) -> None:
        lp = task._score_codes(codes) - task._null_lp
        p = _softmax(lp)
        pc = [float(p[i][i]) for i in range(len(LABELS))]
        pq = [float(sum(p[i][j] for j, (q, _) in enumerate(LABELS)
                        if q == LABELS[i][0])) for i in range(len(LABELS))]
        pr = [float(sum(p[i][j] for j, (_, r) in enumerate(LABELS)
                        if r == LABELS[i][1])) for i in range(len(LABELS))]
        top1 = float(np.mean([int(np.argmax(p[i]) == i) for i in range(len(LABELS))]))
        results[name] = {"p_correct": float(np.mean(pc)),
                         "p_quality": float(np.mean(pq)),
                         "p_root": float(np.mean(pr)), "top1": top1}
        print(f"{name:>12}: p_correct {np.mean(pc):.3f}  "
              f"p_quality {np.mean(pq):.3f}  p_root {np.mean(pr):.3f}  top1 {top1:.3f}")

    sweep("full_answer", [f"{q} {r}" for q, r in LABELS])
    sweep("quality_only", [q for q, _ in LABELS])
    sweep("root_only", [r for _, r in LABELS])
    sweep("null", ["?" for _ in LABELS])

    # -- 3. prior-mining capacity ------------------------------------------
    rng = random.Random(args.seed)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.vocab)
    cand_tokens: list[str] = []
    seen = set()
    while len(cand_tokens) < args.vocab_sample:
        tid = rng.randrange(vocab_size)
        s = tokenizer.decode([tid]).strip()
        # printable, nonempty, and actually round-trips to ONE token
        if not s or s in seen or not s.isprintable():
            continue
        if len(task._encode(s)) != 1:
            continue
        seen.add(s)
        cand_tokens.append(s)

    print(f"scoring {len(cand_tokens)} random single tokens x {len(LABELS)} labels ...")
    t0 = time.time()
    lp = np.zeros((len(cand_tokens), len(LABELS)))
    B = 32  # codes per scoring call, keeps row count = B*48 bounded
    for lo in range(0, len(cand_tokens), B):
        lp[lo:lo + B] = task._score_codes(cand_tokens[lo:lo + B]) - task._null_lp
        done = min(lo + B, len(cand_tokens))
        print(f"  {done}/{len(cand_tokens)}  ({time.time() - t0:.0f}s)", flush=True)
    p = _softmax(lp)  # [V, 48]

    best_idx = p.argmax(axis=0)  # per label, which token scores it highest
    best_p = p.max(axis=0)
    codebook = {f"{q} {r}": {"token": cand_tokens[best_idx[i]],
                             "p_correct": float(best_p[i]),
                             "is_argmax": bool(np.argmax(p[best_idx[i]]) == i)}
                for i, (q, r) in enumerate(LABELS)}
    results["codebook_search"] = {
        "vocab_sampled": len(cand_tokens),
        "mean_best_p": float(best_p.mean()),
        "labels_with_argmax_token": int(sum(v["is_argmax"] for v in codebook.values())),
        "codebook": codebook,
    }
    print(f"\ncodebook over {len(cand_tokens)} random tokens: "
          f"mean best p_correct {best_p.mean():.3f}, "
          f"{sum(v['is_argmax'] for v in codebook.values())}/48 labels have a "
          f"token that makes them the listener's argmax")
    print("\nsample of the found codebook (raw):")
    for k in list(codebook)[::7]:
        v = codebook[k]
        print(f"  {k:>15} <- {v['token']!r:>14}  p={v['p_correct']:.3f}  argmax={v['is_argmax']}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
