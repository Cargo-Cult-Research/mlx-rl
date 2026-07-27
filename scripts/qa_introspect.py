"""Mode-detector signal capture: is "lookup vs story" readable at
generation time?

For each question (qa_abstain pool, forced-answer prompt) this captures,
per item:

    greedy answer + external correctness label   (gold aliases — the label
                                                  every signal is judged
                                                  against; NOT dispersion
                                                  bands, which would score
                                                  self-consistency against
                                                  itself)
    S1  mean/min token logprob over the answer value (teacher-forced rescore)
    S2  next-token entropy at the answer-commitment position
    S3  top1-top2 logit margin at the same position
    S4  k-sample agreement + distinct-answer count (temp-1 resamples)
    S7 (data)  hidden states at the last prompt token and at the
               answer-commitment token -> signals.npz, for offline linear
               probes

The commitment position is the first token of the answer VALUE (just past
"<answer>"), located by incremental decode — the point where the model
commits to a specific referent.

Analysis mode computes per-signal AUROC against correctness:

    .venv/bin/python scripts/qa_introspect.py --profile tiny --n 300 \
        --out runs/introspect-tiny
    .venv/bin/python scripts/qa_introspect.py --analyze runs/introspect-tiny

Curvature signals (S5 perturbation flip-rate, S6 decision radius) live in
the jlens repo with the Jacobian machinery. Plan: jlens/docs/mode-detector.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

SIGNALS = ("mean_lp", "min_lp", "value_mean_lp", "neg_entropy_commit",
           "margin_commit", "agreement", "neg_distinct")


def value_token_span(tokenizer, tokens: list[int]) -> tuple[int, int] | None:
    """[start, end) token indices of the answer value (text between
    "<answer>" and "</answer>"), via incremental decode. None if the tag
    structure never appears."""
    text = ""
    start = None
    for i, t in enumerate(tokens):
        new = tokenizer.decode(tokens[: i + 1])
        if start is None and "<answer>" in new:
            start = i  # first token that extends past the open tag
            continue
        if start is not None and "</answer>" in new:
            return (start, i) if i > start else None
        text = new
    return None


def greedy_generate(model, tokenizer, prompt_ids: list[int],
                    max_new: int, extra_eos: tuple[int, ...]) -> list[int]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    eos = set(tokenizer.eos_token_ids) | set(extra_eos)
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids]), cache=cache)
    out: list[int] = []
    tok = int(mx.argmax(logits[0, -1]).item())
    for _ in range(max_new):
        out.append(tok)
        if tok in eos:
            break
        logits = model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]).item())
    return out


def rescore(model, prompt_ids: list[int], completion: list[int]):
    """Teacher-forced pass over prompt+completion. Returns per-completion-
    position (logprob of realized token, entropy, top1-top2 margin) and
    hidden states (last prompt position + every completion position)."""
    import mlx.core as mx

    ids = mx.array([prompt_ids + completion])
    h = model.model(ids)  # final-norm hidden states [1, L, d]
    if hasattr(model, "lm_head"):
        logits = model.lm_head(h)
    else:  # tied embeddings
        logits = model.model.embed_tokens.as_linear(h)
    logits = logits[0].astype(mx.float32)
    p0 = len(prompt_ids)
    # position j predicts token j+1: completion token i realized at j=p0-1+i
    sel = logits[p0 - 1: p0 - 1 + len(completion)]
    logp = sel - mx.logsumexp(sel, axis=-1, keepdims=True)
    realized = logp[mx.arange(len(completion)), mx.array(completion)]
    probs = mx.exp(logp)
    entropy = -mx.sum(probs * logp, axis=-1)
    top2 = mx.sort(sel, axis=-1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    h_prompt = h[0, p0 - 1].astype(mx.float16)
    h_completion = h[0, p0 - 1: p0 - 1 + len(completion)].astype(mx.float16)
    mx.eval(realized, entropy, margin, h_prompt, h_completion)
    import numpy as np
    return (np.array(realized), np.array(entropy), np.array(margin),
            np.array(h_prompt), np.array(h_completion))


def capture(a) -> None:
    import numpy as np

    from mlx_lm import load as mlx_load

    from mlx_rl import machine
    from mlx_rl.engine import rollout_groups
    from mlx_rl.profiles import get_profile
    from mlx_rl.rollout import encode_prompt
    from mlx_rl.tasks import get_task
    from mlx_rl.tasks.qa_abstain import grade, normalize, parse_reply
    from mlx_rl.train import _completion_text

    # Forced-answer framing (mirrors qa_calibrate): the detector judges
    # committed answers; abstention is chapter 2's business.
    FORCED = ("Answer this question with a short factual answer, wrapped in "
              "answer tags like <answer>Paris</answer>. Give your best guess "
              "even if you are not sure.\n\nQuestion: {q}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    task = get_task("qa_abstain", dataset=a.dataset)
    rng = random.Random(a.seed)
    rows = rng.sample(task._train, min(a.n, len(task._train)))
    (out / "config.json").write_text(json.dumps({
        "qa_introspect": True, "model": prof.model, "profile": prof.name,
        "dataset": a.dataset, "n": len(rows), "k": a.k, "seed": a.seed,
    }, indent=2) + "\n")

    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(38.0, note="qa introspection signal capture")
    try:
        model, tokenizer = mlx_load(prof.model)
        h_prompt_all, h_commit_all = [], []
        t0 = time.time()
        with (out / "signals.jsonl").open("w") as f:
            for lo in range(0, len(rows), a.batch_questions):
                chunk = rows[lo:lo + a.batch_questions]
                prompts = [encode_prompt(
                    tokenizer,
                    [{"role": "user", "content": FORCED.format(q=r["question"])}],
                    **prof.chat_kwargs) for r in chunk]
                # S4: temp-1 resamples, batched
                groups, _ = rollout_groups(
                    model, tokenizer, prompts, a.k, a.max_new_tokens, 1.0,
                    extra_eos=tuple(prof.extra_eos))
                for row, pid, group in zip(chunk, prompts, groups):
                    comp = greedy_generate(model, tokenizer, pid,
                                           a.max_new_tokens,
                                           tuple(prof.extra_eos))
                    text = _completion_text(
                        tokenizer, type("C", (), {"tokens": comp,
                                                  "finish_reason": None})())
                    kind, value = parse_reply(text)
                    correct = kind == "answer" and grade(value, row["aliases"])
                    lp, ent, marg, h_p, h_c = rescore(model, pid, comp)
                    span = value_token_span(tokenizer, comp)
                    rec = {
                        "qid": row["qid"], "question": row["question"],
                        "greedy_kind": kind, "greedy_value": value,
                        "correct": bool(correct),
                        "mean_lp": float(lp.mean()),
                        "min_lp": float(lp.min()),
                    }
                    if span:
                        s, e = span
                        rec["value_mean_lp"] = float(lp[s:e].mean())
                        rec["neg_entropy_commit"] = -float(ent[s])
                        rec["margin_commit"] = float(marg[s])
                        h_commit = h_c[s]
                    else:
                        h_commit = np.zeros_like(h_p)
                    vals = []
                    for c in group:
                        k2, v2 = parse_reply(_completion_text(tokenizer, c))
                        vals.append(normalize(v2) if k2 == "answer" and v2
                                    else f"<{k2}>")
                    top = max(vals.count(v) for v in set(vals))
                    rec["agreement"] = top / len(vals)
                    rec["neg_distinct"] = -len(set(vals))
                    rec["samples"] = vals[:8]
                    f.write(json.dumps(rec) + "\n")
                    h_prompt_all.append(h_p)
                    h_commit_all.append(h_commit)
                done = min(lo + a.batch_questions, len(rows))
                print(f"{done}/{len(rows)} ({(time.time()-t0)/done:.1f}s/q)",
                      flush=True)
        np.savez_compressed(out / "signals.npz",
                            h_prompt=np.stack(h_prompt_all),
                            h_commit=np.stack(h_commit_all))
    finally:
        machine.release(holder)
    analyze(out)


def auroc(scores, labels) -> float:
    """Rank-based AUROC (ties get average rank)."""
    import numpy as np
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    for v in np.unique(scores):
        m = scores == v
        ranks[m] = ranks[m].mean()
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def analyze(out: Path) -> None:
    recs = [json.loads(l)
            for l in (out / "signals.jsonl").read_text().splitlines()]
    # The detector's question is "given a COMMITTED answer, is it right?" —
    # malformed greedies measure format ability, not knowledge; keep them
    # out of the headline AUROC.
    answered = [r for r in recs if r["greedy_kind"] == "answer"]
    n_pos = sum(r["correct"] for r in answered)
    print(f"\n{len(recs)} items: {len(answered)} answered "
          f"({len(recs) - len(answered)} malformed/other), "
          f"correct {n_pos}/{len(answered)} — AUROC of each signal for "
          f"predicting correctness of the committed answer:")
    results = {}
    for sig in SIGNALS:
        pairs = [(r[sig], r["correct"]) for r in answered if sig in r]
        if len(pairs) < 20:
            continue
        a = auroc([p[0] for p in pairs], [p[1] for p in pairs])
        results[sig] = {"auroc": round(a, 3), "n": len(pairs)}
        print(f"  {sig:20s} {a:.3f}   (n={len(pairs)})")
    # Raw extremes per the show-raw-data rule: most/least confident by S4.
    by_agree = sorted(recs, key=lambda r: r["agreement"])
    print("\nmost diffuse (story-mode candidates):")
    for r in by_agree[:3]:
        print(f"  [{r['agreement']:.2f}] {r['question'][:60]} -> "
              f"{r['greedy_value']!r} correct={r['correct']} "
              f"samples={r['samples'][:4]}")
    print("most concentrated (lookup-mode candidates):")
    for r in by_agree[-3:]:
        print(f"  [{r['agreement']:.2f}] {r['question'][:60]} -> "
              f"{r['greedy_value']!r} correct={r['correct']}")
    (out / "auroc.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="tiny")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=8, help="S4 resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-questions", type=int, default=16)
    ap.add_argument("--out", default=f"runs/introspect-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    ap.add_argument("--analyze", metavar="RUN_DIR", default=None,
                    help="skip capture; re-analyze an existing run dir")
    a = ap.parse_args()
    if a.analyze:
        analyze(Path(a.analyze))
    else:
        capture(a)


if __name__ == "__main__":
    main()
