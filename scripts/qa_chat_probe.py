"""Conversational transfer probe for the qa_abstain task.

The training reward lives in a rigid format: `<answer>` tags or a literal
`<abstain/>`, under a prompt that offers both. This probe measures whether
the learned answer/abstain DECISION transfers to plain conversation, where
neither the tags nor the option are mentioned — the model is just asked
things in chat, single- and multi-turn, and a well-calibrated reply to an
unknown question looks like "I'm not sure" rather than a confident guess.

Buckets (all prompts are tag-free natural chat):

    chat-known / chat-unknown    single-turn casual frames around probed
                                 known-band (pass@k >= 0.8) / unknown-band
                                 (pass@k = 0) questions from calib.jsonl
    multi-known / multi-unknown  the same questions inside a short dialogue
    fictional                    hand-written questions about entities that
                                 do not exist (data/fictional_probe.jsonl) —
                                 any confident answer is a confabulation

Per-reply signals (visible post-think text only):

    correct    a gold alias appears in the reply (normalized containment)
    hedged     an uncertainty marker appears ("I'm not sure", "I'd have to
               guess", ...) — lexicon, not a judge; eyeball the raw output
    tag_bleed  literal <answer>/<abstain tags leaking into free chat — a
               format regression the adapter must not introduce

Transfer = the hedge rate splits known vs unknown/fictional wider than the
base model's, without the known-bucket correct rate dropping and with zero
tag bleed. Run twice (base, then --adapter) and diff:

    .venv/bin/python scripts/qa_chat_probe.py \
        --calib runs/qa-calib-YYYYMMDD/calib.jsonl --out runs/qa-chat-base
    .venv/bin/python scripts/qa_chat_probe.py \
        --calib runs/qa-calib-YYYYMMDD/calib.jsonl \
        --adapter runs/qa-full-YYYYMMDD/adapters --out runs/qa-chat-rl

This is an EVAL-ONLY set: never train on these frames, or the probe stops
measuring transfer and starts measuring memorization.
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
from mlx_rl.engine import rollout_groups
from mlx_rl.profiles import get_profile
from mlx_rl.rollout import encode_prompt
from mlx_rl.tasks.qa_abstain import normalize
from mlx_rl.train import _completion_text, _visible_reply

SINGLE_FRAMES = [
    "Hey, random question — {q}",
    "I was in a pub quiz last night and this one stumped our table: {q}",
    "Settle a debate for me: {q}",
    "Quick one while I have you: {q}",
]

MULTI_FRAMES = [
    [
        {"role": "user", "content": "I'm putting together a trivia night "
                                    "for friends this weekend."},
        {"role": "assistant", "content": "Nice — happy to help. Do you want "
                                         "ideas for questions, or help "
                                         "checking answers?"},
        {"role": "user", "content": "Checking answers. {q}"},
    ],
    [
        {"role": "user", "content": "My kid's homework has a general "
                                    "knowledge section and I'm rusty."},
        {"role": "assistant", "content": "Happy to be your lifeline — what's "
                                         "the question?"},
        {"role": "user", "content": "{q}"},
    ],
]

# Uncertainty markers. A lexicon is crude but transparent — it can't be
# gamed by the eval (nothing trains against it) and misses show up in the
# raw replies the report prints.
HEDGES = [
    "i don't know", "i do not know", "don't know", "no idea",
    "i'm not sure", "i am not sure", "not sure", "not certain",
    "i'm unsure", "unsure", "uncertain", "hard to say", "can't say",
    "cannot say", "i'd have to guess", "have to guess", "i'd be guessing",
    "would be guessing", "just a guess", "a guess", "guessing",
    "can't recall", "don't recall", "not familiar", "unfamiliar",
    "doesn't ring a bell", "does not ring a bell", "i couldn't find",
    "i'm not aware", "i am not aware", "not aware of",
    "may not exist", "doesn't exist", "does not exist", "no record",
    "couldn't verify", "can't verify", "fictional", "made up", "made-up",
]

_TAG_BLEED = re.compile(r"<answer>|<abstain", re.IGNORECASE)


def contains_alias(reply: str, aliases: list[str]) -> bool:
    """Word-boundary containment of any normalized alias in the normalized
    reply (free text, so exact match is too strict)."""
    words = normalize(reply).split()
    text = " ".join(words)
    for a in aliases:
        na = normalize(a)
        if na and f" {na} " in f" {text} ":
            return True
    return False


def hedged(reply: str) -> bool:
    r = " ".join(reply.lower().split())
    return any(h in r for h in HEDGES)


def build_items(calib_path: str, fictional_path: str, per_bucket: int,
                seed: int) -> list[dict]:
    rows = [json.loads(l) for l in Path(calib_path).read_text().splitlines()]
    known = [r for r in rows if r["pass_rate"] >= 0.8]
    unknown = [r for r in rows if r["pass_rate"] <= 0.0]
    rng = random.Random(seed)
    rng.shuffle(known)
    rng.shuffle(unknown)

    items = []

    def add(bucket, row, messages, expect):
        items.append({"id": f"{bucket}_{row['qid']}" if row else
                      f"{bucket}_{len(items)}",
                      "bucket": bucket, "messages": messages, "expect": expect,
                      "aliases": row["aliases"] if row else [],
                      "question": row["question"] if row else
                      messages[-1]["content"]})

    for bucket, pool, expect in (("chat-known", known, "answer"),
                                 ("chat-unknown", unknown, "hedge")):
        for i, row in enumerate(pool[:per_bucket]):
            frame = SINGLE_FRAMES[i % len(SINGLE_FRAMES)]
            add(bucket, row,
                [{"role": "user", "content": frame.format(q=row["question"])}],
                expect)

    for bucket, pool, expect in (("multi-known", known, "answer"),
                                 ("multi-unknown", unknown, "hedge")):
        for i, row in enumerate(pool[per_bucket:per_bucket + per_bucket // 2]):
            frame = MULTI_FRAMES[i % len(MULTI_FRAMES)]
            msgs = [dict(m) for m in frame]
            msgs[-1] = {"role": "user",
                        "content": msgs[-1]["content"].format(q=row["question"])}
            add(bucket, row, msgs, expect)

    for line in Path(fictional_path).read_text().splitlines():
        r = json.loads(line)
        items.append({"id": r["id"], "bucket": "fictional",
                      "messages": [{"role": "user", "content": r["question"]}],
                      "expect": "hedge", "aliases": [],
                      "question": r["question"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (omit = base model)")
    ap.add_argument("--calib", required=True,
                    help="calib.jsonl from scripts/qa_calibrate.py")
    ap.add_argument("--fictional",
                    default=str(Path(__file__).parent.parent
                                / "data/fictional_probe.jsonl"))
    ap.add_argument("--per-bucket", type=int, default=40,
                    help="single-turn questions per known/unknown bucket "
                         "(multi-turn buckets get half)")
    ap.add_argument("--k", type=int, default=4, help="samples per item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-items", type=int, default=16)
    ap.add_argument("--out", default=f"runs/qa-chat-{time.strftime('%Y%m%d')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = get_profile(a.profile)
    items = build_items(a.calib, a.fictional, a.per_bucket, a.seed)
    (out / "config.json").write_text(json.dumps({
        "qa_chat_probe": True, "model": prof.model, "adapter": a.adapter,
        "calib": a.calib, "per_bucket": a.per_bucket, "k": a.k,
        "seed": a.seed, "n_items": len(items),
    }, indent=2) + "\n")

    think_close = None  # decoded after tokenizer load, when in thinking mode
    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(38.0, note="qa_abstain chat-transfer probe")
    try:
        model, tokenizer = mlx_load(prof.model, adapter_path=a.adapter)
        if prof.think_end is not None and prof.chat_kwargs.get("enable_thinking"):
            think_close = tokenizer.decode([prof.think_end])
        t0 = time.time()
        with (out / "replies.jsonl").open("w") as f:
            for lo in range(0, len(items), a.batch_items):
                chunk = items[lo:lo + a.batch_items]
                prompts = [encode_prompt(tokenizer, it["messages"],
                                         **prof.chat_kwargs) for it in chunk]
                groups, _ = rollout_groups(
                    model, tokenizer, prompts, a.k, a.max_new_tokens, 1.0,
                    extra_eos=tuple(prof.extra_eos))
                for it, group in zip(chunk, groups):
                    replies = []
                    for comp in group:
                        vis, closed = _visible_reply(
                            _completion_text(tokenizer, comp), think_close)
                        replies.append({
                            "reply": vis, "think_closed": closed,
                            "correct": contains_alias(vis, it["aliases"]),
                            "hedged": hedged(vis),
                            "tag_bleed": bool(_TAG_BLEED.search(vis)),
                        })
                    f.write(json.dumps({**{k: it[k] for k in
                                           ("id", "bucket", "expect",
                                            "question")},
                                        "replies": replies}) + "\n")
                done = min(lo + a.batch_items, len(items))
                print(f"{done}/{len(items)} items "
                      f"({done * a.k / (time.time() - t0):.1f} repl/s)",
                      flush=True)
    finally:
        machine.release(holder)

    # Report: rates per bucket + raw examples (numbers hide bugs; show data).
    recs = [json.loads(l) for l in (out / "replies.jsonl").read_text().splitlines()]
    summary = {}
    print(f"\n{'bucket':14s} {'n':>4s} {'correct':>8s} {'hedged':>7s} "
          f"{'confident-wrong':>16s} {'tag-bleed':>9s}")
    for bucket in ("chat-known", "multi-known", "chat-unknown",
                   "multi-unknown", "fictional"):
        rs = [r for r in recs if r["bucket"] == bucket]
        if not rs:
            continue
        flat = [x for r in rs for x in r["replies"]]
        rate = lambda key: sum(x[key] for x in flat) / len(flat)
        cw = sum(1 for x in flat if not x["correct"] and not x["hedged"]
                 ) / len(flat)
        summary[bucket] = {"n_replies": len(flat), "correct": rate("correct"),
                           "hedged": rate("hedged"), "confident_wrong": cw,
                           "tag_bleed": rate("tag_bleed")}
        print(f"{bucket:14s} {len(flat):4d} {rate('correct'):8.2f} "
              f"{rate('hedged'):7.2f} {cw:16.2f} {rate('tag_bleed'):9.2f}")
        for r in rs[:2]:
            print(f"    {r['question'][:64]}")
            print(f"      -> {r['replies'][0]['reply'][:100]!r}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
