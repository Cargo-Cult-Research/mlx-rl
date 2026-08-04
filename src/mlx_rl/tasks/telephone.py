"""Telephone: emergent-code game against a frozen copy of the same model.

lifecycle: core (ears-project POC; graduates or dies with the centaur arc)

The speaker (the trained policy) is told a secret — a chord label drawn from
quality × root = {major, minor, diminished, augmented} × {C ... B}, 48
classes, ~5.6 bits — and must transmit it to a "twin" through a channel of
k tokens (default 1). The listener is the SAME model with adapters disabled
(the GRPO reference trick): frozen, in-context only, exactly the situation
of a black-box partner. Reward is the frozen listener's forced-choice
probability of the true label given the code.

Why this is the right first loop for "is there a there there" (2026-08-03
design discussion): no single natural token names both quality and root, so
the natural-language strategy caps out at partial credit; beating that cap
requires mining the frozen listener's prior for tokens that carry BOTH —
precisely the RLPrompt/GCG phenomenon, now as a communication channel. No
audio, no second model, no new plumbing: every confounder stripped except
the claim itself.

Reward details:
- The channel is the first k tokens of the reply, period. No format gate —
  a format gate would make early groups all-zero and feed the collapse
  watchdog; truncation-as-channel keeps every rollout informative.
- Listener scores are CONTRASTIVE: log P(candidate | code) minus
  log P(candidate | null code), cached once. This cancels the listener's
  per-candidate prior and length bias (sum-logprob favors short candidates;
  "major C" vs "diminished F#" differ by tokens).
- reward = softmax over the 48 contrasted scores at the true label, in
  [0, 1] and smooth — GRPO-friendly. Marginals over quality and root are
  logged as parts to show partial-information codes on the way up.

NOT a reward model: the listener is deterministic teacher-forced scoring of
a frozen net — verifiable in the sense that the same (code, label) always
earns the same reward, and nothing optimizes the grader.
"""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np

from .base import Example, RewardResult, register

QUALITIES = ("major", "minor", "diminished", "augmented")
ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
LABELS = [(q, r) for q in QUALITIES for r in ROOTS]
NULL_CODE = "?"


def _speaker_prompt(quality: str, root: str, k: int) -> str:
    unit = "exactly one token" if k == 1 else f"at most {k} tokens"
    return (
        "You are playing a code game with your identical twin — same model, "
        f"same weights, but the twin sees ONLY your code, nothing else. "
        f"Secret chord: {quality} {root}. "
        f"Send a code of {unit} (any string, it does not have to be an "
        "English word) that lets the twin recover BOTH the quality "
        "(major/minor/diminished/augmented) AND the root (C, C#, D, D#, E, "
        "F, F#, G, G#, A, A#, B). Reply with the code only."
    )


def _listener_prompt(code: str) -> str:
    return (
        "Your identical twin (same model, same weights) was told a secret "
        f'chord and could send you only a tiny code. The code is: "{code}". '
        "The chord's quality is one of major, minor, diminished, augmented; "
        "its root is one of C, C#, D, D#, E, F, F#, G, G#, A, A#, B. "
        'Decode it: reply with quality and root only, like "minor F#".'
    )


@register
class TelephoneTask:
    name = "telephone"

    # keeps qwen3.x rollouts out of thinking mode; harmlessly ignored by
    # templates without the variable (tiny/Qwen2.5).
    chat_template_kwargs = {"enable_thinking": False}

    def __init__(self, k_tokens: int = 1, score_chunk: int = 64):
        self.k = k_tokens
        self.score_chunk = score_chunk
        self.model = None
        self.tokenizer = None
        self._null_lp: np.ndarray | None = None  # [48] cached null-code scores
        self._cand_ids: list[list[int]] | None = None
        # The listener is frozen, so (code -> scores) is a pure function:
        # cache it. Sampled codes repeat heavily (peaked first-token
        # distribution), and as the policy converges the hit rate -> 1.
        self._score_cache: dict[str, np.ndarray] = {}

    # -- trainer hook ------------------------------------------------------
    def bind_model(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._cand_ids = [self._encode(f"{q} {r}") for q, r in LABELS]
        self._null_lp = self._score_codes([NULL_CODE])[0]
        # bind-time sanity: the game must be winnable when the code IS the
        # answer. If this prints ~chance the decoder prompt is broken and no
        # amount of RL will save the run.
        probe = [f"{q} {r}" for q, r in random.Random(0).sample(LABELS, 4)]
        lp = self._score_codes(probe) - self._null_lp
        p = _softmax(lp)
        hits = [float(p[i][LABELS.index(tuple(c.split()))]) for i, c in enumerate(probe)]
        print(f"telephone bind sanity — p(correct | code=answer): "
              f"{[round(h, 3) for h in hits]} (chance {1 / len(LABELS):.3f})")

    # -- Task protocol -----------------------------------------------------
    def sample(self, rng: random.Random) -> Example:
        q, r = rng.choice(LABELS)
        return Example(
            messages=[{"role": "user", "content": _speaker_prompt(q, r, self.k)}],
            meta={"quality": q, "root": r, "idx": LABELS.index((q, r))},
        )

    def batch_reward(self, examples: list[Example],
                     completions: list[str]) -> list[RewardResult]:
        codes = [self._channel(c) for c in completions]
        fresh = sorted({c for c in codes if c not in self._score_cache})
        if fresh:
            for code, row in zip(fresh, self._score_codes(fresh)):
                self._score_cache[code] = row
        lp = np.stack([self._score_cache[c] for c in codes]) - self._null_lp
        p = _softmax(lp)
        out = []
        for ex, code, pi in zip(examples, codes, p):
            i = ex.meta["idx"]
            q, r = LABELS[i]
            p_quality = float(sum(pi[j] for j, (qq, _) in enumerate(LABELS) if qq == q))
            p_root = float(sum(pi[j] for j, (_, rr) in enumerate(LABELS) if rr == r))
            total = float(pi[i]) if code else 0.0
            out.append(RewardResult(total, {
                "p_correct": float(pi[i]),
                "p_quality": p_quality,
                "p_root": p_root,
            }))
        return out

    def reward(self, example: Example, completion: str) -> RewardResult:
        return self.batch_reward([example], [completion])[0]

    # -- internals ---------------------------------------------------------
    def _encode(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def _channel(self, completion: str) -> str:
        """The transmitted code: the first k tokens of the visible reply."""
        ids = self._encode(completion.strip())[: self.k]
        return self.tokenizer.decode(ids).strip()

    def _score_codes(self, codes: list[str]) -> np.ndarray:
        """[len(codes), 48] sum-logprob of each candidate answer under the
        FROZEN listener, teacher-forced. One row per (code, candidate)."""
        from ..models import adapters_disabled, selective_logprobs

        rows, row_sel = [], []
        for code in codes:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": _listener_prompt(code)}],
                add_generation_prompt=True, **self.chat_template_kwargs)
            for cand in self._cand_ids:
                full = list(prompt) + list(cand)
                # target positions of the candidate tokens, in shifted coords
                rows.append(full)
                row_sel.append(list(range(len(prompt) - 1, len(full) - 1)))

        pad = next(iter(sorted(self.tokenizer.eos_token_ids)))
        kmax = max(len(s) for s in row_sel)
        scores = np.zeros(len(rows), dtype=np.float64)
        with adapters_disabled(self.model):
            for lo in range(0, len(rows), self.score_chunk):
                chunk = rows[lo:lo + self.score_chunk]
                sels = row_sel[lo:lo + self.score_chunk]
                lmax = max(len(r) for r in chunk)
                inp = np.full((len(chunk), lmax - 1), pad, dtype=np.int64)
                tgt = np.full((len(chunk), kmax), pad, dtype=np.int64)
                sel = np.zeros((len(chunk), kmax), dtype=np.int64)
                msk = np.zeros((len(chunk), kmax), dtype=np.float64)
                for i, (row, s) in enumerate(zip(chunk, sels)):
                    inp[i, : len(row) - 1] = row[:-1]
                    tgt[i, : len(s)] = [row[j + 1] for j in s]
                    sel[i, : len(s)] = s
                    msk[i, : len(s)] = 1.0
                lp = selective_logprobs(
                    self.model, mx.array(inp), mx.array(tgt), mx.array(sel))
                lp = np.array(lp.astype(mx.float32), dtype=np.float64)  # np can't read bf16
                scores[lo:lo + self.score_chunk] = (lp * msk).sum(axis=1)
                mx.clear_cache()
        return scores.reshape(len(codes), len(LABELS))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
