"""GRPO core math.

Group-relative advantages plus the clipped policy-gradient objective with a
KL penalty to the frozen reference policy. Everything here is pure
(arrays in, arrays out) so it is unit-testable without a model.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def group_advantages(
    rewards: mx.array, normalize_std: bool = True, eps: float = 1e-4
) -> mx.array:
    """rewards [n_prompts, group_size] -> advantages of the same shape.

    The group mean is the baseline. A zero-variance group (all completions
    equally rewarded) yields zero advantage — no gradient signal.
    """
    adv = rewards - mx.mean(rewards, axis=1, keepdims=True)
    if normalize_std:
        std = mx.sqrt(mx.var(rewards, axis=1, keepdims=True))
        adv = adv / (std + eps)
    return adv


def active_groups(rewards: mx.array, eps: float = 1e-6) -> mx.array:
    """Boolean [n_prompts]: groups with any reward spread.

    Zero-variance groups carry no learning signal — but their fp16 kernel
    noise still produces a nonzero 'gradient' that Adam rescales to a
    full-size step. Updating on them is a destructive random walk, so the
    trainer drops them (and skips the step entirely if none remain).
    """
    return mx.sqrt(mx.var(rewards, axis=1)) > eps


def subsample_token_mask(
    mask: np.ndarray, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Token-subset backprop (S-GRPO, arXiv:2504.20834): keep a uniform
    random `frac` of each row's 1-entries, zeroing the rest.

    Uniform-over-completion-tokens is the principled choice here: the
    advantage is one scalar per sequence, so the objective carries no
    per-token credit signal — a uniform subset with the denominator scaled
    by `frac` is an unbiased estimator of the full-token gradient. Exact
    per-row counts (max(1, round(frac*n)), sampled without replacement)
    rather than Bernoulli: lower variance, and no row ever goes empty.
    Rounding makes a row's inclusion probability k/n deviate from frac by
    at most 1/(2n) — negligible at real rollout lengths.
    """
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        pos = np.flatnonzero(mask[i])
        if pos.size == 0:
            continue
        k = max(1, int(round(frac * pos.size)))
        keep = rng.choice(pos, size=min(k, pos.size), replace=False)
        out[i, keep] = mask[i, keep]
    return out


def token_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """logits [B, L, V], targets [B, L] -> log p(target_t) [B, L]."""
    return -nn.losses.cross_entropy(logits, targets, reduction="none")


def grpo_objective(
    cur_lp: mx.array,
    old_lp: mx.array,
    ref_lp: mx.array,
    advantages: mx.array,
    mask: mx.array,
    denom: float,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
) -> tuple[mx.array, mx.array, mx.array]:
    """Clipped surrogate + k3 KL penalty.

    cur_lp / old_lp / ref_lp: [B, L] per-token logprobs (old/ref are treated
    as constants). advantages: [B], one scalar per sequence. mask: [B, L],
    1.0 on completion tokens. denom: completion-token count of the FULL batch
    so that microbatch losses sum to the batch loss under grad accumulation
    (global per-token normalization, avoiding GRPO's length bias).

    Returns (loss, pg_sum, kl_sum); pg_sum/kl_sum are unnormalized sums for
    metric aggregation across microbatches.
    """
    old_lp = mx.stop_gradient(old_lp)
    ref_lp = mx.stop_gradient(ref_lp)

    # Mask inside the exponent: garbage logprobs at padded positions would
    # otherwise overflow to inf and poison the sum as inf * 0 = nan.
    ratio = mx.exp((cur_lp - old_lp) * mask)
    adv = advantages[:, None]
    surrogate = mx.minimum(
        ratio * adv, mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    )
    pg_sum = (surrogate * mask).sum()

    d = (ref_lp - cur_lp) * mask
    kl = mx.exp(d) - d - 1.0  # k3 estimator, non-negative
    kl_sum = (kl * mask).sum()

    loss = (-pg_sum + kl_coef * kl_sum) / denom
    return loss, pg_sum, kl_sum
