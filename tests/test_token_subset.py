"""Token-subset backprop (S-GRPO): selection, gathering, and estimator checks.

The contract under test: at frac=1.0 the selective path reproduces the dense
masked loss exactly (same tokens, same math, smaller logits slab); at frac<1
the subset loss is an unbiased estimator of the dense loss once the
denominator is scaled by frac.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_rl.grpo import grpo_objective, subsample_token_mask, token_logprobs
from mlx_rl.models import selective_logprobs
from mlx_rl.rollout import gather_selected


def _mask(rows):
    """Binary mask from per-row (start, count) completion spans."""
    m = np.zeros((len(rows), 12), dtype=np.float32)
    for i, (start, count) in enumerate(rows):
        m[i, start : start + count] = 1.0
    return m


def test_subsample_is_subset_with_exact_counts():
    mask = _mask([(2, 8), (0, 3), (5, 1)])
    rng = np.random.default_rng(0)
    sub = subsample_token_mask(mask, 0.5, rng)
    # only ever zeroes entries, never creates them
    assert np.all(mask - sub >= 0)
    # per-row: max(1, round(frac * n))
    assert sub[0].sum() == 4
    assert sub[1].sum() == 2  # round(1.5) -> 2
    assert sub[2].sum() == 1  # floor of 1 kept


def test_subsample_frac1_is_identity_and_empty_rows_stay_empty():
    mask = _mask([(2, 8), (0, 0)])
    sub = subsample_token_mask(mask, 1.0, np.random.default_rng(0))
    assert np.array_equal(sub, mask)


def test_gather_selected_padding_is_masked():
    mask = _mask([(2, 5), (0, 2)])
    tgt = np.arange(24).reshape(2, 12)
    old = np.random.default_rng(1).normal(size=(2, 12)).astype(np.float32)
    idx, sel_mask, tgt_s, old_s, _ = gather_selected(mask, tgt, old, old, 0, 2)
    assert idx.shape == (2, 5)  # K = longest row's selection
    assert sel_mask[0].sum() == 5 and sel_mask[1].sum() == 2
    # gathered values match the source at the selected positions
    assert np.array_equal(tgt_s[0], tgt[0, 2:7])
    assert np.array_equal(tgt_s[1][:2], tgt[1, :2])


class _Trunk(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.proj = nn.Linear(H, H)

    def __call__(self, x):
        return mx.tanh(self.proj(self.embed_tokens(x)))


class _MiniModel(nn.Module):
    """Structural stand-in for an mlx-lm CausalLM: .model trunk + .lm_head."""

    def __init__(self, V=50, H=16):
        super().__init__()
        self.args = SimpleNamespace(tie_word_embeddings=False)
        self.model = _Trunk(V, H)
        self.lm_head = nn.Linear(H, V, bias=False)

    def __call__(self, x):
        return self.lm_head(self.model(x))


def _batch(seed=0, B=3, L=12, V=50):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, V, size=(B, L))
    tgt = rng.integers(0, V, size=(B, L))
    mask = _mask([(2, 8), (0, 5), (4, 6)])
    old = rng.normal(-2.0, 0.3, size=(B, L)).astype(np.float32)
    ref = rng.normal(-2.0, 0.3, size=(B, L)).astype(np.float32)
    adv = rng.normal(size=B).astype(np.float32)
    return inp, tgt, mask, old, ref, adv


def _dense_loss_and_grads(model, inp, tgt, mask, old, ref, adv, denom):
    def f(model):
        cur = token_logprobs(model(mx.array(inp)), mx.array(tgt))
        loss, _, _ = grpo_objective(
            cur, mx.array(old), mx.array(ref), mx.array(adv),
            mx.array(mask), denom)
        return loss

    return nn.value_and_grad(model, f)(model)


def _selective_loss_and_grads(model, inp, tgt, mask, old, ref, adv, denom):
    idx, sel_mask, tgt_s, old_s, ref_s = gather_selected(
        mask, tgt, old, ref, 0, mask.shape[0])

    def f(model):
        cur = selective_logprobs(
            model, mx.array(inp), mx.array(tgt_s), mx.array(idx))
        loss, _, _ = grpo_objective(
            cur, mx.array(old_s), mx.array(ref_s), mx.array(adv),
            mx.array(sel_mask), denom)
        return loss

    return nn.value_and_grad(model, f)(model)


class _WrappedModel(nn.Module):
    """VLM-style wrapper (qwen3_5/qwen3_5_moe): CausalLM nested one level
    down as .language_model — the structure that actually serves qwen36."""

    def __init__(self, V=50, H=16):
        super().__init__()
        self.language_model = _MiniModel(V, H)

    def __call__(self, x):
        return self.language_model(x)


def test_selective_path_unwraps_language_model_wrapper():
    model = _WrappedModel()
    inp, tgt, mask, old, ref, adv = _batch()
    denom = float(mask.sum())
    l_dense, _ = _dense_loss_and_grads(
        model, inp, tgt, mask, old, ref, adv, denom)
    l_sel, _ = _selective_loss_and_grads(
        model, inp, tgt, mask, old, ref, adv, denom)
    assert abs(float(l_dense) - float(l_sel)) < 1e-6


def test_selective_path_matches_dense_at_full_selection():
    """frac=1.0: same tokens through the same math -> same loss AND grads."""
    model = _MiniModel()
    inp, tgt, mask, old, ref, adv = _batch()
    denom = float(mask.sum())
    l_dense, g_dense = _dense_loss_and_grads(
        model, inp, tgt, mask, old, ref, adv, denom)
    l_sel, g_sel = _selective_loss_and_grads(
        model, inp, tgt, mask, old, ref, adv, denom)
    assert abs(float(l_dense) - float(l_sel)) < 1e-6
    from mlx.utils import tree_flatten
    for (k, a), (k2, b) in zip(tree_flatten(g_dense), tree_flatten(g_sel)):
        assert k == k2
        assert float(mx.abs(a - b).max()) < 1e-6, k


def test_subset_loss_is_unbiased_for_dense_loss():
    """E_draws[selective loss @ frac, denom*frac] == dense loss."""
    model = _MiniModel()
    inp, tgt, mask, old, ref, adv = _batch(seed=3)
    # spans where frac*n is integral, so per-row inclusion prob == frac
    # exactly (rounding otherwise adds an O(1/2n) bias — negligible at real
    # rollout lengths, visible on toy rows)
    mask = _mask([(2, 8), (0, 6), (4, 4)])
    denom = float(mask.sum())
    l_dense, _ = _dense_loss_and_grads(
        model, inp, tgt, mask, old, ref, adv, denom)
    frac = 0.5
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(400):
        sub = subsample_token_mask(mask, frac, rng)
        l, _ = _selective_loss_and_grads(
            model, inp, tgt, sub, old, ref, adv, denom * frac)
        draws.append(float(l))
    est = np.mean(draws)
    sem = np.std(draws) / np.sqrt(len(draws))
    # within 4 standard errors of the dense loss (and sem must be sane)
    assert abs(est - float(l_dense)) < 4 * sem + 1e-6, (est, float(l_dense), sem)
