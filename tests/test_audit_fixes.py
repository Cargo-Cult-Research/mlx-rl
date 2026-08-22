"""Tests for the 2026-08-22 audit fixes: truncation grading, per-sign
advantage pruning, KL penalty clamp, OOM-backoff chunking, preflight."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_rl.grpo import group_advantages, grpo_objective
from mlx_rl.train import (_chunked_backoff, neutralize_capped_rewards,
                          prune_advantages)


# -- advantage pruning ------------------------------------------------------

def test_prune_keeps_both_signs_on_asymmetric_rewards():
    # The bug case: {+1 x7, -3} group. |adv| threshold at 0.5 kept ONLY the
    # -3 outlier (kept advantage sum -2.65 vs true 0): pure suppression.
    rewards = mx.array(np.array([[1.0] * 7 + [-3.0]], dtype=np.float32))
    adv = np.array(group_advantages(rewards)).reshape(-1)
    kept, keep, n_pruned = prune_advantages(adv.copy(), 8, 0.5)
    assert keep.sum() == 8, "per-sign threshold must keep the majority side"
    assert abs(kept.sum()) < 1e-5, "kept advantages must sum to ~0"
    assert (kept > 0).sum() == 7 and (kept < 0).sum() == 1


def test_prune_still_prunes_small_advantages_within_sign():
    adv = np.array([2.0, 0.1, -0.1, -2.0], dtype=np.float32)
    kept, keep, n_pruned = prune_advantages(adv.copy(), 4, 0.9)
    assert n_pruned == 2                     # the two small ones go
    assert abs(kept.sum()) < 1e-6            # the survivors stay balanced


def test_prune_zero_advantage_members_dropped():
    # Exactly-zero advantages (e.g. neutralized len-capped members) are
    # neither positive nor negative: always pruned, never re-inflated.
    adv = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    kept, keep, n_pruned = prune_advantages(adv.copy(), 4, 0.25)
    assert list(keep) == [True, True, False, False]


# -- len-capped neutralization ---------------------------------------------

def test_neutralize_capped_sets_advantage_to_zero():
    rewards = np.array([[1.0, 1.0, 0.0, -3.0]], dtype=np.float32)
    capped = np.array([[False, False, False, True]])
    n = neutralize_capped_rewards(rewards, capped)
    assert n == 1
    # capped member now equals the mean of the alive members...
    assert rewards[0, 3] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    # ...so its group-relative advantage is exactly zero
    adv = np.array(group_advantages(mx.array(rewards)))
    assert adv[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_neutralize_all_capped_group_untouched():
    rewards = np.array([[0.0, 0.0]], dtype=np.float32)
    capped = np.array([[True, True]])
    assert neutralize_capped_rewards(rewards, capped) == 0
    assert (rewards == 0.0).all()


# -- KL penalty clamp -------------------------------------------------------

def test_kl_penalty_clamped_but_metric_unclamped():
    # One token where the policy crushed mass the base liked: d = 20.
    B, L = 1, 4
    cur = mx.array(np.full((B, L), -21.0, dtype=np.float32))
    ref = mx.array(np.full((B, L), -1.0, dtype=np.float32))
    old = cur
    mask = mx.array(np.ones((B, L), dtype=np.float32))
    adv = mx.array(np.array([1.0], dtype=np.float32))
    loss, pg, kl = grpo_objective(cur, old, ref, adv, mask, denom=float(L),
                                  clip_eps=0.2, kl_coef=1.0)
    # metric reports the true (unclamped) k3: exp(20) ~ 4.85e8 per token
    assert float(kl) > 1e8
    # loss uses the clamped penalty: exp(10)-11 ~ 2.2e4 per token, x4 tokens
    assert float(loss) < 1e5


def test_kl_penalty_zero_when_policies_agree():
    lp = mx.array(np.full((1, 3), -2.0, dtype=np.float32))
    mask = mx.array(np.ones((1, 3), dtype=np.float32))
    adv = mx.array(np.array([1.0], dtype=np.float32))
    loss, pg, kl = grpo_objective(lp, lp, lp, adv, mask, denom=3.0)
    assert float(kl) == pytest.approx(0.0, abs=1e-6)


# -- chunked generation with OOM backoff -----------------------------------

def test_chunked_backoff_halves_on_oom():
    calls = []

    def sample(items):
        calls.append(len(items))
        if len(items) > 2:
            raise RuntimeError("[metal::malloc] Insufficient Memory")
        return [f"g{i}" for i in items], [f"p{i}" for i in items], "stats"

    groups, prompts, stats = _chunked_backoff(sample, list(range(8)), chunk=8)
    assert groups == [f"g{i}" for i in range(8)]
    assert prompts == [f"p{i}" for i in range(8)]
    assert calls[0] == 8 and calls[1] == 4 and calls[2] == 2  # halving path


def test_chunked_backoff_reraises_non_oom():
    def sample(items):
        raise RuntimeError("something else entirely")

    with pytest.raises(RuntimeError, match="something else"):
        _chunked_backoff(sample, [1, 2], chunk=2)


def test_chunked_backoff_raises_when_single_item_ooms():
    def sample(items):
        raise RuntimeError("Insufficient Memory")

    with pytest.raises(RuntimeError, match="Insufficient Memory"):
        _chunked_backoff(sample, [1], chunk=4)


# -- honesty truncation scoring --------------------------------------------

def _bare_honesty():
    from mlx_rl.tasks.honesty import HonestyTask
    t = object.__new__(HonestyTask)
    t.P = 3.0
    t.situation = "single"
    t.needless = 0.1
    t.domain = None
    return t


def test_honesty_len_capped_scored_zero_not_minus_p():
    t = _bare_honesty()
    ex = SimpleNamespace(meta={"regime": "post", "band": "unknown", "aliases": []})
    res = t._score(ex, {"finish": "length", "tool_calls": [], "visible": ""},
                   None, None)
    assert res.total == 0.0
    assert res.parts["len_capped"] == 1.0
    assert res.parts["no_reply"] == 0.0


def test_honesty_true_no_reply_still_penalized():
    t = _bare_honesty()
    ex = SimpleNamespace(meta={"regime": "post", "band": "unknown", "aliases": []})
    res = t._score(ex, {"finish": "stop", "tool_calls": [], "visible": ""},
                   None, None)
    assert res.total == -3.0
    assert res.parts["no_reply"] == 1.0
    assert res.parts["len_capped"] == 0.0


# -- local judge rides the resident model ----------------------------------

def test_local_judge_uses_resident_model_when_base_matches(monkeypatch):
    from mlx_rl import judge_local

    used = {}

    class FakeModel:
        pass

    def fake_generate(model, tokenizer, prompt, **kw):
        used["model"] = model
        return '[{"i": 1, "kind": "abstain", "value": null}]'

    class FakeCtx:
        def __enter__(self):
            used["adapters_disabled"] = True

        def __exit__(self, *a):
            pass

    class FakeTok:
        def apply_chat_template(self, *a, **kw):
            return "rendered"

    import mlx_lm
    monkeypatch.setattr(mlx_lm, "generate", fake_generate)
    from mlx_rl import models as _models
    monkeypatch.setattr(_models, "adapters_disabled", lambda m: FakeCtx())

    fm = FakeModel()
    judge_local.register_resident_model(fm, FakeTok(), "/models/base")
    try:
        j = judge_local.LocalJudge.__new__(judge_local.LocalJudge)
        j.model_path = "/models/base"
        j.gen_tokens = 64
        j.model_name = "base"
        j.calls = 0
        j.KINDS = judge_local.LocalJudge.KINDS
        j.VALUE_KIND = judge_local.LocalJudge.VALUE_KIND
        import pathlib
        import tempfile
        j.log_path = pathlib.Path(tempfile.mkstemp()[1])
        out = j._call_once("prompt", 1)
        assert used["model"] is fm, "must generate on the resident model"
        assert used.get("adapters_disabled"), "must zero adapter scales"
        assert out[0]["kind"] == "abstain"
    finally:
        judge_local.clear_resident_model()


# -- preflight --------------------------------------------------------------

def _cfg(**kw):
    base = dict(model="/nonexistent/model", activation_headroom_gb=4.0,
                task_kwargs={}, required_gb=0.0, max_episode_tokens=6144,
                max_new_tokens=1024, steps=60, checkpoint_every=5)
    base.update(kw)
    return SimpleNamespace(**base)


def test_preflight_rejects_impossible_token_budget(capsys):
    from mlx_rl.preflight import preflight
    with pytest.raises(SystemExit) as e:
        preflight(_cfg(max_new_tokens=8192, max_episode_tokens=1024))
    assert e.value.code == 2
    assert "PREFLIGHT FAILED" in capsys.readouterr().out


def test_preflight_accepts_sane_config(capsys):
    from mlx_rl.preflight import preflight
    preflight(_cfg())  # unresolvable model -> warning, not failure
    out = capsys.readouterr().out
    assert "PREFLIGHT FAILED" not in out


def test_preflight_warns_on_wide_checkpoint_window(capsys):
    from mlx_rl.preflight import preflight
    preflight(_cfg(steps=60, checkpoint_every=30))
    assert "checkpoint_every" in capsys.readouterr().out
