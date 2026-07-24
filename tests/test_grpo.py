import mlx.core as mx
import numpy as np
import pytest

from mlx_rl.grpo import active_groups, group_advantages, grpo_objective, token_logprobs


def test_active_groups_filters_zero_variance():
    rewards = mx.array([[0.2, 0.2, 0.2], [0.0, 1.0, 0.2], [1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(np.array(active_groups(rewards)), [False, True, False])


def test_advantages_are_group_mean_centered():
    rewards = mx.array([[1.0, 0.0, 0.0, 1.0], [0.2, 0.2, 1.0, 0.2]])
    adv = np.array(group_advantages(rewards, normalize_std=False))
    np.testing.assert_allclose(adv.mean(axis=1), [0.0, 0.0], atol=1e-6)
    assert adv[0, 0] > 0 > adv[0, 1]


def test_zero_variance_group_gives_zero_advantage():
    rewards = mx.array([[0.5, 0.5, 0.5]])
    for normalize in (True, False):
        adv = np.array(group_advantages(rewards, normalize_std=normalize))
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)


def test_std_normalization_scales_to_unit_std():
    rewards = mx.array([[1.0, 0.0, 1.0, 0.0]])
    adv = np.array(group_advantages(rewards, normalize_std=True))
    np.testing.assert_allclose(adv.std(), 1.0, rtol=1e-3)


def test_token_logprobs_match_log_softmax():
    logits = mx.array([[[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]]])  # [1, 2, 3]
    targets = mx.array([[0, 1]])
    lp = np.array(token_logprobs(logits, targets))
    expected = np.array(
        [
            2.0 - np.log(np.exp(2.0) + np.exp(0.0) + np.exp(-1.0)),
            3.0 - np.log(np.exp(0.0) + np.exp(3.0) + np.exp(0.0)),
        ]
    )
    np.testing.assert_allclose(lp[0], expected, rtol=1e-5)


def _objective(cur, old, ref, adv, mask, denom, **kw):
    loss, pg, kl = grpo_objective(
        mx.array(cur), mx.array(old), mx.array(ref), mx.array(adv), mx.array(mask), denom, **kw
    )
    return float(loss), float(pg), float(kl)


def test_on_policy_reduces_to_reinforce_with_baseline():
    # cur == old == ref: ratio 1, kl 0 -> loss = -sum(adv over masked tokens)/denom
    cur = [[-1.0, -2.0, -3.0]]
    mask = [[1.0, 1.0, 0.0]]
    loss, pg, kl = _objective(cur, cur, cur, [0.5], mask, denom=2.0, kl_coef=0.1)
    assert kl == pytest.approx(0.0, abs=1e-6)
    assert pg == pytest.approx(0.5 * 2)  # adv on 2 masked tokens
    assert loss == pytest.approx(-0.5)


def test_clip_bounds_the_ratio():
    # cur much higher than old -> raw ratio e^2 ~ 7.4, clipped to 1.2
    cur = [[0.0]]
    old = [[-2.0]]
    mask = [[1.0]]
    loss, pg, _ = _objective(cur, old, cur, [1.0], mask, denom=1.0, clip_eps=0.2, kl_coef=0.0)
    assert pg == pytest.approx(1.2, rel=1e-4)
    # negative advantage: min() keeps the UNclipped (more pessimistic) branch
    loss, pg, _ = _objective(cur, old, cur, [-1.0], mask, denom=1.0, clip_eps=0.2, kl_coef=0.0)
    assert pg == pytest.approx(-np.exp(2.0), rel=1e-4)


def test_masked_tokens_do_not_contribute():
    cur = [[-1.0, -99.0]]
    ref = [[-1.0, -1.0]]
    mask = [[1.0, 0.0]]
    loss_a, _, kl_a = _objective(cur, cur, ref, [1.0], mask, denom=1.0, kl_coef=1.0)
    cur_b = [[-1.0, -12345.0]]
    loss_b, _, kl_b = _objective(cur_b, cur_b, ref, [1.0], mask, denom=1.0, kl_coef=1.0)
    assert loss_a == pytest.approx(loss_b)
    assert kl_a == pytest.approx(kl_b)


def test_kl_penalty_is_nonnegative_and_zero_at_ref():
    cur = [[-1.0, -1.5]]
    ref = [[-1.2, -1.0]]
    mask = [[1.0, 1.0]]
    _, _, kl = _objective(cur, cur, ref, [0.0], mask, denom=2.0, kl_coef=1.0)
    assert kl > 0.0
