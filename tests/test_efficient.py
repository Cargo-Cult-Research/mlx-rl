"""Efficient-RL levers (docs/efficient-rl-edge-reading.md):
two-stage group sampling (--group-stage1), advantage-based update pruning
(--update-adv-frac), and the stage-1 abandon rule."""
import random

import numpy as np
import pytest

from mlx_rl.train import _stage1_dead


# ------------------------------ stage-1 rule ------------------------------
def test_saturated_abandons_only_uniform_max():
    assert _stage1_dead([1.0, 1.0, 1.0], "saturated")
    assert not _stage1_dead([0.0, 0.0, 0.0], "saturated")  # SAGE may rescue
    assert not _stage1_dead([0.4, 0.4, 0.4], "saturated")  # improvable
    assert not _stage1_dead([1.0, 0.0, 1.0], "saturated")  # live spread


def test_uniform_abandons_any_flat_group():
    assert _stage1_dead([0.0, 0.0], "uniform")
    assert _stage1_dead([0.4, 0.4], "uniform")
    assert _stage1_dead([0.4, 0.4000001], "uniform")  # sub-eps spread = uniform
    assert not _stage1_dead([0.0, 1.0], "uniform")


# --------------------------- advantage pruning ----------------------------
def test_prune_keep_mask_matches_train_loop_math():
    # replicate the train-loop keep computation on a 2-group batch
    group_size, frac = 4, 0.5
    adv = np.array([1.5, -1.5, 0.1, -0.1,   # group 1: spread
                    0.9, -0.3, -0.3, -0.3])  # group 2: 1-vs-3 split
    a2 = np.abs(adv).reshape(-1, group_size)
    keep = (a2 >= frac * a2.max(axis=1, keepdims=True)).reshape(-1)
    assert keep.tolist() == [True, True, False, False,
                             True, False, False, False]
    # denom must be computed BEFORE pruning: full-batch truncation, so the
    # retained terms keep their weight and dropped ones contribute zero.


# --------------------------- two-stage collect -----------------------------
class _FakeTask:
    """Deterministic rewards per example: 'sat' is always 1.0 (stage-1
    saturated -> abandoned); 'live' alternates 0/1 (spread -> survives)."""
    name = "fake"

    def __init__(self):
        self.calls = 0
        self._live_next = 0.0

    def reward(self, example, completion):
        from mlx_rl.tasks.base import RewardResult
        self.calls += 1
        if example.meta["kind"] == "sat":
            return RewardResult(1.0, {"correct": 1.0})
        self._live_next = 1.0 - self._live_next
        return RewardResult(self._live_next, {"correct": self._live_next})


@pytest.mark.integration
def test_two_stage_skips_saturated_group(tiny_train):
    from mlx_rl.config import TrainConfig
    from mlx_rl.tasks.base import Example
    from mlx_rl.train import collect_rollouts

    model, tokenizer = tiny_train
    cfg = TrainConfig(group_size=4, group_stage1=2, stage1_skip="saturated",
                      max_new_tokens=16)
    task = _FakeTask()
    msg = [{"role": "user", "content": "Say something."}]
    examples = [Example(messages=msg, meta={"kind": "sat"}),
                Example(messages=msg, meta={"kind": "live"})]
    rollouts, _, skipped = collect_rollouts(model, tokenizer, examples, cfg, task)
    assert skipped == 1
    assert len(rollouts) == 4  # only the live group, full G
    # grading calls: 2+2 stage-1, then only the live group's 2 stage-2
    # members (stage-1 results come from the cache — no double grade)
    assert task.calls == 6


@pytest.mark.integration
def test_two_stage_all_dead_returns_empty(tiny_train):
    from mlx_rl.config import TrainConfig
    from mlx_rl.tasks.base import Example
    from mlx_rl.train import collect_rollouts

    model, tokenizer = tiny_train
    cfg = TrainConfig(group_size=4, group_stage1=2, stage1_skip="saturated",
                      max_new_tokens=16)
    examples = [Example(messages=[{"role": "user", "content": "hi"}],
                        meta={"kind": "sat"})]
    rollouts, _, skipped = collect_rollouts(
        model, tokenizer, examples, cfg, _FakeTask())
    assert rollouts == [] and skipped == 1


@pytest.fixture(scope="module")
def tiny_train():
    from huggingface_hub import snapshot_download

    from mlx_rl.config import LoraConfig
    from mlx_rl.models import load_policy
    from mlx_rl.profiles import get_profile

    prof = get_profile("tiny")
    try:
        snapshot_download(prof.model, local_files_only=True)
    except Exception:
        pytest.skip("tiny model not in local HF cache")
    model, tokenizer, _ = load_policy(prof.model, LoraConfig(rank=8))
    return model, tokenizer
