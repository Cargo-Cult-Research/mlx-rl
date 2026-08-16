import time
import types

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.generate import BatchGenerator, BatchStats
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_rl.engine import _credit_prompt, clone_cache_list


def _kv(batch=1, heads=2, seq=4, dim=8):
    c = KVCache()
    k = mx.random.normal((batch, heads, seq, dim))
    v = mx.random.normal((batch, heads, seq, dim))
    c.update_and_fetch(k, v)
    return c


def test_clone_kvcache_isolated():
    base = _kv(seq=4)
    (clone,) = clone_cache_list([base])
    # advance only the clone
    clone.update_and_fetch(
        mx.random.normal((1, 2, 3, 8)), mx.random.normal((1, 2, 3, 8))
    )
    assert base.offset == 4
    assert clone.offset == 7
    # base's visible KV unchanged
    k_base, _ = base.state
    assert k_base.shape[2] == 4


def test_clone_shares_prompt_kv_physically():
    base = _kv(seq=4)
    (clone,) = clone_cache_list([base])
    # before either writes, the underlying buffers are the same object
    assert clone.keys is base.keys
    # and the visible prefix stays numerically identical after clone advances
    clone.update_and_fetch(mx.ones((1, 2, 1, 8)), mx.ones((1, 2, 1, 8)))
    np.testing.assert_allclose(
        np.array(base.state[0]), np.array(clone.state[0][..., :4, :]), rtol=0
    )


def test_clone_arrayscache_isolated():
    base = ArraysCache(size=2)
    base[0] = mx.zeros((1, 3))
    (clone,) = clone_cache_list([base])
    clone[0] = mx.ones((1, 3))
    assert float(base[0].sum()) == 0.0
    assert float(clone[0].sum()) == 3.0


def test_clone_list_returns_independent_objects():
    caches = [_kv(), ArraysCache(size=1)]
    clones = clone_cache_list(caches)
    assert len(clones) == 2
    assert all(a is not b for a, b in zip(clones, caches))


def _stats_over(work):
    """Run `work(gen)` inside the real BatchGenerator.stats contextmanager.

    Driven with a bare namespace as `self` — stats() only touches the three
    _*_counter attributes, so this exercises upstream's actual generation_time
    formula without loading a model. The generator's own 1-token prompt pass
    is stood in for by a nominal credit so prompt_tps never divides by zero.
    """
    gen = types.SimpleNamespace()
    stats = BatchStats()
    with BatchGenerator.stats(gen, stats):
        _credit_prompt(gen, 1, 1e-6)
        work(gen)
        gen._gen_tokens_counter = 8
    return stats


def test_shared_prefill_is_booked_as_prompt_not_generation():
    """Regression: an out-of-band prefill must not be charged to decode.

    BatchGenerator.stats derives generation_time as (wall in block) minus
    _prompt_time_counter, so a prefill done before insert() lands in
    generation_time unless credited — which reads as a decode slowdown
    scaling linearly with context length.
    """
    def credited(gen):
        t = time.perf_counter()
        time.sleep(0.20)                      # stand-in for prefill_cache
        _credit_prompt(gen, 1000, time.perf_counter() - t)
        time.sleep(0.05)                      # stand-in for decode

    stats = _stats_over(credited)
    assert stats.prompt_tokens == 1001
    assert stats.prompt_time == pytest.approx(0.20, abs=0.05)
    # decode window only — the 0.2s prefill is excluded
    assert stats.generation_time == pytest.approx(0.05, abs=0.05)
    assert stats.generation_tps > 50          # 8 tok / ~0.05s, not 8 / 0.25s


def test_uncredited_prefill_would_corrupt_generation_tps():
    """Guard the premise: without the credit the artifact is present."""
    def uncredited(gen):
        time.sleep(0.20)
        time.sleep(0.05)

    stats = _stats_over(uncredited)
    assert stats.generation_time == pytest.approx(0.25, abs=0.05)
    assert stats.generation_tps < 50


def test_credit_prompt_tolerates_missing_counters():
    """No stats context (or an upstream rename) must not raise."""
    _credit_prompt(types.SimpleNamespace(), 10, 1.0)
