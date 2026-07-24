import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_rl.engine import clone_cache_list


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
