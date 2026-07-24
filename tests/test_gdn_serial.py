"""Serial GDN scan (gdn_serial.py): numerics vs the stock ops path, GQA
head expansion, mask delegation, and install idempotence.

Small dims but kernel-legal ones: gated_delta_kernel requires Dk % 32 == 0
(n_per_t = Dk/32 per thread) and launches a (32, Dv, B*Hv) grid."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_rl import gdn_serial

metal = pytest.mark.skipif(
    not mx.metal.is_available(), reason="serial scan forward needs Metal"
)

B, T, Hk, Hv, Dk, Dv = 2, 40, 2, 4, 32, 32
CHUNK = 16  # deliberately not dividing T (last segment is ragged)


def _inputs(seed=0, dtype=mx.float32):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    k = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    a = mx.random.normal((B, T, Hv)).astype(dtype)
    b = mx.random.normal((B, T, Hv)).astype(dtype)
    A_log = mx.random.normal((Hv,)).astype(dtype)
    dt_bias = mx.random.normal((Hv,)).astype(dtype)
    mx.eval(q, k, v, a, b, A_log, dt_bias)
    return q, k, v, a, b, A_log, dt_bias


def _loss(update, wrt):
    """Scalar loss through `update`, differentiable wrt the named primal."""
    q, k, v, a, b, A_log, dt_bias = _inputs()
    prims = {"q": q, "k": k, "v": v, "a": a, "b": b}

    def f(x):
        p = {**prims, wrt: x}
        y, state = update(p["q"], p["k"], p["v"], p["a"], p["b"],
                          A_log, dt_bias, use_kernel=False)
        return (y.astype(mx.float32) ** 2).mean() + (state ** 2).mean()

    return f, prims[wrt]


@metal
def test_forward_matches_ops():
    from mlx_lm.models.gated_delta import gated_delta_update

    q, k, v, a, b, A_log, dt_bias = _inputs()
    upd = gdn_serial.make_serial_update(chunk=CHUNK)
    y_s, st_s = upd(q, k, v, a, b, A_log, dt_bias, use_kernel=False)
    y_o, st_o = gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                                   use_kernel=False)
    mx.eval(y_s, y_o, st_s, st_o)
    # fp32 kernel vs fp32 ops, RELATIVE: the synthetic gate keeps g ~ 1, so
    # the state (and y) grows to ~1e4 over T=40 steps — absolute tolerances
    # are meaningless at that scale
    for s, o in ((y_s, y_o), (st_s, st_o)):
        scale = float(mx.abs(o).max()) or 1.0
        assert float(mx.abs(s - o).max()) / scale < 1e-4


@metal
@pytest.mark.parametrize("wrt", ["q", "k", "v", "a", "b"])
def test_grads_match_ops(wrt):
    from mlx_lm.models.gated_delta import gated_delta_update

    upd = gdn_serial.make_serial_update(chunk=CHUNK)
    f_s, x = _loss(upd, wrt)
    f_o, _ = _loss(gated_delta_update, wrt)
    g_s = mx.grad(f_s)(x)
    g_o = mx.grad(f_o)(x)
    mx.eval(g_s, g_o)
    scale = float(mx.abs(g_o).max()) or 1.0
    assert float(mx.abs(g_s - g_o).max()) / scale < 1e-3, wrt


@metal
def test_mask_delegates_to_stock():
    from mlx_lm.models.gated_delta import gated_delta_update

    q, k, v, a, b, A_log, dt_bias = _inputs()
    mask = mx.concatenate(
        [mx.ones((B, T - 8), dtype=mx.bool_),
         mx.zeros((B, 8), dtype=mx.bool_)], axis=1)
    upd = gdn_serial.make_serial_update(chunk=CHUNK)
    y_s, st_s = upd(q, k, v, a, b, A_log, dt_bias, mask=mask,
                    use_kernel=True)
    y_o, st_o = gated_delta_update(q, k, v, a, b, A_log, dt_bias, mask=mask,
                                   use_kernel=True)
    mx.eval(y_s, y_o, st_s, st_o)
    assert float(mx.abs(y_s - y_o).max()) == 0.0
    assert float(mx.abs(st_s - st_o).max()) == 0.0


def test_install_idempotent_and_uninstall():
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models import gated_delta

    orig = q35.gated_delta_update
    try:
        assert gdn_serial.install(chunk=CHUNK) is True
        patched = q35.gated_delta_update
        assert patched._mlx_rl_serial_chunk == CHUNK
        assert gdn_serial.install(chunk=CHUNK) is False  # no re-patch
        assert q35.gated_delta_update is patched
        assert gdn_serial.install(chunk=CHUNK * 2) is True  # new chunk
        gdn_serial.uninstall()
        assert q35.gated_delta_update is gated_delta.gated_delta_update
    finally:
        q35.gated_delta_update = orig
