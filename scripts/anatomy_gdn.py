"""GDN-scan backward memory isolation — testing THE hypothesis for where
the training memory goes.

Finding that motivates this (2026-07-14): `GatedDeltaNet.__call__` dispatches
`use_kernel=not self.training` — in TRAINING mode every GDN layer abandons the
fused Metal kernel (not differentiable) and runs `gated_delta_ops`, a Python
loop over all T timesteps whose per-step fp32 recurrent state is
[B, Hv=32, Dv=128, Dk=128] = 2.10 MB. During a checkpointed layer's backward
the recompute rebuilds the scan and every per-step state stays live for the
VJP — predicted ~2.1 MB x S (x a temporaries multiplier) PER GDN LAYER, vs
4 KB/token for the residual stream. If true, this — not attention, not the
vocab slab — is the sequence-memory wall measured in probe_backward.

This probe needs NO model weights (a single synthetic layer with the real
config dims, random init) — it runs next to the serving stack without taking
the memory lease.

Measures, per sequence length:
  A. one GDN DecoderLayer  (mx.checkpoint-wrapped, like training)
  B. one full-attn DecoderLayer (same wrapping) — the contrast arm
  C. GDN layer with the scan chunk-checkpointed (the candidate fix):
     mx.checkpoint per chunk of the scan -> only chunk-boundary states are
     retained; verified numerically against A (same loss, same input grad).

Usage:
    .venv/bin/python scripts/anatomy_gdn.py --lengths 1024,2048,3584
    .venv/bin/python scripts/anatomy_gdn.py --moe   # 4-bit MoE mlp variant
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import gated_delta
from mlx_lm.models.gated_delta import gated_delta_ops, gated_delta_update
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

CONFIG = Path.home() / "models/mlx/Qwen3.6-35B-A3B-4bit/config.json"


def build_layer(args: TextModelArgs, layer_idx: int, quantize_moe: bool):
    layer = DecoderLayer(args, layer_idx)
    if quantize_moe:
        nn.quantize(layer, bits=4, group_size=64,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized"))
    layer.set_dtype(mx.bfloat16)
    layer.train()  # the whole point: training mode forces the ops scan
    return layer


def measure(fn, x):
    """Peak GiB + seconds for one grad-through-layer backward."""
    grad_fn = mx.grad(lambda x_: (fn(x_).astype(mx.float32) ** 2).mean())
    # warmup compiles kernels for the shape; timed run measures peak
    for _ in range(2):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        g = grad_fn(x)
        mx.eval(g)
        dt = time.time() - t0
        peak = mx.get_peak_memory() / 1024**3
        del g
    mx.clear_cache()
    return peak, dt


def chunked_ops(q, k, v, g, beta, state, mask=None, chunk=64):
    """gated_delta_ops with the scan checkpointed per chunk: retain only
    chunk-boundary states; within-chunk states are recomputed in the
    backward. Drops retained-scan memory from O(S) x 2.1 MB to
    O(S/chunk) x 2.1 MB + one chunk's transients."""
    assert mask is None, "isolation probe runs unpadded"
    T = q.shape[1]

    def seg(state, qc, kc, vc, gc, bc):
        return gated_delta_ops(qc, kc, vc, gc, bc, state, None)

    seg_ckpt = mx.checkpoint(seg)
    ys = []
    for t0 in range(0, T, chunk):
        sl = slice(t0, min(t0 + chunk, T))
        y, state = seg_ckpt(
            state, q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl])
        ys.append(y)
    return mx.concatenate(ys, axis=1), state


def gated_delta_update_chunked(q, k, v, a, b, A_log, dt_bias, state=None,
                               mask=None, use_kernel=True, chunk=64):
    """Drop-in for gated_delta_update that chunk-checkpoints the ops path."""
    beta = mx.sigmoid(b)
    g = gated_delta.compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (rep := v.shape[2] // q.shape[2]) > 1:
        q = mx.repeat(q, rep, -2)
        k = mx.repeat(k, rep, -2)
    return chunked_ops(q, k, v, g, beta, state, mask, chunk=chunk)


def make_serial_scan(chunk=64, use_kernel_fwd=True):
    """The candidate REAL fix: scan as an mx.custom_function.

    Forward: opaque to autodiff -> can use the fused Metal kernel (the fast
    inference path training currently can't touch). Backward: custom VJP
    walks the segments in REVERSE, recomputing each segment's
    forward-for-VJP on the fly, with mx.depends chaining each segment's
    recompute onto the previous segment's cotangent so the lazy scheduler
    CANNOT materialize all recomputes at once — live set = one segment,
    not the whole scan.
    """

    def seg_fn(st, qc, kc, vc, gc, bc):
        return gated_delta_ops(qc, kc, vc, gc, bc, st, None)

    @mx.custom_function
    def scan(q, k, v, g, beta, state0):
        if use_kernel_fwd and mx.metal.is_available():
            return gated_delta.gated_delta_kernel(q, k, v, g, beta, state0)
        return gated_delta_ops(q, k, v, g, beta, state0, None)

    @scan.vjp
    def scan_vjp(primals, cotangents, outputs):
        q, k, v, g, beta, state0 = primals
        dy, dstate = cotangents
        T = q.shape[1]
        n = (T + chunk - 1) // chunk
        # boundary states, recomputed cheaply (kernel, no graph retained)
        bounds = [state0]
        st = state0
        for i in range(n - 1):
            sl = slice(i * chunk, (i + 1) * chunk)
            _, st = gated_delta.gated_delta_kernel(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], st)
            bounds.append(st)
        dq = [None] * n
        dk_ = [None] * n
        dv = [None] * n
        dg = [None] * n
        db = [None] * n
        dst = dstate
        for i in reversed(range(n)):
            sl = slice(i * chunk, min((i + 1) * chunk, T))
            prim = [bounds[i], q[:, sl], k[:, sl], v[:, sl], g[:, sl],
                    beta[:, sl]]
            if i < n - 1:
                # serialize on ALL of the previous segment's grads, not just
                # the state cotangent: with multiple grad outputs the
                # executor materializes one concat tree at a time, so any
                # not-yet-forced grad keeps its segment's recomputed states
                # alive — chaining the full bundle retires each segment
                # completely before the next recompute starts
                prim = mx.depends(
                    prim, [dst, dq[i + 1], dk_[i + 1], dv[i + 1],
                           dg[i + 1], db[i + 1]])
            _, grads = mx.vjp(seg_fn, prim, [dy[:, sl], dst])
            dst = grads[0]
            dq[i], dk_[i], dv[i], dg[i], db[i] = grads[1:]
        cat = lambda xs: mx.concatenate(xs, axis=1)
        return cat(dq), cat(dk_), cat(dv), cat(dg), cat(db), dst

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
               use_kernel=True):
        assert mask is None
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if (rep := v.shape[2] // q.shape[2]) > 1:
            q = mx.repeat(q, rep, -2)
            k = mx.repeat(k, rep, -2)
        return scan(q, k, v, g, beta, state)

    return update


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lengths", default="1024,2048,3584")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--moe", action="store_true",
                    help="use the real 4-bit MoE mlp (GatherQMM VJP arm)")
    ap.add_argument("--no-outer-ckpt", action="store_true",
                    help="skip the per-layer mx.checkpoint wrap (isolates "
                         "the nesting interaction with the chunked scan)")
    a = ap.parse_args()
    lengths = [int(s) for s in a.lengths.split(",")]

    cfg = json.load(open(CONFIG))
    args = TextModelArgs.from_dict(cfg.get("text_config", cfg))
    if not a.moe:
        args.num_experts = 0  # dense small MLP: isolates the attention/scan

    mx.random.seed(0)
    gdn = build_layer(args, 0, a.moe)   # idx 0: GatedDeltaNet
    attn = build_layer(args, 3, a.moe)  # idx 3: full attention

    Hv, Dv, Dk = (args.linear_num_value_heads, args.linear_value_head_dim,
                  args.linear_key_head_dim)
    state_mb = Hv * Dv * Dk * 4 / 1e6
    print(f"config: Hv={Hv} Dv={Dv} Dk={Dk} -> scan state {state_mb:.2f} "
          f"MB/step fp32; moe={a.moe} chunk={a.chunk}", flush=True)
    print(f"{'S':>6} {'arm':>12} {'peak_GiB':>9} {'s':>7} "
          f"{'pred_scan_GB':>13}", flush=True)

    orig_update = gated_delta.gated_delta_update
    for S in lengths:
        x = mx.random.normal((1, S, args.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        pred = state_mb * S / 1e3

        wrap = (lambda f: f) if a.no_outer_ckpt else mx.checkpoint
        for name, layer in (("gdn", gdn), ("attn", attn)):
            m = "causal" if name == "attn" else None
            f = wrap(lambda x_, l=layer, m=m: l(x_, mask=m))
            peak, dt = measure(f, x)
            print(f"{S:>6} {name:>12} {peak:>9.2f} {dt:>7.1f} "
                  f"{pred if name == 'gdn' else 0:>13.1f}", flush=True)

        # candidate fix: chunk-checkpointed scan, numerics-checked
        try:
            import mlx_lm.models.qwen3_5 as q35
            q35.gated_delta_update = (
                lambda *aa, **kk: gated_delta_update_chunked(
                    *aa, **{**kk, "chunk": a.chunk}))
            f = wrap(lambda x_: gdn(x_))
            peak, dt = measure(f, x)
            print(f"{S:>6} {'gdn-chunked':>12} {peak:>9.2f} {dt:>7.1f} "
                  f"{'':>13}", flush=True)
            # numerics: same output and same input-grad as the stock path
            y_c = gdn(x)
            g_c = mx.grad(
                lambda x_: (gdn(x_).astype(mx.float32) ** 2).mean())(x)
            q35.gated_delta_update = orig_update
            y_s = gdn(x)
            g_s = mx.grad(
                lambda x_: (gdn(x_).astype(mx.float32) ** 2).mean())(x)
            mx.eval(y_c, y_s, g_c, g_s)
            dy = float(mx.abs(y_c.astype(mx.float32)
                              - y_s.astype(mx.float32)).max())
            dg = float(mx.abs(g_c.astype(mx.float32)
                              - g_s.astype(mx.float32)).max())
            gscale = float(mx.abs(g_s.astype(mx.float32)).max())
            print(f"       numerics: |dy|max={dy:.2e} |dg|max={dg:.2e} "
                  f"(grad scale {gscale:.2e})", flush=True)

            # the real candidate: custom_function scan, kernel fwd,
            # mx.depends-serialized segmented VJP
            q35.gated_delta_update = make_serial_scan(chunk=a.chunk)
            f = wrap(lambda x_: gdn(x_))
            peak, dt = measure(f, x)
            print(f"{S:>6} {'gdn-serial':>12} {peak:>9.2f} {dt:>7.1f} "
                  f"{'':>13}", flush=True)
            y_c = gdn(x)
            g_c = mx.grad(
                lambda x_: (gdn(x_).astype(mx.float32) ** 2).mean())(x)
            mx.eval(y_c, g_c)
            dy = float(mx.abs(y_c.astype(mx.float32)
                              - y_s.astype(mx.float32)).max())
            dg = float(mx.abs(g_c.astype(mx.float32)
                              - g_s.astype(mx.float32)).max())
            print(f"       serial numerics: |dy|max={dy:.2e} "
                  f"|dg|max={dg:.2e} (grad scale {gscale:.2e})", flush=True)
            # bisection: GatedDeltaNet module alone (no residual/MLP) under
            # the same serial patch — where does the retention edge live?
            f = wrap(lambda x_: gdn.linear_attn(x_))
            peak, dt = measure(f, x)
            print(f"{S:>6} {'gdnmod-serial':>12} {peak:>9.2f} {dt:>7.1f}",
                  flush=True)
            q35.gated_delta_update = orig_update
            f = wrap(lambda x_: gdn.linear_attn(x_))
            peak, dt = measure(f, x)
            print(f"{S:>6} {'gdnmod-stock':>12} {peak:>9.2f} {dt:>7.1f}",
                  flush=True)
        finally:
            q35.gated_delta_update = orig_update


if __name__ == "__main__":
    main()
