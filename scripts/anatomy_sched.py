"""Scheduler-mechanism repro: can a segmented reverse-scan backward be
memory-bounded in MLX, and what does it take?

Context (anatomy_gdn.py): the training-mode GDN scan backward
holds ~4 fp32 state temporaries x every timestep live (~17 GiB @2048 for ONE
layer). Segmenting with mx.checkpoint leaves a ~10 GiB floor; a custom-VJP
with mx.depends chaining left ~17 GiB. Question: is the floor the lazy
scheduler's execution ORDER (fixable with graph edges / eval barriers) or
something else?

Three variants of the identical reversed per-segment VJP loop, built OUTSIDE
any transform (dy given):
  free   — no ordering constraint (the naive graph)
  deps   — segment i's recompute inputs mx.depends on segment i+1's grads
  eval   — mx.eval after each segment (hard barrier; upper bound on what
           any scheduling fix can achieve)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import gated_delta
from mlx_lm.models.gated_delta import gated_delta_ops

CONFIG = (Path(os.environ.get("MLX_RL_MODELS_DIR",
                              str(Path.home() / "models/mlx")))
          / "Qwen3.6-35B-A3B-4bit" / "config.json")


def seg_fn(st, qc, kc, vc, gc, bc):
    return gated_delta_ops(qc, kc, vc, gc, bc, st, None)


def run(S, chunk, mode, Hv, Hk, Dk, Dv):
    mx.random.seed(0)
    B = 1
    q = mx.random.normal((B, S, Hv, Dk)).astype(mx.bfloat16)  # pre-repeated
    k = mx.random.normal((B, S, Hv, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, S, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(0.8, 1.0, (B, S, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.1, 0.9, (B, S, Hv)).astype(mx.float32)
    state0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    dy = mx.random.normal((B, S, Hv, Dv)).astype(mx.bfloat16)
    mx.eval(q, k, v, g, beta, dy)

    n = (S + chunk - 1) // chunk
    # boundary states via the fused kernel (cheap, no graph)
    bounds = [state0]
    st = state0
    for i in range(n - 1):
        sl = slice(i * chunk, (i + 1) * chunk)
        _, st = gated_delta.gated_delta_kernel(
            q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], st)
        bounds.append(st)
    mx.eval(*bounds)

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.time()
    dst = mx.zeros_like(state0)
    outs = []
    for i in reversed(range(n)):
        sl = slice(i * chunk, min((i + 1) * chunk, S))
        prim = [bounds[i], q[:, sl], k[:, sl], v[:, sl], g[:, sl],
                beta[:, sl]]
        if mode == "deps" and i < n - 1:
            prim = mx.depends(prim, [dst])
        _, grads = mx.vjp(seg_fn, prim, [dy[:, sl], dst])
        dst = grads[0]
        outs.append(grads[1])  # keep dq segments (like the real vjp would)
        if mode == "eval":
            mx.eval(dst, grads[1])
    mx.eval(dst, *outs)
    dt = time.time() - t0
    return mx.get_peak_memory() / 1024**3, dt


def run_in_grad(S, chunk, Hv, Hk, Dk, Dv):
    """Same reversed segment loop, but as a custom_function vjp under
    mx.grad — bisects whether the transform context causes the retention."""
    mx.random.seed(0)
    B = 1
    q = mx.random.normal((B, S, Hv, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((B, S, Hv, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((B, S, Hv, Dv)).astype(mx.bfloat16)
    g = mx.random.uniform(0.8, 1.0, (B, S, Hv)).astype(mx.float32)
    beta = mx.random.uniform(0.1, 0.9, (B, S, Hv)).astype(mx.float32)
    state0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    mx.eval(q, k, v, g, beta)

    @mx.custom_function
    def scan(q, k, v, g, beta, st0):
        return gated_delta.gated_delta_kernel(q, k, v, g, beta, st0)

    @scan.vjp
    def scan_vjp(primals, cotangents, outputs):
        q, k, v, g, beta, st0 = primals
        dy, dstate = cotangents
        n = (S + chunk - 1) // chunk
        bounds = [st0]
        st = st0
        for i in range(n - 1):
            sl = slice(i * chunk, (i + 1) * chunk)
            _, st = gated_delta.gated_delta_kernel(
                q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], st)
            bounds.append(st)
        dst = dstate
        outs = [[] for _ in range(5)]
        for i in reversed(range(n)):
            sl = slice(i * chunk, min((i + 1) * chunk, S))
            prim = [bounds[i], q[:, sl], k[:, sl], v[:, sl], g[:, sl],
                    beta[:, sl]]
            if i < n - 1:
                prim = mx.depends(prim, [dst])
            _, grads = mx.vjp(seg_fn, prim, [dy[:, sl], dst])
            dst = grads[0]
            for j in range(5):
                outs[j].append(grads[j + 1])
        cat = lambda xs: mx.concatenate(xs[::-1], axis=1)
        return tuple(cat(o) for o in outs) + (dst,)

    def loss(q):
        y, _ = scan(q, k, v, g, beta, state0)
        return (y.astype(mx.float32) ** 2).mean()

    for _ in range(2):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        gq = mx.grad(loss)(q)
        mx.eval(gq)
        dt = time.time() - t0
        peak = mx.get_peak_memory() / 1024**3
        del gq
    mx.clear_cache()
    return peak, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()

    cfg = json.load(open(CONFIG)).get("text_config")
    Hv, Hk = cfg["linear_num_value_heads"], cfg["linear_num_key_heads"]
    Dk, Dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    print(f"S={a.length} chunk={a.chunk} Hv={Hv} Dk={Dk} Dv={Dv}")
    print(f"{'mode':>6} {'peak_GiB':>9} {'s':>7}")
    for mode in ("free", "deps", "eval"):
        peak, dt = run(a.length, a.chunk, mode, Hv, Hk, Dk, Dv)
        print(f"{mode:>6} {peak:>9.2f} {dt:>7.1f}", flush=True)
        mx.clear_cache()
    peak, dt = run_in_grad(a.length, a.chunk, Hv, Hk, Dk, Dv)
    print(f"{'grad':>6} {peak:>9.2f} {dt:>7.1f}  (same loop as custom vjp "
          "under mx.grad)", flush=True)


if __name__ == "__main__":
    main()
