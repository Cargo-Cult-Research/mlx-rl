"""Memory-bounded GatedDeltaNet scan for training — the sequence-memory fix.

Root cause (docs/memory-and-compute-anatomy.md, 2026-07-14): in training mode
`GatedDeltaNet.__call__` dispatches `use_kernel=not self.training`, so every
GDN layer abandons the fused Metal kernel (no VJP) and runs `gated_delta_ops`
— a Python loop over all T timesteps whose per-step fp32 recurrent state is
[B, Hv, Dv, Dk] = 2.10 MB/token on qwen36. The backward retains every step's
state (~4 temporaries deep): 34.3 GiB @4096 for ONE of the 30 GDN layers, vs
3.1 GiB for a full-attention layer. That scan — not attention, not the vocab
slab — was the wall probe_backward measured.

The fix: the scan becomes an mx.custom_function. Forward is opaque to
autodiff, so it can run the fused kernel (training forward = serving
forward, and faster). Backward is a hand-written VJP that walks the scan in
`chunk`-sized segments in REVERSE, recomputing each segment's
forward-for-VJP on the fly from kernel-computed boundary states.

The one non-obvious mechanism (isolated in scripts/anatomy_sched.py): with
multiple gradient outputs MLX's executor materializes one output concat tree
at a time, so any not-yet-forced gradient keeps its segment's recomputed
states alive — the live set becomes the SUM over segments, not the max.
mx.depends-chaining each segment's recompute on ALL of the previous
segment's gradients forces whole segments to retire in order: live set = one
segment. Measured on one GDN layer @4096: 34.3 -> 2.37 GiB, backward 3.5x
faster, gradients identical to the exact checkpointed reference (<=1.5e-8).

Numerics caveat: the forward output is the kernel's, the VJP recomputes with
gated_delta_ops — they agree to bf16 rounding (|dy|max ~1.6e-2 on
unit-normal synthetics), the same kernel-vs-ops difference serving already
has. old_lp / ref_lp / policy logprobs all shift together.

Scope: patches the `gated_delta_update` name inside mlx_lm.models.qwen3_5
(qwen3_5_moe reuses that module's Model — one patch covers both). Calls with
a padding mask, or off-GPU, delegate to the stock implementation; the
training path (cache=None) always sees mask=None (create_ssm_mask returns
None without a cache). use_kernel=True inference calls also route through
the kernel here, unchanged in behavior.
"""

from __future__ import annotations

import mlx.core as mx


def make_serial_update(chunk: int = 64):
    """Build a drop-in replacement for gated_delta_update whose training-mode
    (use_kernel=False) backward is memory-bounded at ~one chunk of scan
    states instead of the whole sequence."""
    from mlx_lm.models import gated_delta
    from mlx_lm.models.gated_delta import gated_delta_ops

    def seg_fn(st, qc, kc, vc, gc, bc):
        return gated_delta_ops(qc, kc, vc, gc, bc, st, None)

    @mx.custom_function
    def scan(q, k, v, g, beta, state0):
        return gated_delta.gated_delta_kernel(q, k, v, g, beta, state0)

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

    orig = gated_delta.gated_delta_update

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
               use_kernel=True):
        if (mask is not None or mx.default_device() != mx.gpu
                or not mx.metal.is_available()):
            # padded/batched-decode calls (mask from a lengths-aware cache)
            # and CPU fallbacks keep stock behavior
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask,
                        use_kernel)
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if (rep := v.shape[2] // q.shape[2]) > 1:
            # GQA head expansion OUTSIDE the custom_function: autodiff owns
            # the repeat, so dk/dq sum back over repeats automatically
            q = mx.repeat(q, rep, -2)
            k = mx.repeat(k, rep, -2)
        return scan(q, k, v, g, beta, state)

    update._mlx_rl_serial_chunk = chunk
    return update


def install(chunk: int = 64) -> bool:
    """Patch the qwen3_5 module's gated_delta_update (the name its call site
    resolves). Idempotent; returns True when a fresh patch was applied."""
    import mlx_lm.models.qwen3_5 as q35

    if getattr(q35.gated_delta_update, "_mlx_rl_serial_chunk", None) == chunk:
        return False
    q35.gated_delta_update = make_serial_update(chunk)
    return True


def uninstall() -> None:
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models import gated_delta

    q35.gated_delta_update = gated_delta.gated_delta_update
