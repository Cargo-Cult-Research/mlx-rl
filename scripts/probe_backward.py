"""Backward-memory probe: where is the swap cliff, really?

One fwd and one fwd+bwd of the exact training path (profile model, rank-16
LoRA on the last 12 layers, micro_batch 1) at several total sequence
lengths, with mx.clear_cache() between measurements (matching the trainer's
cache discipline). Prints per-length wall time, MLX peak GiB, and system
swap growth.

Run it on a quiet machine: other memory-hungry resident processes make the
numbers pessimistic.

Usage:
    .venv/bin/python scripts/probe_backward.py            # 1536,1792,2048,2304
    .venv/bin/python scripts/probe_backward.py --lengths 2560,3072
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_rl import machine
from mlx_rl.config import LoraConfig
from mlx_rl.grpo import grpo_objective, token_logprobs
from mlx_rl.memory import swap_used_gb
from mlx_rl.models import load_policy, selective_logprobs
from mlx_rl.profiles import get_profile

SWAP_BAIL_GB = 5.0  # stop probing longer lengths once we're clearly swapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--lengths", default="1536,1792,2048,2304")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-manage-machine", action="store_true")
    ap.add_argument("--lora-layers", type=int, default=12,
                    help="LoRA depth (last-N layers) — the backward-depth knob")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--token-subset-frac", type=float, default=0.0,
                    help="S-GRPO selective path: loss on this fraction of "
                         "positions (0 = dense). Measures what token-subset "
                         "backprop actually buys at each length.")
    ap.add_argument("--no-gdn-serial", dest="gdn_serial",
                    action="store_false",
                    help="stock GDN-scan backward instead of the serialized "
                         "custom-VJP (gdn_serial.py)")
    ap.add_argument("--gdn-chunk", type=int, default=64)
    ap.add_argument("--grad-check", action="store_true",
                    help="at the first length, also compute the stock-path "
                         "grads on identical inputs and print the max LoRA-"
                         "grad deviation (serial-vs-stock, real weights)")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    if a.gdn_serial:
        from mlx_rl import gdn_serial

        gdn_serial.install(a.gdn_chunk)

    prof = get_profile(a.profile)
    lora = LoraConfig(
        rank=a.rank, num_layers=a.lora_layers,
        keys=list(prof.lora_keys) if prof.lora_keys else None,
    )
    print(f"probe: lora_layers={a.lora_layers} rank={a.rank} "
          f"grad_checkpoint={a.grad_checkpoint} "
          f"token_subset_frac={a.token_subset_frac}", flush=True)
    holder = None
    if not a.no_manage_machine:
        holder = machine.acquire(68.1, note="backward-memory probe (v4 cap sizing)")
    try:
        model, tokenizer, info = load_policy(
            prof.model, lora, headroom_gb=1.5, grad_checkpoint=a.grad_checkpoint
        )
        print(f"loaded: {info}", flush=True)
        swap0 = swap_used_gb()

        def loss_fn(model, inp, tgt, mask, old_lp, ref_lp, adv, denom,
                    sel_idx=None):
            cur_lp = (token_logprobs(model(inp), tgt) if sel_idx is None
                      else selective_logprobs(model, inp, tgt, sel_idx))
            return grpo_objective(cur_lp, old_lp, ref_lp, adv, mask, denom, 0.2, 0.01)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        if a.grad_check and a.gdn_serial:
            # Real-weights equivalence, serial vs stock, identical inputs, at
            # a length where the stock path is safe. The two forwards differ
            # by kernel-vs-ops bf16 rounding, so grads agree to noise level,
            # not bit-exactly — report loss delta, max relative deviation,
            # and cosine over the concatenated LoRA grads.
            from mlx.utils import tree_flatten

            from mlx_rl import gdn_serial

            L = 1024
            ids = mx.random.randint(100, 10_000, (1, L))
            gc_args = (ids[:, :-1], ids[:, 1:],
                       mx.ones((1, L - 1), dtype=mx.float32),
                       mx.zeros((1, L - 1), dtype=mx.float32),
                       mx.zeros((1, L - 1), dtype=mx.float32),
                       mx.array([1.0], dtype=mx.float32), float(L))

            def flat_grads():
                (loss, _, _), grads = loss_and_grad(model, *gc_args)
                mx.eval(loss, grads)
                vec = mx.concatenate(
                    [v.reshape(-1).astype(mx.float32)
                     for _, v in tree_flatten(grads)])
                return float(loss), vec

            loss_s, g_s = flat_grads()
            gdn_serial.uninstall()
            loss_o, g_o = flat_grads()
            gdn_serial.install(a.gdn_chunk)
            cos = float((g_s @ g_o) / (mx.linalg.norm(g_s)
                                       * mx.linalg.norm(g_o)))
            scale = float(mx.abs(g_o).max()) or 1.0
            rel = float(mx.abs(g_s - g_o).max()) / scale
            print(f"grad-check @L={L}: loss serial {loss_s:.6f} vs stock "
                  f"{loss_o:.6f} (d={loss_s - loss_o:+.2e}); LoRA grads "
                  f"cosine {cos:.6f}, max rel dev {rel:.2e}", flush=True)
            mx.clear_cache()

        print(f"{'L':>6} {'fwd_s':>7} {'bwd_s':>7} {'peak_GiB':>9} {'swap_+GB':>9}")
        for L in lengths:
            ids = mx.random.randint(100, 10_000, (1, L))
            inp, tgt = ids[:, :-1], ids[:, 1:]
            adv = mx.array([1.0], dtype=mx.float32)
            denom = float(L)
            tgt_full = tgt  # old/ref scoring stays dense in the trainer
            sel_idx = None
            if a.token_subset_frac > 0:
                # uniform positions, sorted — the shape the trainer produces
                K = max(1, int(round(a.token_subset_frac * (L - 1))))
                perm = np.sort(np.random.default_rng(0).choice(
                    L - 1, size=K, replace=False))
                sel_idx = mx.array(perm[None, :])
                tgt = mx.take_along_axis(tgt, sel_idx, axis=1)
                denom = float(K)
            mask = mx.ones(tgt.shape, dtype=mx.float32)
            zeros = mx.zeros(tgt.shape, dtype=mx.float32)

            # fwd only (the old_lp / ref_lp scoring passes)
            mx.clear_cache()
            t0 = time.time()
            lp = token_logprobs(model(inp), tgt_full)
            mx.eval(lp)
            t_fwd = time.time() - t0
            del lp

            # fwd+bwd: warm-up compiles kernels for this shape, then measure
            for run in ("warmup", "timed"):
                mx.clear_cache()
                mx.reset_peak_memory()
                t0 = time.time()
                (loss, _, _), grads = loss_and_grad(
                    model, inp, tgt, mask, zeros, zeros, adv, denom, sel_idx
                )
                mx.eval(loss, grads)
                t_bwd = time.time() - t0
                peak = mx.get_peak_memory() / 1024**3
                del grads, loss
            swap_d = swap_used_gb() - swap0
            print(f"{L:>6} {t_fwd:>7.1f} {t_bwd:>7.1f} {peak:>9.1f} {swap_d:>+9.1f}",
                  flush=True)
            mx.clear_cache()
            if swap_d > SWAP_BAIL_GB:
                print(f"swap grew {swap_d:.1f} GB > {SWAP_BAIL_GB} — cliff found, "
                      "skipping longer lengths", flush=True)
                break
    finally:
        machine.release(holder)


if __name__ == "__main__":
    main()
