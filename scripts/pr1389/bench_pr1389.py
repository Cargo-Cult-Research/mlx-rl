"""Benchmark upstream ml-explore/mlx-lm PR #1389 (chunk-parallel UT/WY gated
delta) against mlx-rl's gdn_serial on the anatomy harness: one training-mode
GDN DecoderLayer, real qwen36 dims, peak GiB + wall time + grad numerics.

Decision input for: adopt #1389 locally vs keep gdn_serial until it merges.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import mlx.core as mx
import mlx.nn as nn

import mlx_lm.models.qwen3_5 as q35
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

from mlx_rl import gdn_serial

# PR #1389's gated_delta.py, fetched at head 6fc3a29
spec = importlib.util.spec_from_file_location(
    "gd1389", Path(__file__).parent / "gated_delta_pr1389.py")
gd1389 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd1389)

CONFIG = (Path(os.environ.get("MLX_RL_MODELS_DIR",
                              str(Path.home() / "models/mlx")))
          / "Qwen3.6-35B-A3B-4bit" / "config.json")


def measure(fn, x):
    grad_fn = mx.grad(lambda x_: (fn(x_).astype(mx.float32) ** 2).mean())
    for _ in range(2):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        g = grad_fn(x)
        mx.eval(g)
        dt = time.time() - t0
        peak = mx.get_peak_memory() / 1024**3
    mx.clear_cache()
    return peak, dt, g


def main():
    cfg = json.load(open(CONFIG))
    args = TextModelArgs.from_dict(cfg["text_config"])
    args.num_experts = 0  # dense small MLP, same as the anatomy runs

    mx.random.seed(0)
    layer = DecoderLayer(args, 0)  # GDN layer
    layer.set_dtype(mx.bfloat16)
    layer.train()

    orig = q35.gated_delta_update
    print(f"{'S':>6} {'arm':>12} {'peak_GiB':>9} {'s':>7}")
    for S in (1024, 2048, 4096):
        x = mx.random.normal((1, S, args.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        f = mx.checkpoint(lambda x_: layer(x_))

        # arm 1: our serial scan
        q35.gated_delta_update = gdn_serial.make_serial_update(chunk=64)
        p, t, g_serial = measure(f, x)
        print(f"{S:>6} {'serial':>12} {p:>9.2f} {t:>7.2f}", flush=True)

        # arm 2: PR #1389 chunked UT/WY (their training fallback path)
        q35.gated_delta_update = gd1389.gated_delta_update
        p, t, g_1389 = measure(f, x)
        print(f"{S:>6} {'pr1389':>12} {p:>9.2f} {t:>7.2f}", flush=True)

        # numerics: both vs the stock sequential ops path
        q35.gated_delta_update = orig
        p, t, g_stock = measure(f, x)
        print(f"{S:>6} {'stock':>12} {p:>9.2f} {t:>7.2f}", flush=True)
        scale = float(mx.abs(g_stock.astype(mx.float32)).max()) or 1.0
        for name, g in (("serial", g_serial), ("pr1389", g_1389)):
            d = float(mx.abs(g.astype(mx.float32)
                             - g_stock.astype(mx.float32)).max())
            print(f"       |dgrad| {name} vs stock: {d:.2e} "
                  f"(rel {d / scale:.2e})", flush=True)
        del g_serial, g_1389, g_stock
        mx.clear_cache()
    q35.gated_delta_update = orig


if __name__ == "__main__":
    main()
