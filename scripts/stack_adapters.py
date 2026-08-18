"""Stack LoRA adapters exactly: k rank-r adapters -> one rank-k·r adapter.

    delta = sum_i  scale * B_i A_i  ==  scale * [B_1 ... B_k] · [A_1; ...; A_k]

so concatenating A along rank and B along rank reproduces the sum of the
deltas exactly (same scale for all — asserted). The output is a normal mlx-lm
adapter dir (adapters.safetensors + adapter_config.json with the summed
rank) that loads anywhere a single adapter does. Missing keys in one adapter
(different layer sets) are zero-filled, which is the identity for that block.

    uv run python scripts/stack_adapters.py ~/models/adapters/A ~/models/adapters/B --out ~/models/adapters/A+B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = [Path(p).expanduser() for p in a.adapters]
    cfgs = [json.loads((d / "adapter_config.json").read_text()) for d in dirs]
    scales = {c["lora_parameters"]["scale"] for c in cfgs}
    assert len(scales) == 1, f"adapters must share a LoRA scale, got {scales}"
    ranks = [c["lora_parameters"]["rank"] for c in cfgs]
    weights = [dict(mx.load(str(d / "adapters.safetensors")).items()) for d in dirs]
    keys = sorted({k for w in weights for k in w})
    bases = sorted({k[: -len(".lora_a")] for k in keys if k.endswith(".lora_a")})
    out = {}
    for base in bases:
        As, Bs = [], []
        for w, r in zip(weights, ranks):
            ka, kb = base + ".lora_a", base + ".lora_b"
            if ka in w:
                As.append(w[ka]); Bs.append(w[kb])
            else:  # this adapter didn't touch the layer: zero block (identity)
                shape_a = next(x[ka].shape for x in weights if ka in x)
                shape_b = next(x[kb].shape for x in weights if kb in x)
                As.append(mx.zeros((shape_a[0], r), dtype=mx.float32))
                Bs.append(mx.zeros((r, shape_b[1]), dtype=mx.float32))
        out[base + ".lora_a"] = mx.concatenate(As, axis=1)   # (in, sum r)
        out[base + ".lora_b"] = mx.concatenate(Bs, axis=0)   # (sum r, out)
    dest = Path(a.out).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dest / "adapters.safetensors"), out)
    cfg = json.loads(json.dumps(cfgs[0]))
    cfg["lora_parameters"]["rank"] = sum(ranks)
    cfg["num_layers"] = max(c.get("num_layers", 0) for c in cfgs)
    cfg["stacked_from"] = [str(d) for d in dirs]
    (dest / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (dest / "MANIFEST.md").write_text("Stacked (exact rank-concatenation) from:\n" +
                                      "".join(f"- {d}\n" for d in dirs))
    print(f"stacked {len(dirs)} adapters (ranks {ranks} -> {sum(ranks)}) -> {dest}")


if __name__ == "__main__":
    main()
