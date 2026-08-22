"""Promote a training checkpoint to a loadable adapter directory.

Training writes step checkpoints (adapter-00060.safetensors) plus mlx-rl's own
config; mlx-lm loads adapters.safetensors plus a config in ITS shape
(fine_tune_type / num_layers / lora_parameters). This converts one to the other.

    uv run python scripts/promote_adapter.py runs/night/.../papers-single \
        --out ~/models/adapters/papers-single-60
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def promote(run_dir: Path, out: Path, step: int | None = None) -> Path:
    ckpts = sorted((run_dir / "adapters").glob("adapter-*.safetensors"))
    if not ckpts:
        raise SystemExit(f"no checkpoint under {run_dir / 'adapters'}")
    src = ckpts[-1] if step is None else run_dir / "adapters" / f"adapter-{step:05d}.safetensors"
    if not src.exists():
        raise SystemExit(f"no such checkpoint: {src}")
    cfg = json.loads((run_dir / "adapters" / "adapter_config.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out / "adapters.safetensors")
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": cfg["num_layers"],
        "lora_parameters": {k: cfg[k] for k in ("rank", "scale", "dropout", "keys")},
    }, indent=2))
    (out / "MANIFEST.md").write_text(
        f"# {out.name}\n\nPromoted from `{run_dir}` checkpoint `{src.name}`.\n"
        f"Base model: {cfg.get('model')}\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=None, help="default: newest checkpoint")
    a = ap.parse_args()
    print("promoted ->", promote(Path(a.run_dir), Path(a.out).expanduser(), a.step))


if __name__ == "__main__":
    main()
