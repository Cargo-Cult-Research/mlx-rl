"""Promote a training checkpoint to a loadable adapter directory.

Training writes step checkpoints (adapter-00060.safetensors) plus mlx-rl's own
config; mlx-lm loads adapters.safetensors plus a config in ITS shape
(fine_tune_type / num_layers / lora_parameters). This converts one to the
other and writes a MANIFEST.md with the provenance you need later to trust
the weights: base model, task, eval trajectory, mlx-rl commit.

    uv run python scripts/promote_adapter.py runs/myrun --name sage-arith   # -> library
    uv run python scripts/promote_adapter.py runs/myrun --out runs/myrun/promoted
    uv run python scripts/promote_adapter.py runs/myrun --step 8 ...        # non-final ckpt

The library defaults to ~/models/adapters and can be overridden with
MLX_RL_ADAPTERS_DIR. Runs stay in runs/ (gitignored, disposable); the library
is for adapters that earned a name. Regression tiers before serving one for
real: see "Adapter lifecycle" in README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

LIBRARY = Path(os.environ.get("MLX_RL_ADAPTERS_DIR", str(Path.home() / "models" / "adapters")))


def promote(run_dir: Path, out: Path, step: int | None = None) -> Path:
    ckpts = sorted((run_dir / "adapters").glob("adapter-*.safetensors"))
    if not ckpts:
        # exit 2 = "nothing to promote", the ONLY status a driver may
        # swallow. Every other failure (missing config, bad perms) exits 1
        # so `|| say "nothing to promote"` stops eating real errors.
        raise SystemExit(2)
    src = ckpts[-1] if step is None else run_dir / "adapters" / f"adapter-{step:05d}.safetensors"
    if not src.exists():
        raise SystemExit(f"no such checkpoint: {src}; have {[c.name for c in ckpts]}")
    cfg = json.loads((run_dir / "adapters" / "adapter_config.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out / "adapters.safetensors")
    # mlx-lm native schema (tuner/utils.py::load_adapters)
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": cfg["num_layers"],
        "lora_parameters": {k: cfg[k] for k in ("rank", "scale", "dropout", "keys")},
    }, indent=2) + "\n")
    (out / "MANIFEST.md").write_text(_manifest(run_dir, src, cfg, out))
    return out


def _manifest(run_dir: Path, ckpt: Path, lora: dict, dest: Path) -> str:
    run_cfg = {}
    if (run_dir / "config.json").exists():
        run_cfg = json.loads((run_dir / "config.json").read_text())
    model = run_cfg.get("model", lora.get("model"))
    evals = []
    if (run_dir / "metrics.jsonl").exists():
        evals = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()
                 if "eval_reward" in l]
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- git missing is not a promotion failure
        commit = "unknown"
    eval_lines = "\n".join(
        f"- step {m.get('step')}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in m.items() if k.startswith("eval_") and isinstance(v, float))
        for m in evals)
    return f"""# {dest.name}

- **base model:** {model}
- **checkpoint:** {ckpt.name} (from {run_dir})
- **task:** {run_cfg.get("task")} {json.dumps(run_cfg.get("task_kwargs", {}))}
- **chat kwargs:** {json.dumps(run_cfg.get("chat_kwargs", {}))}
- **LoRA:** rank {lora["rank"]}, scale {lora["scale"]}, last {lora["num_layers"]} layers, keys {lora["keys"]}
- **mlx-rl commit:** {commit}
- **promoted:** {time.strftime("%Y-%m-%d %H:%M")}

## Held-out eval trajectory (greedy, plain decoding)

{eval_lines}

## Regression status

- [ ] Tier 0 — off-task in-repo check (toolformat canonical rate)
- [ ] Tier 1 — single-shot coding slice (external harness, mlx_lm.server + this adapter)
- [ ] Tier 2 — agentic coding slice (external harness + this adapter)

Serve for validation:
`mlx_lm.server --model {model} --adapter-path {dest}`
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", help="explicit destination directory")
    dest.add_argument("--name", help=f"library name: writes {LIBRARY}/<name> (must not exist)")
    ap.add_argument("--step", type=int, default=None, help="default: newest checkpoint")
    a = ap.parse_args()
    out = Path(a.out).expanduser() if a.out else LIBRARY / a.name
    if a.name and out.exists():
        raise SystemExit(f"{out} already exists — pick another --name or remove it")
    try:
        print("promoted ->", promote(Path(a.run_dir), out, a.step))
    except SystemExit as e:
        if e.code == 2:
            print(f"nothing to promote: no checkpoint under {a.run_dir}/adapters")
        raise


if __name__ == "__main__":
    main()
