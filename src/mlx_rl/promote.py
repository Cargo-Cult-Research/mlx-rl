"""Promote a run's adapter checkpoint to an adapter library.

    uv run python -m mlx_rl.promote runs/myrun --name sage-arith
    uv run python -m mlx_rl.promote runs/foo --step 8      # non-final checkpoint

Writes <library>/<name>/ in **mlx-lm's native adapter format**
(adapters.safetensors + adapter_config.json with fine_tune_type/lora_parameters),
so the promoted adapter is directly consumable by:

    mlx_lm.server --model <base> --adapter-path <library>/<name>
    mlx_lm.load(<base>, adapter_path=...)

plus a MANIFEST.md recording provenance (base model, run config, eval
trajectory, mlx-rl commit) — the part you need later to trust the weights.

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", help="training run directory (contains adapters/)")
    p.add_argument("--name", default=None, help="library name (default: run dir name)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    a = p.parse_args()

    run = Path(a.run_dir).resolve()
    ckpts = sorted((run / "adapters").glob("adapter-*.safetensors"))
    if not ckpts:
        raise SystemExit(f"no adapter checkpoints under {run}/adapters/")
    if a.step is not None:
        want = run / "adapters" / f"adapter-{a.step:05d}.safetensors"
        if not want.exists():
            raise SystemExit(f"{want.name} not found; have: {[c.name for c in ckpts]}")
        ckpt = want
    else:
        ckpt = ckpts[-1]

    cfg = json.loads((run / "config.json").read_text())
    lora = cfg["lora"]
    name = a.name or run.name
    dest = LIBRARY / name
    if dest.exists():
        raise SystemExit(f"{dest} already exists — pick another --name or remove it")
    dest.mkdir(parents=True)

    shutil.copy2(ckpt, dest / "adapters.safetensors")
    # mlx-lm native schema (tuner/utils.py::load_adapters)
    (dest / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": lora["num_layers"],
                "lora_parameters": {
                    "rank": lora["rank"],
                    "scale": lora["scale"],
                    "dropout": lora["dropout"],
                    "keys": lora["keys"],
                },
            },
            indent=2,
        )
        + "\n"
    )

    evals = [
        json.loads(l)
        for l in (run / "metrics.jsonl").read_text().splitlines()
        if "eval_reward" in l
    ]
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    eval_lines = "\n".join(
        f"- step {m.get('step')}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in m.items()
            if k.startswith("eval_") and isinstance(v, float)
        )
        for m in evals
    )
    (dest / "MANIFEST.md").write_text(f"""# {name}

- **base model:** {cfg["model"]}
- **checkpoint:** {ckpt.name} (from {run})
- **task:** {cfg["task"]} {json.dumps(cfg["task_kwargs"])}
- **chat kwargs:** {json.dumps(cfg["chat_kwargs"])}
- **SAGE:** r={cfg.get("sage_r", 0)} m={cfg.get("sage_m")} tr={cfg.get("sage_tr")}
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
`mlx_lm.server --model {cfg["model"]} --adapter-path {dest}`
""")
    print(f"promoted {ckpt.name} -> {dest}")
    print(f"  serve: mlx_lm.server --model {cfg['model']} --adapter-path {dest}")


if __name__ == "__main__":
    main()
