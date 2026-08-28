"""Retrofit the held-out subjects onto a finished run.

    uv run python scripts/retrofit_eval.py --run runs/curve/20260822-trivia-v3 \
        --also runs/curve/20260822-trivia-v3-to120 \
        --cells papers:single,packages:single --every 10 --n 64

A run only ever scored the subject it trained on, so it cannot tell learning
from memorising: a policy that picks up one domain's surface quirks and one
that learns to check before answering look identical there. Every checkpoint
is on disk, so the unseen subjects can be scored after the fact.

Two things make this cheap. The model is loaded ONCE with its LoRA layers
attached and each checkpoint is a ~17 MB load_weights, not a 22 GB reload
(what matrix_eval would do, per arm). And step 0 needs no file at all:
adapters_disabled() zeroes the LoRA scale, which is bit-identical to base
without a second copy in memory.

Protocol is deliberately the TRAINING eval's, not matrix_eval's: greedy, same
seed, same held-out split, local judge. Curves produced here therefore overlay
the run's own eval curve on one set of axes. Scored under matrix_eval's
sampled/Opus protocol they would be a different measurement that cannot be
plotted beside it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import mlx.core as mx  # noqa: E402

from mlx_rl import machine  # noqa: E402
from mlx_rl.config import LoraConfig, TrainConfig  # noqa: E402
from mlx_rl.models import adapters_disabled, load_policy  # noqa: E402
from mlx_rl.profiles import get_profile  # noqa: E402
from mlx_rl.train import build_eval_cells, evaluate  # noqa: E402


def checkpoints(run_dirs: list[Path], every: int) -> list[tuple[int, Path | None]]:
    """[(step, file)] at multiples of `every`, plus step 0 = base (None)."""
    found: dict[int, Path] = {}
    for d in run_dirs:
        for f in (d / "adapters").glob("adapter-*.safetensors"):
            found[int(f.stem.split("-")[1])] = f
    steps = [s for s in sorted(found) if s % every == 0]
    return [(0, None)] + [(s, found[s]) for s in steps]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir holding config.json")
    ap.add_argument("--also", action="append", default=[],
                    help="extra run dirs whose checkpoints continue the same curve")
    ap.add_argument("--cells", required=True, help="e.g. papers:single,packages:single")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = Path(a.run)
    src = json.loads((run / "config.json").read_text())
    prof = get_profile(a.profile)
    # Carry the run's OWN lora spec, keys included. With a different key set
    # the checkpoint's names would not match any parameter, load_weights
    # (strict=False) would quietly apply nothing, and every point on the curve
    # would be the base model wearing an adapter's label. _apply below turns
    # that silent failure into a loud one.
    lsrc = src.get("lora") or {}
    lora = LoraConfig(rank=lsrc.get("rank", 16), scale=lsrc.get("scale", 20.0),
                      dropout=lsrc.get("dropout", 0.0),
                      num_layers=lsrc.get("num_layers", 12), keys=lsrc.get("keys"))
    cfg = TrainConfig(
        model=src["model"], task="honesty", profile=a.profile,
        chat_kwargs=dict(src.get("chat_kwargs") or prof.chat_kwargs),
        task_kwargs=dict(src["task_kwargs"]),
        max_new_tokens=src.get("max_new_tokens", 1024),
        max_episode_tokens=src.get("max_episode_tokens", 6144),
        max_tool_rounds=src.get("max_tool_rounds", 4),
        rollout_batch_size=src.get("rollout_batch_size", 48),
        think_end=prof.think_end, extra_eos=tuple(prof.extra_eos),
        eval_n=a.n, eval_cells=a.cells, eval_cells_n=a.n,
        seed=src.get("seed", 0), lora=lora,
    )
    # Local judge, as the run itself used: the point is a curve comparable to
    # the run's own, and switching rulers mid-comparison would break that.
    cfg.task_kwargs["judge_backend"] = "local"

    pts = checkpoints([run] + [Path(x) for x in a.also], a.every)
    out = Path(a.out or f"runs/retrofit/{run.name}-{time.strftime('%Y%m%d-%H%M')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[retrofit] {len(pts)} points {[s for s, _ in pts]} x {a.cells} @ n={a.n}", flush=True)

    holder = machine.acquire(38.0, wait_s=3600, note="retrofit eval")
    try:
        model, tokenizer, info = load_policy(cfg.model, lora, required_gb=38.0)
        print(f"[retrofit] loaded {info.get('model_path')}", flush=True)
        cells = build_eval_cells(cfg)
        with (out / "metrics.jsonl").open("a") as f:
            for step, ckpt in pts:
                t0 = time.time()
                if ckpt is not None:
                    _apply(model, ckpt)
                rec: dict = {"step": step, "checkpoint": str(ckpt) if ckpt else "base"}
                for name, task in cells.items():
                    ctx = adapters_disabled(model) if ckpt is None else _null()
                    try:
                        with ctx:
                            r = evaluate(model, tokenizer, task, cfg)
                    except Exception as e:                      # noqa: BLE001
                        print(f"  [{step}:{name}] skipped: {type(e).__name__}: {e}", flush=True)
                        continue
                    for k, v in r.items():
                        rec[f"eval_{name}_" + (k[5:] if k.startswith("eval_") else k)] = v
                    print(f"  step {step:3d} {name:16s} reward {r.get('eval_reward', 0.0):+.3f} "
                          f"correct {r.get('eval_correct', 0.0):.2f} "
                          f"called {r.get('eval_called', 0.0):.2f}", flush=True)
                    mx.clear_cache()
                rec["wall_s"] = round(time.time() - t0, 1)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  step {step:3d} done in {rec['wall_s']}s", flush=True)
    finally:
        machine.release(holder)
    print(f"[retrofit] wrote {out}/metrics.jsonl", flush=True)


def _apply(model, ckpt: Path) -> None:
    """Load a checkpoint's LoRA weights, refusing a partial match.

    strict=False is required (the file holds only the adapters, not the base),
    which also means a wrong key set applies nothing and reports success. So
    the match is checked explicitly: anything short of every trainable
    parameter means the curve would be measuring the wrong model.
    """
    from mlx.utils import tree_flatten

    weights = dict(mx.load(str(ckpt)).items())
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    missing = [k for k in names if k not in weights]
    if missing:
        raise SystemExit(
            f"{ckpt.name}: {len(missing)}/{len(names)} LoRA parameters absent from the "
            f"checkpoint (first: {missing[0]}). The run's lora.keys do not match the "
            f"model being loaded; refusing to report base-model numbers as adapter numbers.")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
