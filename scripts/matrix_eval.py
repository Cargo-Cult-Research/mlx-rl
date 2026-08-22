"""Fill the transfer matrix: adapters (arms) × cells (domain, situation).

    uv run python scripts/matrix_eval.py --cells papers:single,papers:toolfail,trivia:single,trivia:toolfail \
        --arm base= --arm web-tools=~/models/adapters/qa-arxiv-mt-arm2-60 --n 32 --k 2

Each cell is a HonestyTask(domain, situation); the same seeded held-out items
per cell for every arm; real tools (+ controlled failures where the cell says
so); cap message; judge-graded. Writes runs/matrix/<stamp>/results.json —
{arm: {cell: {reward, base_delta, called, correct, abstain, denial,
fabricated_provenance, ...}}} — which the rl-dash matrix page renders.
Cells accept options after '@': papers:toolfail@fail_rate=0.7,pushback=1.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import mlx.core as mx  # noqa: E402
from mlx_lm import load as mlx_load  # noqa: E402

from mlx_rl import machine  # noqa: E402
from mlx_rl.config import TrainConfig  # noqa: E402
from mlx_rl.profiles import get_profile  # noqa: E402
from mlx_rl.tasks.honesty import HonestyTask  # noqa: E402
from mlx_rl.train import _episode_record, _sample_episodes, collect_multiturn  # noqa: E402

PARTS = ("called", "success", "found_target", "correct", "abstain", "denial", "no_reply",
         "claims_result", "reports_failure", "fabricated_provenance", "tool_failed",
         "named", "missing", "named_install", "missing_install",
         "items", "answered_items", "verified_items", "unbacked_items")
CALIB = {"papers": "runs/arxiv-calib-20260816/calib-strict.jsonl",
         "trivia": "runs/qa-calib-20260724/calib.jsonl"}


def parse_cell(spec: str):
    name, _, opts = spec.partition("@")
    domain, situation = name.split(":")
    kw = {}
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=")
        kw[k] = float(v) if v.replace(".", "", 1).isdigit() else v
    if "pushback" in kw:
        kw["pushback"] = bool(int(kw["pushback"]))
    return domain, situation, kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--arm", action="append", required=True, help="name=adapter_dir ('' = base)")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=f"runs/matrix/{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()
    prof = get_profile(a.profile)
    cells = [parse_cell(c) for c in a.cells.split(",")]
    tasks, examples = {}, {}
    for domain, situation, kw in cells:
        key = f"{domain}:{situation}" + ("@" + ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
        if domain in CALIB:          # packages needs no calibration file
            kw = {"calib_file": CALIB[domain], **kw}
        t = HonestyTask(domain=domain, situation=situation, **kw)
        rng = random.Random(a.seed)
        tasks[key] = t
        examples[key] = [t.eval_sample(rng) for _ in range(a.n)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [(n, str(Path(p).expanduser()) if p else None) for n, _, p in (s.partition("=") for s in a.arm)]
    results: dict = {"n": a.n, "k": a.k, "cells": list(tasks), "arms": {}}
    holder = None if a.no_manage_machine else machine.acquire(38.0, note="matrix eval")
    try:
        with (out / "episodes.jsonl").open("w") as f:
            for name, adapter in arms:
                model, tokenizer = mlx_load(prof.model, adapter_path=adapter)
                results["arms"][name] = {"adapter": adapter, "cells": {}}
                for key, task in tasks.items():
                    t0 = time.time()
                    cfg = TrainConfig(model=prof.model, task="honesty", profile=a.profile,
                                      chat_kwargs=dict(prof.chat_kwargs), max_new_tokens=a.max_new_tokens,
                                      max_tool_rounds=getattr(task, "tool_rounds", 4), think_end=prof.think_end,
                                      extra_eos=tuple(prof.extra_eos), rollout_batch_size=a.batch)
                    exs = examples[key]
                    rows = []
                    if task.turns > 1:
                        from dataclasses import replace as _replace
                        rolls, _, _ = collect_multiturn(model, tokenizer, exs,
                                                        _replace(cfg, group_size=a.k, temperature=1.0), task)
                        for r in rolls:
                            rows.append((r.meta, r.reward, r.reward_parts, r.text[-300:], r.tool_calls))
                    else:
                        per = max(1, a.batch // a.k)
                        groups = []
                        lo = 0
                        while lo < len(exs):
                            # Memory does not accumulate across cells (measured: active
                            # returns to its post-load value every time). What kills a run
                            # is the TRANSIENT peak on an expensive cell -- long episodes,
                            # many tool rounds, six-item swamped replies -- so back the
                            # chunk off until it fits instead of failing the whole arm.
                            n_try = per
                            while True:
                                try:
                                    g, _, _ = _sample_episodes(model, tokenizer, exs[lo:lo + n_try],
                                                               cfg, task, a.k, 1.0)
                                    break
                                except RuntimeError as e:
                                    if "Insufficient Memory" not in str(e) and "out of memory" not in str(e).lower():
                                        raise
                                    mx.clear_cache()
                                    if n_try == 1:
                                        raise
                                    n_try = max(1, n_try // 2)
                                    print(f"   [oom] retrying chunk at {n_try} prompts", flush=True)
                            groups.extend(g)
                            lo += n_try
                            mx.clear_cache()
                        fx, frec = [], []
                        for ex, group in zip(exs, groups):
                            for ep in group:
                                fx.append(ex)
                                frec.append(_episode_record(tokenizer, ep, None))
                        for ex, rec, res in zip(fx, frec, task.episode_reward(fx, frec)):
                            rows.append((ex.meta, res.total, res.parts, rec["visible"], rec["tool_calls"]))  # full text: re-gradable offline
                    agg = {"n": len(rows), "reward": sum(r[1] for r in rows) / max(1, len(rows))}
                    for p in PARTS:
                        agg[p] = sum(r[2].get(p, 0.0) for r in rows) / max(1, len(rows))
                    by_turn = {}
                    for meta, rew, parts, _, _ in rows:
                        by_turn.setdefault(meta.get("turn", 0), []).append(rew)
                    agg["by_turn"] = {str(t): sum(v) / len(v) for t, v in by_turn.items()}
                    agg["wall_s"] = round(time.time() - t0)
                    results["arms"][name]["cells"][key] = agg
                    for meta, rew, parts, vis, calls in rows:
                        f.write(json.dumps({"arm": name, "cell": key, "meta": meta, "reward": rew,
                                            "parts": parts, "visible": vis, "tool_calls": calls},
                                           ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"== {name:12s} {key:32s} reward {agg['reward']:+.2f}  called {agg['called']:.2f} "
                          f"correct {agg['correct']:.2f} abstain {agg['abstain']:.2f} denial {agg['denial']:.2f} "
                          f"fab_prov {agg['fabricated_provenance']:.2f} reports_fail {agg['reports_failure']:.2f} "
                          f"noreply {agg['no_reply']:.2f}  named {agg['named']:.2f} "
                          f"nonexistent {agg['missing']:.2f} (install-only {agg['missing_install']:.2f})  ({agg['wall_s']}s)", flush=True)
                    (out / "results.json").write_text(json.dumps(results, indent=1))
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
    finally:
        if holder:
            machine.release(holder)
    # base deltas
    base = results["arms"].get("base", {}).get("cells", {})
    for name, arm in results["arms"].items():
        for key, agg in arm["cells"].items():
            if key in base:
                agg["base_delta"] = round(agg["reward"] - base[key]["reward"], 3)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    import os
    import traceback
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        os._exit(code)
