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
from mlx_rl.tasks.honesty import CALIB, HonestyTask  # noqa: E402
from mlx_rl.train import _episode_record, _sample_episodes, collect_multiturn  # noqa: E402

PARTS = ("called", "success", "found_target", "correct", "abstain", "denial", "no_reply",
         "claims_result", "reports_failure", "fabricated_provenance", "tool_failed",
         "named", "missing", "named_install", "missing_install",
         "items", "answered_items", "verified_items", "unbacked_items")


def _item_se(rows) -> float:
    """Standard error of the cell's reward, clustered by item (k episodes of
    one item are correlated draws, not independent ones). Without this every
    cell mean printed as if it were exact; at n=32 the SE is ~0.3 reward —
    larger than most arm-vs-base deltas that got interpreted."""
    by_item: dict = {}
    for i, row in enumerate(rows):
        by_item.setdefault(row[0].get("question", i), []).append(row[1])
    means = [sum(v) / len(v) for v in by_item.values()]
    n = len(means)
    if n < 2:
        return 0.0
    m = sum(means) / n
    var = sum((x - m) ** 2 for x in means) / (n - 1)
    return (var / n) ** 0.5



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
    # Generous by machine rule: 768 was a three-figure cap on a 262k-context
    # model, and it manufactured results — the same base cell measured -1.67
    # at 768 and +0.02 at 2048 (round2 vs day-decisive, same items, same day).
    # A truncated completion is a void measurement, not a wrong answer.
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=f"runs/matrix/{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()
    prof = get_profile(a.profile)
    cells = [parse_cell(c) for c in a.cells.split(",")]
    tasks, examples = {}, {}
    for domain, situation, kw in cells:
        key = f"{domain}:{situation}" + ("@" + ",".join(f"{k}={v}" for k, v in kw.items()) if kw else "")
        if domain in CALIB:
            kw = {"calib_file": CALIB[domain], **kw}
        t = HonestyTask(domain=domain, situation=situation, **kw)
        rng = random.Random(a.seed)
        tasks[key] = t
        examples[key] = [t.eval_sample(rng) for _ in range(a.n)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [(n, str(Path(p).expanduser()) if p else None) for n, _, p in (s.partition("=") for s in a.arm)]
    results: dict = {"n": a.n, "k": a.k, "cells": list(tasks), "arms": {}}
    with machine.lease(38.0, "matrix eval", manage=not a.no_manage_machine):
        with (out / "episodes.jsonl").open("w") as f:
            for name, adapter in arms:
                model, tokenizer = mlx_load(prof.model, adapter_path=adapter)
                results["arms"][name] = {"adapter": adapter, "cells": {}}
                for key, task in tasks.items():
                    t0 = time.time()
                    # Web/tool traffic snapshot, so the cell's numbers carry
                    # their own weather report (live-web fallback mix differs
                    # across arms evaluated hours apart — an invisible
                    # confound until it is written down per cell).
                    ws0 = dict(task.web.stats) if getattr(task, "web", None) else {}
                    ts0 = {k: v for k, v in (getattr(task, "tool_stats", {}) or {}).items()
                           if isinstance(v, (int, float))}
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
                            batch_retried = False
                            while True:
                                try:
                                    g, _, _ = _sample_episodes(model, tokenizer, exs[lo:lo + n_try],
                                                               cfg, task, a.k, 1.0)
                                    break
                                except ValueError as e:
                                    # mlx_lm 0.31.3 BatchKVCache bookkeeping can
                                    # corrupt under episode park/re-insert (a
                                    # per-layer cache index goes negative ->
                                    # "[broadcast_shapes] ... cannot be broadcast").
                                    # Sampling-dependent: the same cell passed on
                                    # retry twice on 08-21. One retry with a fresh
                                    # generator, then fail the arm loudly.
                                    if "broadcast_shapes" not in str(e) or batch_retried:
                                        raise
                                    batch_retried = True
                                    mx.clear_cache()
                                    print(f"   [batch-cache] broadcast crash at chunk {lo}; "
                                          "retrying once with a fresh generator", flush=True)
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
                    # Length-capped episodes measure the token budget, not the
                    # policy: excluded from every mean, reported as a rate.
                    capped = [r for r in rows if r[2].get("len_capped")]
                    alive = [r for r in rows if not r[2].get("len_capped")] or rows
                    if capped:
                        print(f"   [len-capped] {len(capped)}/{len(rows)} episodes hit "
                              f"max_new_tokens={a.max_new_tokens} — excluded from means",
                              flush=True)
                    agg = {"n": len(rows), "n_graded": len(alive),
                           "len_capped": len(capped) / max(1, len(rows)),
                           "reward": sum(r[1] for r in alive) / max(1, len(alive)),
                           "reward_se": round(_item_se(alive), 4)}
                    for p in PARTS:
                        agg[p] = sum(r[2].get(p, 0.0) for r in alive) / max(1, len(alive))
                    by_turn = {}
                    for meta, rew, parts, _, _ in alive:
                        by_turn.setdefault(meta.get("turn", 0), []).append(rew)
                    agg["by_turn"] = {str(t): sum(v) / len(v) for t, v in by_turn.items()}
                    agg["wall_s"] = round(time.time() - t0)
                    if getattr(task, "web", None):
                        agg["web_stats"] = {k: v - ws0.get(k, 0)
                                            for k, v in task.web.stats.items()
                                            if isinstance(v, (int, float))}
                    ts1 = {k: v for k, v in (getattr(task, "tool_stats", {}) or {}).items()
                           if isinstance(v, (int, float))}
                    if ts1:
                        agg["tool_stats"] = {k: v - ts0.get(k, 0) for k, v in ts1.items()}
                    results["arms"][name]["cells"][key] = agg
                    for meta, rew, parts, vis, calls in rows:
                        f.write(json.dumps({"arm": name, "cell": key, "meta": meta, "reward": rew,
                                            "parts": parts, "visible": vis, "tool_calls": calls},
                                           ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"== {name:12s} {key:32s} reward {agg['reward']:+.2f}±{agg['reward_se']:.2f}  called {agg['called']:.2f} "
                          f"correct {agg['correct']:.2f} abstain {agg['abstain']:.2f} denial {agg['denial']:.2f} "
                          f"fab_prov {agg['fabricated_provenance']:.2f} reports_fail {agg['reports_failure']:.2f} "
                          f"noreply {agg['no_reply']:.2f}  named {agg['named']:.2f} "
                          f"nonexistent {agg['missing']:.2f} (install-only {agg['missing_install']:.2f})  ({agg['wall_s']}s)", flush=True)
                    # Per-cell progress goes to .partial; the real results.json
                    # appears only when the run COMPLETES. Drivers use its
                    # existence as "arm done" — a partial file made killed
                    # arms unrerunnable (skipped forever as complete).
                    (out / "results.json.partial").write_text(json.dumps(results, indent=1))
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
    # base deltas
    base = results["arms"].get("base", {}).get("cells", {})
    for name, arm in results["arms"].items():
        for key, agg in arm["cells"].items():
            if key in base:
                agg["base_delta"] = round(agg["reward"] - base[key]["reward"], 3)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.json.partial").unlink(missing_ok=True)
    print("wrote", out)


if __name__ == "__main__":
    import os
    import traceback
    code = 0
    try:
        main()
    except SystemExit as e:   # argparse --help / usage errors are not crashes
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        os._exit(code)
