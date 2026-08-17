"""Transfer eval: how do adapters trained against DIFFERENT tool backends do
on the same realistic questions with the REAL tools?

The question this answers (raised 2026-08-16): the snapshot "sandbox" backend
(clean empties, date gate, title index) is a shortcut — but a shortcut can
transfer cleanly, and if it does it is the cheaper curriculum. Nobody gets
to assume either way; this measures it. Every arm sees the SAME seeded
question set from qa_arxiv's held-out split, with backend="web"
(mlx_rl.webtools: DuckDuckGo + fetch_url, cached), and is scored by the
same reward. Cost is reported next to reward — generated tokens and tool
rounds per episode — because "makes sense of the noise" and "burns 2k
tokens doing it" are different results.

    uv run python scripts/arxiv_transfer_eval.py --n 64 --k 2 \
        --arm base= \
        --arm sandbox-v3-60=~/models/adapters/qa-arxiv-sandbox-v3-60 \
        --arm web-v4-60=~/models/adapters/qa-arxiv-web-v4-60

Arms load one at a time (one resident model). Output: per-arm × regime
table + runs/arxiv-transfer-<stamp>/{episodes.jsonl,summary.json}.
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
from mlx_rl.tasks.qa_arxiv import QAArxivTask  # noqa: E402
from mlx_rl.train import _episode_record, _sample_episodes, collect_multiturn  # noqa: E402

PARTS = ("called", "found_target", "grounded", "answered", "correct", "abstain",
         "denial", "no_reply", "checked_absent")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--arm", action="append", required=True,
                    help="name=adapter_dir (empty dir = base). Repeatable.")
    ap.add_argument("--calib", default="runs/arxiv-calib-20260816/calib-strict.jsonl")
    ap.add_argument("--snapshot", default="data/arxiv_snapshot.jsonl")
    ap.add_argument("--webcache", default="runs/webcache")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=2, help="samples per question (temp 1); 0 = 1 greedy")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-tool-rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32,
                    help="rows in flight (64 long real-tool rows OOM'd Metal on 2026-08-17)")
    ap.add_argument("--regime-mix", default=None, help='JSON, e.g. {"known":.2,"uncertain":.2,"unknown":.3,"fictional":.3}')
    ap.add_argument("--thinking", action="store_true",
                    help="serve with the profile's thinking mode on (grading after the "
                         "final </think>; unclosed think = no reply)")
    ap.add_argument("--no-tools", action="store_true",
                    help="serve WITHOUT tools offered (the no-tool cell of the grid)")
    ap.add_argument("--turns", type=int, default=1,
                    help=">1: multi-turn transcripts (each member carries its own history); "
                         "per-turn breakdown reported")
    ap.add_argument("--out", default=f"runs/arxiv-transfer-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    prof = get_profile(a.profile)
    kw = {}
    if a.regime_mix:
        kw["regime_mix"] = json.loads(a.regime_mix)
    task = QAArxivTask(snapshot=a.snapshot, backend="web", webcache_dir=a.webcache,
                       calib_file=a.calib, turns=a.turns, **kw)
    rng = random.Random(a.seed)
    examples = [task.eval_sample(rng) for _ in range(a.n)]  # SAME set for every arm
    ck = dict(prof.chat_kwargs)
    if a.thinking:
        ck.update(prof.think_chat_kwargs)
    if a.no_tools:
        task.tools = []
        for ex in examples:
            ex.chat_kwargs = {}
    cfg = TrainConfig(model=prof.model, task="qa_arxiv", profile=a.profile,
                      chat_kwargs=ck, max_new_tokens=a.max_new_tokens,
                      max_tool_rounds=a.max_tool_rounds, think_end=prof.think_end,
                      extra_eos=tuple(prof.extra_eos), rollout_batch_size=a.batch)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in a.arm:
        name, _, path = spec.partition("=")
        arms.append((name, str(Path(path).expanduser()) if path else None))
    k, temp = (a.k, 1.0) if a.k > 0 else (1, 0.0)

    holder = None if a.no_manage_machine else machine.acquire(38.0, note="arxiv transfer eval")
    summary = {"n": a.n, "k": k, "temperature": temp, "seed": a.seed, "arms": {}}
    try:
        with (out / "episodes.jsonl").open("w") as f:
            for name, adapter in arms:
                t0 = time.time()
                model, tokenizer = mlx_load(prof.model, adapter_path=adapter)
                if a.turns > 1:
                    from dataclasses import replace as _replace
                    from mlx_rl.tasks.base import Example, RewardResult
                    mcfg = _replace(cfg, group_size=k, temperature=temp)
                    rolls, _, _ = collect_multiturn(model, tokenizer, examples, mcfg, task)

                    class _Ep:  # minimal episode view of a Rollout
                        def __init__(s, r):
                            s.gen_count = sum(r.gen_mask) if r.gen_mask else len(r.completion_tokens)
                            s.rounds = len(r.tool_calls)
                    flat_ex = [Example(messages=[], meta=r.meta) for r in rolls]
                    flat_ep = [_Ep(r) for r in rolls]
                    flat_rec = [{"visible": r.text[-400:], "tool_calls": r.tool_calls,
                                 "finish": r.finish} for r in rolls]
                    results = [RewardResult(r.reward, dict(r.reward_parts)) for r in rolls]
                else:
                    # Chunk so rows never queue behind the completion batch:
                    # queued rows (their cloned prompt caches parked in the
                    # generator's unprocessed list across many steps) came
                    # back with corrupted KV on 2026-08-17 (empty-KV
                    # broadcast error) — training never queues, evals did.
                    per = max(1, a.batch // k)
                    groups = []
                    for lo in range(0, len(examples), per):
                        g, _, stats = _sample_episodes(model, tokenizer, examples[lo:lo + per],
                                                       cfg, task, k, temp)
                        groups.extend(g)
                        mx.clear_cache()
                    flat_ex, flat_rec, flat_ep = [], [], []
                    for ex, group in zip(examples, groups):
                        for ep in group:
                            flat_ex.append(ex)
                            flat_ep.append(ep)
                            flat_rec.append(_episode_record(tokenizer, ep, None))
                    results = task.episode_reward(flat_ex, flat_rec)
                agg: dict[str, dict] = {}
                for ex, ep, rec, res in zip(flat_ex, flat_ep, flat_rec, results):
                    keys = ["all", ex.meta["regime"]]
                    if a.turns > 1:
                        keys.append(f"turn{ex.meta.get('turn', 0)}")
                    for key in keys:
                        d = agg.setdefault(key, {"n": 0, "reward": 0.0, "gen_tokens": 0,
                                                 "rounds": 0, **{p: 0.0 for p in PARTS}})
                        d["n"] += 1
                        d["reward"] += res.total
                        d["gen_tokens"] += ep.gen_count
                        d["rounds"] += ep.rounds
                        for p in PARTS:
                            d[p] += res.parts.get(p, 0.0)
                    f.write(json.dumps({"arm": name, "meta": ex.meta, "visible": rec["visible"],
                                        "tool_calls": rec["tool_calls"], "finish": rec["finish"],
                                        "gen_tokens": ep.gen_count, "reward": res.total,
                                        "parts": res.parts}) + "\n")
                    f.flush()
                for d in agg.values():
                    n = d["n"]
                    for kk in list(d):
                        if kk != "n":
                            d[kk] = d[kk] / n
                summary["arms"][name] = {"adapter": adapter, "agg": agg,
                                         "web": dict(task.web.stats), "wall_s": round(time.time() - t0)}
                print(f"\n== {name} ({adapter or 'base'})  {time.time() - t0:.0f}s  web {task.web.stats}")
                print(f"{'regime':10s} {'n':>4s} {'reward':>7s} {'tokens':>7s} {'rounds':>6s} "
                      + " ".join(f"{p[:8]:>8s}" for p in PARTS))
                for key in sorted(agg, key=lambda x: (x != "all", x)):
                    d = agg[key]
                    print(f"{key:10s} {d['n']:4d} {d['reward']:7.2f} {d['gen_tokens']:7.0f} {d['rounds']:6.2f} "
                          + " ".join(f"{d[p]:8.2f}" for p in PARTS))
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        if holder:
            machine.release(holder)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import os
    import traceback
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    finally:  # ddgs' hung HTTP threads must not keep a model-holding process alive
        sys.stdout.flush()
        os._exit(code)
