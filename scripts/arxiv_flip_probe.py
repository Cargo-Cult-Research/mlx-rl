"""The falsification test for qa_arxiv (design doc §7): same paper, two stated
dates straddling its publication date. Under the later date the search finds
it and the right move is answer-from-result; under the earlier date the index
hides it and the right move is search -> "can't find it". If the policy's
behaviour does NOT flip, it learned a year, not the comparison — whatever the
aggregate buckets say.

Also runs the same pairs on the fictional slice (never findable) and on the
known slice (should answer without needing the tool). Reports, per arm and
condition: called / found / answered / correct / abstain / denial / reward.

    uv run python scripts/arxiv_flip_probe.py --adapter runs/qa-arxiv-arm1-<date>/adapters \
        --calib runs/arxiv-calib-<date>/calib.jsonl --n 40 --k 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_rl import machine  # noqa: E402
from mlx_lm import load as mlx_load  # noqa: E402

from mlx_rl.config import TrainConfig  # noqa: E402
from mlx_rl.profiles import get_profile  # noqa: E402
from mlx_rl.tasks.qa_arxiv import FRAMES, QAArxivTask  # noqa: E402
from mlx_rl.train import _episode_record, _sample_episodes  # noqa: E402


def _pair(task, row, qtype, frame, days):
    """Two examples for one paper: today = published +/- days, else identical."""
    from mlx_rl.tasks.base import Example
    from mlx_rl.tasks.qa_arxiv import DATE_LINE, _surname
    out = []
    pub = date.fromisoformat(row["published"])
    content = frame.format(t=row["title"], y=row["published"][:4])
    first = row["authors"][0] if row.get("authors") else ""
    aliases = ([first, _surname(first)] if qtype == "authors"
               else [row["published"][:4]])
    for cond, delta in (("post", days), ("future", -days)):
        today = (pub + timedelta(days=delta)).isoformat()
        sys_parts = [task.system_text] if task.system_text else []
        if task.tool_first:
            sys_parts.append(task.tool_first)
        sys_parts.append(DATE_LINE.format(today=today))
        out.append(Example(
            messages=[{"role": "system", "content": " ".join(sys_parts)},
                      {"role": "user", "content": content}],
            meta={"id": row["id"], "title": row["title"], "qtype": qtype,
                  "aliases": aliases, "published": row["published"],
                  "today": today, "regime": cond, "question": content,
                  "asserted_year": row["published"][:4], "fictional": False},
            chat_kwargs={"tools": task.tools}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--adapter", default=None, help="adapter dir (None = base)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snapshot", default="data/arxiv_snapshot.jsonl")
    ap.add_argument("--n", type=int, default=40, help="papers per slice")
    ap.add_argument("--k", type=int, default=3, help="samples per example")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=f"runs/arxiv-flip-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    prof = get_profile(a.profile)
    task = QAArxivTask(snapshot=a.snapshot, calib_file=a.calib)
    rng = random.Random(7)
    unknown = [r for r in task._pools["eval"]["unknown"] if r.get("authors")]
    rng.shuffle(unknown)
    examples = []
    for row in unknown[: a.n]:
        qtype = rng.choice(list(FRAMES))
        frame = rng.choice(FRAMES[qtype])
        examples += _pair(task, row, qtype, frame, a.days)
    tags = ["flip"] * len(examples)
    # known and fictional slices through the ordinary sampler
    for slice_name, regime in (("known", "known"), ("fictional", "fictional")):
        got = 0
        r2 = random.Random(11)
        while got < a.n:
            ex = task._example(r2, "eval")
            if ex.meta["regime"] == regime:
                examples.append(ex)
                tags.append(slice_name)
                got += 1
    cfg = TrainConfig(model=prof.model, task="qa_arxiv", profile=a.profile,
                      chat_kwargs=dict(prof.chat_kwargs),
                      max_new_tokens=a.max_new_tokens, max_tool_rounds=2,
                      think_end=prof.think_end, extra_eos=tuple(prof.extra_eos))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    holder = None if a.no_manage_machine else machine.acquire(38.0, note="arxiv flip probe")
    try:
        model, tokenizer = mlx_load(prof.model, adapter_path=a.adapter)
        groups, _, _ = _sample_episodes(model, tokenizer, examples, cfg, task, a.k, 1.0)
    finally:
        if holder:
            machine.release(holder)
    flat_ex, flat_rec, flat_tag = [], [], []
    for ex, tag, group in zip(examples, tags, groups):
        for ep in group:
            flat_ex.append(ex)
            flat_rec.append(_episode_record(tokenizer, ep, None))
            flat_tag.append(tag)
    results = task.episode_reward(flat_ex, flat_rec)
    agg: dict[str, dict] = {}
    with (out / "episodes.jsonl").open("w") as f:
        for ex, rec, tag, res in zip(flat_ex, flat_rec, flat_tag, results):
            key = f"{tag}:{ex.meta['regime']}"
            d = agg.setdefault(key, {"n": 0})
            d["n"] += 1
            for k in ("called", "found_target", "answered", "correct", "abstain",
                      "denial", "checked_absent", "no_reply"):
                d[k] = d.get(k, 0.0) + res.parts.get(k, 0.0)
            d["reward"] = d.get("reward", 0.0) + res.total
            f.write(json.dumps({"tag": tag, "meta": ex.meta, "visible": rec["visible"],
                                "tool_calls": rec["tool_calls"], "finish": rec["finish"],
                                "reward": res.total, "parts": res.parts}) + "\n")
    print(f"\n{'condition':18s} {'n':>4s} " + " ".join(f"{k:>8s}" for k in
          ("called", "found", "answer", "correct", "abstain", "denial", "checked", "reward")))
    for key in sorted(agg):
        d = agg[key]
        n = d["n"]
        print(f"{key:18s} {n:4d} " + " ".join(f"{d[k]/n:8.2f}" for k in
              ("called", "found_target", "answered", "correct", "abstain",
               "denial", "checked_absent", "reward")))
    (out / "summary.json").write_text(json.dumps(
        {"adapter": a.adapter, "n": a.n, "k": a.k, "days": a.days,
         "agg": {k: {kk: (v / d["n"] if kk != "n" else v) for kk, v in d.items()}
                 for k, d in agg.items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
