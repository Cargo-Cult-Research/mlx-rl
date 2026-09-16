"""Measure what the BASE model already knows about each snapshot paper.

qa_arxiv's "known" regime must be measured, not assumed from a famous flag
(design doc §D4: the knowledge cutoff is fuzzy, so use per-question base
pass rate). For every real paper in the snapshot, ask the first-author
question k times at temperature 1 with NO tools and NO system prompt, and
record the fraction of replies that contain the first author's surname.

Output: jsonl {"id", "title", "pass_rate", "k"} — pass it to the task as
task_kwargs calib_file; known = pass_rate >= 0.8.

    uv run python scripts/arxiv_calibrate.py --out runs/arxiv-calib-<date>/calib.jsonl

Writes to runs/<probe>/; copying the result into data/labels/ is the
promotion step (the papers domain reads data/labels/papers-pass@K-qwen36.jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load as mlx_load

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_rl import machine  # noqa: E402
from mlx_rl.engine import rollout_groups  # noqa: E402
from mlx_rl.profiles import get_profile  # noqa: E402
from mlx_rl.rollout import encode_prompt  # noqa: E402
from mlx_rl.tasks.qa_arxiv import _surname, author_or_year_match  # noqa: E402
from mlx_rl.train import _completion_text, _visible_reply  # noqa: E402

FRAME = 'Who is the first author of the paper "{t}"? Reply with just the name.'


def _contains(reply: str, aliases: list[str]) -> bool:
    return author_or_year_match(reply, aliases)  # same rule as the reward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="qwen36")
    ap.add_argument("--snapshot", default="data/arxiv_snapshot.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-items", type=int, default=32)
    ap.add_argument("--out", default=f"runs/arxiv-calib-{time.strftime('%Y%m%d')}/calib.jsonl")
    ap.add_argument("--no-manage-machine", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.snapshot).read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("fictional") and r.get("authors")]
    prof = get_profile(a.profile)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with machine.lease(38.0, "arxiv calibration probe", manage=not a.no_manage_machine):
        model, tokenizer = mlx_load(prof.model)
        think_close = prof.think_close(tokenizer)
        t0 = time.time()
        n_known = 0
        with out.open("w") as f:
            for lo in range(0, len(rows), a.batch_items):
                chunk = rows[lo:lo + a.batch_items]
                prompts = [encode_prompt(
                    tokenizer, [{"role": "user", "content": FRAME.format(t=r["title"])}],
                    **prof.chat_kwargs) for r in chunk]
                groups, _ = rollout_groups(model, tokenizer, prompts, a.k,
                                           a.max_new_tokens, 1.0,
                                           extra_eos=tuple(prof.extra_eos))
                for r, group in zip(chunk, groups):
                    first = r["authors"][0]
                    aliases = [first, _surname(first)]
                    hits = 0
                    replies = []
                    for comp in group:
                        vis, _ = _visible_reply(_completion_text(tokenizer, comp), think_close)
                        replies.append(vis[:120])
                        hits += int(_contains(vis, aliases))
                    rate = hits / len(group)
                    n_known += int(rate >= 0.8)
                    f.write(json.dumps({"id": r["id"], "title": r["title"],
                                        "pass_rate": rate, "k": len(group),
                                        "famous": bool(r.get("famous")),
                                        "replies": replies}) + "\n")
                f.flush()
                print(f"  {lo + len(chunk)}/{len(rows)}  known so far {n_known}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out}: {n_known} known of {len(rows)}")


if __name__ == "__main__":
    main()
