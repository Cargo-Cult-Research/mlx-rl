"""Stitch a run that spans several directories (--resume-from) into one view.

A resumed run writes a NEW directory rather than appending, so the source
run's record is never overwritten. The cost is that the full history lives in
N directories; this reassembles it.

Segments are given oldest-first. Steps are merged into a single timeline; when
segments overlap (the source ran past the checkpoint it was resumed from),
the LATER segment wins, because those are the steps that actually led to the
final weights. Overlaps are reported, not silently dropped.

    .venv/bin/python scripts/assemble_run.py runs/foo runs/foo-r1 runs/foo-r2 \
        --out runs/foo-assembled

Writes metrics.jsonl (step-ordered), samples.jsonl, and provenance.json.
Adapters are left where they are — provenance.json records which segment owns
each checkpoint step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("segments", nargs="+", help="run dirs, OLDEST first")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    segs = [Path(s) for s in a.segments]
    for s in segs:
        if not (s / "metrics.jsonl").exists():
            raise SystemExit(f"{s} has no metrics.jsonl")

    timeline: dict[int, dict] = {}   # step -> row (later segment wins)
    owner: dict[int, str] = {}
    provenance, samples = [], []
    for seg in segs:
        rows = _rows(seg / "metrics.jsonl")
        resumed = next((r for r in rows if "resumed_from" in r), None)
        steps = [r["step"] for r in rows if "reward_mean" in r]
        overlap = sorted(set(steps) & set(timeline))
        for r in rows:
            if "resumed_from" in r:
                continue
            step = r.get("step")
            if step is None:
                continue
            timeline[step] = {**timeline.get(step, {}), **r} if r.get("final") \
                else r
            owner[step] = seg.name
        samples.extend(_rows(seg / "samples.jsonl"))
        provenance.append({
            "segment": str(seg),
            "resumed_from": resumed["resumed_from"] if resumed else None,
            "resumed_at_step": resumed["step"] if resumed else None,
            "steps": [min(steps), max(steps)] if steps else [],
            "overlap_superseded": overlap,
        })
        if overlap:
            print(f"{seg.name}: supersedes {len(overlap)} earlier step(s) "
                  f"{overlap[0]}-{overlap[-1]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.jsonl").open("w") as f:
        for step in sorted(timeline):
            f.write(json.dumps(timeline[step]) + "\n")
    with (out / "samples.jsonl").open("w") as f:
        for row in sorted(samples, key=lambda r: r.get("step", 0)):
            f.write(json.dumps(row) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "segments": provenance,
        "checkpoint_owner": {str(k): v for k, v in sorted(owner.items())},
    }, indent=2) + "\n")

    trained = [s for s in timeline if "reward_mean" in timeline[s]]
    print(f"assembled {len(segs)} segments -> {out}: "
          f"{len(trained)} training steps, "
          f"{min(trained, default=0)}-{max(trained, default=0)}")
    gaps = [s for s in range(min(trained, default=1), max(trained, default=0))
            if s not in timeline]
    if gaps:
        print(f"WARNING: {len(gaps)} missing step(s), e.g. {gaps[:10]}")


if __name__ == "__main__":
    main()
