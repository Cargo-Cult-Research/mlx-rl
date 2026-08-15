#!/usr/bin/env python3
# lifecycle: one-off (archive when the qwen36-vs-qwen38 duel is published)
"""Join two difficulty-sweep JSONLs by task_id and Telegram one message per task.

Watches both legs' output files and sends a message for a task as soon as BOTH
models have reported it — so when the legs run back to back, every message
arrives live during the second leg and already contains the comparison.

Why a separate watcher rather than a hook inside difficulty_sweep.py: the sweep
is a long unattended GPU job and its inner loop should not grow a network call
that can hang or throw. This process holds no model, can be killed and
restarted freely, and is fully testable against finished runs (--dry-run).

Sent task_ids are journalled to <state>, so a restart never re-sends. Delivery
failures are NOT journalled and are retried on the next poll.

Run (live, alongside the second leg):
    .venv/bin/python scripts/duel_report.py \
        --a runs/sweeps/deepcoder-32k-qwen36.jsonl --a-name qwen36 \
        --b runs/sweeps/deepcoder-32k-qwen38.jsonl --b-name qwen38

Rehearse against the existing pilot legs without sending anything:
    .venv/bin/python scripts/duel_report.py --dry-run --once \
        --a runs/sweeps/deepcoder-pilot-qwen36.jsonl \
        --b runs/sweeps/deepcoder-pilot-qwen3-4b.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ENV_PATH = Path.home() / "code/housekeeping/.env"


def read_env() -> dict:
    return dict(
        line.strip().split("=", 1)
        for line in ENV_PATH.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )


def send(text: str, env: dict) -> bool:
    """Best-effort Telegram. Returns True only on a confirmed 200/ok."""
    r = subprocess.run(
        ["curl", "-s", "-m", "20",
         f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendMessage",
         "-d", f"chat_id={env['TELEGRAM_USER_ID']}",
         "-d", "parse_mode=HTML",
         "--data-urlencode", f"text={text}"],
        capture_output=True, text=True)
    try:
        return json.loads(r.stdout).get("ok", False)
    except (json.JSONDecodeError, AttributeError):
        return False


def load_rows(path: Path) -> dict:
    """task_id -> row. Last row wins if a task was swept more than once."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail line: the sweep is mid-write, pick it up next poll
        if "task_id" in row:
            out[row["task_id"]] = row
    return out


def per_problem_s(row: dict) -> float | None:
    """Seconds this problem itself spent decoding, under batching.

    Sequences in a batch step in lockstep and the batch ends when the longest
    finishes, so seconds-per-step = batch_wall_s / batch_max_len, and this
    problem occupied (its own longest sample) steps. Exact when batch_size=1.
    None for rows written before the timing fields existed.
    """
    wall, bmax = row.get("batch_wall_s"), row.get("batch_max_len")
    if not wall or not bmax:
        return None
    return max(row["lens"]) * (wall / bmax)


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "n/a"
    return f"{s:.0f}s" if s < 90 else f"{s/60:.1f}m"


def verdict(a: dict, b: dict) -> tuple[str, str]:
    """(emoji, label) for the pair — the scannable part of the message."""
    pa, pb = a["n_pass"] > 0, b["n_pass"] > 0
    if pa and pb:
        return "✅", "both solved"
    if pb and not pa:
        return "🆕", "B ONLY"
    if pa and not pb:
        return "🔻", "A only"
    return "❌", "neither"


def leg_line(name: str, row: dict, cap: int) -> str:
    toks = sum(row["lens"])
    trunc = sum(1 for f in row["finishes"] if f == "length")
    mark = f"  ⚠️{trunc}/{len(row['lens'])} at cap" if trunc else ""
    return (f"<b>{name}</b>  {row['n_pass']}/{row['k']} pass  "
            f"{toks:,} tok  {fmt_dur(per_problem_s(row))}{mark}")


def build_message(tid, a, b, a_name, b_name) -> str:
    emoji, label = verdict(a, b)
    cap = a.get("cap")
    head = f"{emoji} <b>{a.get('task','?')} {tid}</b> — {label}"
    body = [leg_line(a_name, a, cap), leg_line(b_name, b, cap)]
    src = a.get("source") or b.get("source")
    foot = f"<i>cap {cap:,} · k={a['k']} · T={a['temperature']}</i>"
    if src:
        foot = f"<i>{src} · </i>" + foot
    return "\n".join([head, *body, foot])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="leg A jsonl (baseline)")
    ap.add_argument("--b", required=True, help="leg B jsonl (challenger)")
    ap.add_argument("--a-name", default="qwen36")
    ap.add_argument("--b-name", default="qwen38")
    ap.add_argument("--state", default=None,
                    help="sent-ids journal (default: <b>.reported)")
    ap.add_argument("--poll-s", type=float, default=60)
    ap.add_argument("--only-disagreements", action="store_true",
                    help="message only when exactly one model solved it "
                         "(the interesting signal); others still counted")
    ap.add_argument("--summary-every", type=int, default=25,
                    help="also send a running tally every N pairs (0=off)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print messages instead of sending; state untouched")
    ap.add_argument("--once", action="store_true",
                    help="one pass then exit (default: poll until both legs "
                         "are complete and every pair is sent)")
    ap.add_argument("--expect", type=int, default=0,
                    help="total tasks expected; with --once-complete, exit "
                         "when this many pairs are sent")
    args = ap.parse_args()

    a_path, b_path = Path(args.a), Path(args.b)
    state = Path(args.state) if args.state else b_path.with_suffix(".reported")
    sent = set()
    if state.exists() and not args.dry_run:
        sent = {json.loads(l)["task_id"] for l in state.read_text().splitlines() if l.strip()}
    env = {} if args.dry_run else read_env()
    tally = {"both": 0, "b_only": 0, "a_only": 0, "neither": 0}

    while True:
        A, B = load_rows(a_path), load_rows(b_path)
        pairs = sorted(set(A) & set(B), key=str)
        new = [t for t in pairs if t not in sent]
        for tid in new:
            a, b = A[tid], B[tid]
            _, label = verdict(a, b)
            tally[{"both solved": "both", "B ONLY": "b_only",
                   "A only": "a_only", "neither": "neither"}[label]] += 1
            skip = args.only_disagreements and label in ("both solved", "neither")
            if not skip:
                msg = build_message(tid, a, b, args.a_name, args.b_name)
                if args.dry_run:
                    print(msg + "\n" + "-" * 50)
                elif not send(msg, env):
                    # Not journalled -> retried next poll. Do not mark sent.
                    print(f"send failed for {tid}, will retry", flush=True)
                    continue
            if not args.dry_run:
                with state.open("a") as fh:
                    fh.write(json.dumps({"task_id": tid, "label": label}) + "\n")
            sent.add(tid)

        if new and args.summary_every and len(sent) % args.summary_every < len(new):
            s = (f"📊 duel {len(sent)}"
                 + (f"/{args.expect}" if args.expect else "")
                 + f" pairs — {args.b_name} only: {tally['b_only']}, "
                 f"{args.a_name} only: {tally['a_only']}, "
                 f"both: {tally['both']}, neither: {tally['neither']}")
            print(s, flush=True) if args.dry_run else send(s, env)

        if args.once:
            break
        if args.expect and len(sent) >= args.expect:
            print("all expected pairs reported", flush=True)
            break
        time.sleep(args.poll_s)

    print(f"pairs reported: {len(sent)}  tally: {tally}", flush=True)


if __name__ == "__main__":
    main()
