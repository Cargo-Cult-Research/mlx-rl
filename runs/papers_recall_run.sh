#!/bin/bash
# papers_recall_run.sh — 2026-07-29: recall-frame follow-up to the papers
# probe. Re-asks the same arxiv papers as authors/year recall questions
# (scripts/papers_recall_probe.py docstring has the hypothesis), base vs the
# promoted qa-abstain-20260726 adapter, then diffs hedge+denial per bucket.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
ADAPTER="$HOME/models/adapters/qa-abstain-20260726"
LOG=runs/papers-recall-20260729.log
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

say "BASE recall probe start (no adapter)"
$PY scripts/papers_recall_probe.py --k 4 \
    --out runs/papers-recall-base-20260729 >>"$LOG" 2>&1 || say "base probe nonzero"

say "RL recall probe start (adapter=$ADAPTER)"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$ADAPTER" \
    --out runs/papers-recall-rl-20260729 >>"$LOG" 2>&1 || say "rl probe nonzero"

say "=== RECALL-FRAME DIFF (base vs RL) ==="
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, os
def load(d):
    p=os.path.join(d,"summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}
B=load("runs/papers-recall-base-20260729"); R=load("runs/papers-recall-rl-20260729")
print(f"{'bucket':22s} {'set':4s} {'n':>4s} {'hedge+denial':>13s} {'conf-wrong':>11s} {'correct':>8s}")
for bk in ("recall-post-authors","recall-post-year","recall-famous-authors","recall-famous-year"):
    for tag,S in (("BASE",B),("RL",R)):
        r=S.get(bk)
        if not r: print(f"{bk:22s} {tag:4s}  (no data)"); continue
        print(f"{bk:22s} {tag:4s} {r['n_replies']:4d} {r['hedge']+r['denial']:13.2f} "
              f"{r['confident_wrong']:11.2f} {r['correct']:8.2f}")
print("\nsummarize-frame reference (qa-chat 20260729, papers-post):")
print("  BASE hedge+denial=0.50  RL hedge+denial=0.51  (delta +0.01)")
PYEOF
say "RECALL PROBE DONE — replies in runs/papers-recall-{base,rl}-20260729/"
