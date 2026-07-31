#!/bin/bash
# papers_recall_informat_run.sh — 2026-07-29: frame-vs-signal control. Same
# paper recall questions, but wrapped in the qa_abstain training PROMPT
# (<answer>/<abstain/>). If the adapter abstains here but not in chat, the
# transfer failure is frame-bound; if it answers confidently even here, the
# uncertainty signal never fires for paper-title entities.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
ADAPTER="$HOME/models/adapters/qa-abstain-20260726"
LOG=runs/papers-recall-20260729.log
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

say "IN-FORMAT base probe start (training PROMPT, no adapter)"
$PY scripts/papers_recall_probe.py --k 4 --in-format --max-new-tokens 96 \
    --out runs/papers-recall-fmt-base-20260729 >>"$LOG" 2>&1 || say "fmt base nonzero"

say "IN-FORMAT RL probe start (training PROMPT, adapter=$ADAPTER)"
$PY scripts/papers_recall_probe.py --k 4 --in-format --max-new-tokens 96 \
    --adapter "$ADAPTER" \
    --out runs/papers-recall-fmt-rl-20260729 >>"$LOG" 2>&1 || say "fmt rl nonzero"

say "=== IN-FORMAT DIFF (base vs RL) ==="
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, os
def load(d):
    p=os.path.join(d,"summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}
B=load("runs/papers-recall-fmt-base-20260729"); R=load("runs/papers-recall-fmt-rl-20260729")
print(f"{'bucket':22s} {'set':4s} {'n':>4s} {'abstain':>8s} {'answered':>9s} {'malformed':>10s} {'correct':>8s}")
for bk in ("recall-post-authors","recall-post-year","recall-famous-authors","recall-famous-year"):
    for tag,S in (("BASE",B),("RL",R)):
        r=S.get(bk)
        if not r: print(f"{bk:22s} {tag:4s}  (no data)"); continue
        answered=1.0-r["abstain"]-r["malformed"]
        print(f"{bk:22s} {tag:4s} {r['n_replies']:4d} {r['abstain']:8.2f} "
              f"{answered:9.2f} {r['malformed']:10.2f} {r['correct']:8.2f}")
print("\nchat-frame reference (same questions, no tag affordance):")
print("  recall-post hedge+denial: BASE 0.03  RL 0.01")
PYEOF
say "IN-FORMAT RECALL PROBE DONE — replies in runs/papers-recall-fmt-{base,rl}-20260729/"
