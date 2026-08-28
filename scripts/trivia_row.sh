#!/bin/bash
# lifecycle: one-off (archive once the trivia row is in the lab book)
# First row of the leave-one-out matrix, measured on a correct gradient.
#
# The trivia adapter (v3, stopped at step 90 once held-out reward went flat)
# against ALL THREE subjects: the one it trained on plus the two it has never
# seen. base is measured in the same pass on the same seeded items, so the
# comparison is within-protocol -- earlier arms were scored under the
# sign-biased gradient and are NOT comparable to this row.
#
# Judge: local, matching the training loop. It agrees with Opus 0.87-0.98 on
# kind but only 0.65 on denial, so absolute levels here are NOT quotable
# against Opus-judged numbers -- but both arms are read with the same ruler,
# so the base-vs-adapter difference, which is what the row is for, holds.
# A small Opus spot-check on a subsample can calibrate the offset later for a
# fraction of the tokens a full Opus pass would cost.
#
# n=128 items x k=2 episodes: reward sd is ~1.5-1.9, so item-clustered SE
# lands near 0.14 -- under the effect sizes seen on the trained subject
# (0.2-0.6) rather than swamping them, which n=32 did.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
ADAPTER=runs/curve/20260822-trivia-v3-to120/promoted
OUT=runs/matrix/$(date +%Y%m%d-%H%M)-trivia-row
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "waiting for the trainer to release the machine"
while pgrep -f "mlx_rl.train" >/dev/null; do sleep 60; done
if [ ! -d "$ADAPTER" ]; then say "no promoted adapter at $ADAPTER -- aborting"; exit 1; fi
say "evaluating base + trivia-v3 on trivia, papers, packages -> $OUT"
uv run python scripts/matrix_eval.py \
  --profile qwen36 \
  --arm base= \
  --arm trivia-v3="$ADAPTER" \
  --cells trivia:single@judge_backend=local,papers:single@judge_backend=local,packages:single@judge_backend=local \
  --n 128 --k 2 --seed 2026 \
  --out "$OUT" 2>&1
say "TRIVIA ROW DONE -> $OUT/results.json"
