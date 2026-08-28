#!/bin/bash
# lifecycle: one-off (archive once the trivia row is in the lab book)
# Score the two subjects the trivia run never saw -- arXiv papers and PyPI
# packages -- at every 10th checkpoint, after the fact. The run only ever
# evaluated the subject it trained on, which cannot separate learning from
# memorising; the checkpoints are all on disk, so the held-out curve can be
# recovered without retraining anything.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
say "waiting for the trainer to release the machine"
while pgrep -f "mlx_rl.train" >/dev/null; do sleep 60; done
say "retrofitting papers + packages onto the trivia run, every 10 steps, n=64"
uv run python scripts/retrofit_eval.py \
  --run runs/curve/20260822-trivia-v3 \
  --also runs/curve/20260822-trivia-v3-to120 \
  --cells papers:single,packages:single \
  --every 10 --n 64 \
  --out runs/retrofit/trivia-v3-heldout 2>&1
say "RETROFIT DONE -> runs/retrofit/trivia-v3-heldout/metrics.jsonl"
