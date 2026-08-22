#!/bin/bash
# ONE measurement, done properly: three subjects, one situation.
#
#   train:  trivia:single, papers:single, packages:single   (one adapter each)
#   eval:   every adapter on all three, plus base
#
# Leave-one-out is then just reading a column down the rows that were not
# trained on it. No stacking, no merging, no composition assumptions.
#
# Swamping is parked; tool-failure is deferred. Neither appears here, which is
# also why no run in this script ever calls the claims judge -- the single
# situation does not use it. Only the commitment judge, which is the one the
# local model handles well.
#
# The packages leg trains under Opus like its two siblings: rows of this table
# have to be comparable, and that is worth one leg's tokens.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
R=runs/night/20260818-night
OUT=runs/day/20260821-single
mkdir -p "$OUT"
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

LEG="$R/packages-single"
if [ ! -f "$LEG/promoted/adapters.safetensors" ]; then
  say "leg packages:single (the last one missing)"
  uv run python -m mlx_rl.train \
    --profile qwen36 --task honesty \
    --task-kwargs '{"domain":"packages","situation":"single"}' \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 60 --batch-prompts 4 --group-size 8 --group-stage1 4 --stage1-skip saturated \
    --update-adv-frac 0.25 --micro-batch 1 --grad-checkpoint \
    --max-tool-rounds 4 --max-episode-tokens 8192 --max-new-tokens 1536 \
    --rollout-batch-size 32 \
    --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
    --eval-every 10 --eval-n 16 --checkpoint-every 5 --seed 0 \
    --lease-wait 900 --required-gb 62 --out "$LEG" 2>&1
  uv run python scripts/promote_adapter.py "$LEG" --out "$LEG/promoted" 2>&1 || say "nothing to promote"
fi

CELLS=trivia:single,papers:single,packages:single
arm () {
  local name=$1 adapter=$2
  [ -f "$OUT/$name/results.json" ] && { say "skip $name"; return 0; }
  say "eval $name"
  uv run python scripts/matrix_eval.py --cells "$CELLS" --arm "$name=$adapter" \
    --n 32 --k 2 --no-manage-machine --out "$OUT/$name" 2>&1
}
arm base ""
for a in trivia-single papers-single packages-single papers-single-seed1; do
  [ -f "$R/$a/promoted/adapters.safetensors" ] && arm "$a" "$PWD/$R/$a/promoted"
done
say "SINGLE ROW DONE"
