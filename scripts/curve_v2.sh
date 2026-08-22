#!/bin/bash
# Attempt 2 at one readable curve.
#
# What the lr 1e-5 run settled: gradient norm averaged 4.73 (max 71) against a
# clip of 1.0, so every step was clipped -- magnitude discarded, direction kept
# -- KL rose 100x to 0.22, and training reward went BACKWARDS over 45 steps.
# lr returns to 3e-6, where norms sat at 0.4-0.6 and nothing clipped.
#
# The other thing that run showed: with 32 eval items the eval curve has a
# standard error near 0.27 against an effect of maybe 0.3, so it cannot resolve
# the answer no matter how clean training is. Eval goes to 160 items, less
# often, which is the trade that actually buys signal.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
OUT=runs/curve/20260822-trivia-v2
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
say "curve v2: trivia:single, 60 steps, lr 3e-6, eval n=160 every 10"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty \
  --task-kwargs '{"domain":"trivia","situation":"single","calib_file":"runs/qa-calib-20260724/calib.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}' \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --checkpoint-every 5 --seed 0 \
  --lease-wait 900 --required-gb 62 --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "CURVE V2 DONE"
