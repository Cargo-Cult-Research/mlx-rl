#!/bin/bash
# lifecycle: one-off (archive when a readable curve is settled — superseded by curve_v2.sh)
# One clean curve. The old legs computed gradients from ~3 sequences per step,
# giving the plotted reward a standard error of 0.4-0.7 against an effect of
# maybe 0.5 -- noise by construction. Calibrated at these settings: 17-30
# sequences per step, 7-8 active groups of 12, gradient norm 0.4-0.6 (so no
# clipping), advantage spread 1.3-1.6.
#
# Batch is 5-8x the old legs AND lr goes 3e-6 -> 1e-5. Two changes at once,
# deliberately: gradient norms of 0.4-0.6 against a clip of 1.0 and a KL of
# 0.002 say the old lr was leaving the policy nearly still, and there is not
# time to test them separately. If the curve misbehaves, lr comes back down
# first -- that is the cheaper thing to revert. The commitment judge runs locally (valid here because
# situation=single never calls the claims judge) which is what makes a 12-prompt
# batch affordable at all.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
OUT=runs/curve/20260821-trivia-lr1e5
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
say "clean curve: trivia:single, 45 steps, 12 prompts x 8, lr 1e-5"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty \
  --task-kwargs '{"domain":"trivia","situation":"single","calib_file":"runs/qa-calib-20260724/calib.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}' \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 45 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 1e-5 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 5 --eval-n 32 --checkpoint-every 5 --seed 0 \
  --lease-wait 900 --required-gb 62 --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "CLEAN CURVE DONE"
