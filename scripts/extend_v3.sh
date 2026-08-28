#!/bin/bash
# lifecycle: one-off (archive when the trivia curve question is settled)
# Extend the v3 curve from 60 steps to 120, queued NOW so the window is not
# lost waiting for a human to notice 60 finished.
#
# --steps is a TOTAL (the loop runs range(start_step, cfg.steps+1)), so 120
# continues the same curve rather than starting a second one. --resume-from
# writes a NEW directory and leaves the source run's record intact, and it
# exempts `steps` from its config-drift refusal -- extending is a supported
# resume, not a hack, so the curve stays comparable end to end.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
SRC=runs/curve/20260822-trivia-v3
DST=runs/curve/20260822-trivia-v3-to120
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "waiting for the 60-step run to finish"
while pgrep -f "mlx_rl.train.*20260822-trivia-v3" >/dev/null; do sleep 120; done
if [ ! -d "$SRC/resume" ]; then say "no resume state in $SRC — nothing to extend"; exit 1; fi
say "extending $SRC -> $DST (steps 60 -> 120, ~17 min/step, ~17h)"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty \
  --task-kwargs '{"domain":"trivia","situation":"single","calib_file":"runs/qa-calib-20260724/calib.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}' \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 120 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --checkpoint-every 5 --seed 0 \
  --resume-from "$SRC" --lease-wait 1800 --required-gb 62 --out "$DST" 2>&1
uv run python scripts/promote_adapter.py "$DST" --out "$DST/promoted" 2>&1 || say "nothing to promote"
say "EXTENSION TO 120 DONE"
