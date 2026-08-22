#!/bin/bash
# lifecycle: one-off (archive when the curve question is answered)
# Attempt 3 at one readable curve — identical config to v2, fixed code.
#
# v2 (killed ~2h in by the 08-22 audit) was training on a corrupted gradient:
# 1 in 8 rewards were -3s that measured the 1024-token cap (truncation scored
# as no_reply), and the |adv| pruning kept only the negative outlier of
# majority-good groups (n_seqs 17/96 — suppression-only updates). Fixed in
# 94e2ee7: len-capped members neutralized to advantage 0, per-sign pruning
# with re-center, KL penalty exponent clamped, checkpoint-before-eval.
# NB the judge-cache fingerprint keys (same commit) make the existing local
# cache cold; the run re-judges as it goes.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
OUT=runs/curve/20260822-trivia-v3
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
say "curve v3: trivia:single, 60 steps, lr 3e-6, eval n=160 every 10 (v2 config, audit-fixed code)"
rc=1
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty \
  --task-kwargs '{"domain":"trivia","situation":"single","calib_file":"runs/qa-calib-20260724/calib.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}' \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --checkpoint-every 5 --seed 0 \
  --lease-wait 900 --required-gb 62 --out "$OUT" 2>&1 && rc=0
if [ $rc -eq 0 ]; then
  uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
  say "CURVE V3 DONE"
else
  say "CURVE V3 FAILED (train exited non-zero — see above)"
  exit 1
fi
