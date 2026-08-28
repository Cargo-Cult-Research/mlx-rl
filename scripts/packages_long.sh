#!/bin/bash
# lifecycle: one-off (archive when the packages row is in the lab book)
# Row two: train on PyPI packages, scoring trivia and arXiv papers at every
# eval so overfitting shows up while it happens instead of in a retrofit.
# Reward is unchanged from the trivia run on purpose -- the per-search cost
# and randomised search budget are deferred so the two rows stay comparable.
#
# --required-gb stays at 62 despite preflight's ~68 GB estimate: the same
# warning fired on both trivia runs, whose real peak was 79 GB of 96 with the
# swap guard never tripping. The estimate is the conservative x2.9 rule, and
# asking the lease to clear more would only make it wait longer for no gain.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"packages","situation":"single","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
CELLS='trivia:single,papers:single'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "waiting for the smoke test to finish"
while pgrep -f "mlx_rl.train.*runs/smoke/packages" >/dev/null; do sleep 20; done
SMOKE=$(ls -dt runs/smoke/packages-* 2>/dev/null | head -1)
if ! grep -q "eval_trivia_single_reward" "$SMOKE/metrics.jsonl" 2>/dev/null; then
  say "smoke test did not record the extra subjects — NOT starting the long run"; exit 1
fi
say "smoke test passed: extra subjects recorded in $SMOKE"

OUT=runs/curve/$(date +%Y%m%d)-packages-v1
say "training packages, 60 steps, ~17 min/step -> $OUT"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --eval-cells "$CELLS" --eval-cells-n 64 \
  --checkpoint-every 5 --seed 0 --lease-wait 1800 --required-gb 62 \
  --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "PACKAGES RUN DONE -> $OUT"
