#!/bin/bash
# lifecycle: one-off (archive when the packages row is in the lab book)
# Row two of the leave-one-out matrix: train on PyPI packages, the subject we
# have never trained on, and score trivia and arXiv papers alongside it at
# every eval so overfitting is visible while it happens rather than in a
# retrofit afterwards.
#
# Reward is UNCHANGED from the trivia run on purpose. The per-search cost and
# the randomised search budget are both deferred: changing the reward now
# would make row two incomparable to row one, on top of costing a rerun.
#
# A one-step smoke test runs first. --eval-cells has never executed inside a
# real training loop, and this run's own history is that first-eval bugs kill
# runs three hours in, after the window is gone and before any checkpoint.
# Fifteen minutes here is cheap against that.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"packages","situation":"single","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
CELLS='trivia:single,papers:single'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "smoke test: 1 step, tiny evals, all three subjects"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 1 --batch-prompts 2 --group-size 4 --group-stage1 2 \
  --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 16 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 1 --eval-n 8 --eval-cells "$CELLS" --eval-cells-n 8 \
  --checkpoint-every 1 --seed 0 --lease-wait 1800 --required-gb 62 \
  --out runs/smoke/packages-$(date +%H%M) 2>&1
if [ $? -ne 0 ]; then say "SMOKE TEST FAILED — not starting the long run"; exit 1; fi
say "smoke test passed"

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
