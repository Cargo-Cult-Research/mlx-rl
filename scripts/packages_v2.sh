#!/bin/bash
# lifecycle: one-off (archive when the packages row is in the lab book)
# Packages row, second attempt. v1 (2026-08-23) was graded by a reward that
# read every backticked identifier as a claim that a PyPI package exists, so
# 17% of answers were fined -3 for naming a class in a code sample. Fixed
# 2026-08-24: the penalty now rides `pip install` lines only, plus a
# dictionary guard so an English word that resolves nowhere is read as prose.
# Regrading v1's 480 stored answers: all 11 real hallucinations survive, all
# 80 false accusations drop, mean reward -0.158 -> +0.508.
#
# Every knob is byte-identical to v1 INCLUDING --seed 0, so the question order
# repeats and the two runs differ by the grader and nothing else. That is the
# whole point; do not "improve" a setting here.
#
# --required-gb stays 62 against preflight's ~68 GB estimate: same warning
# fired on both trivia runs and on v1, whose real peak was 62.8 GB of 96 with
# the swap guard never tripping.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"packages","situation":"single","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
CELLS='trivia:single,papers:single'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

OUT=runs/curve/20260824-packages-v2
say "training packages under the fixed grader, 60 steps, ~15 min/step -> $OUT"
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
say "PACKAGES V2 DONE -> $OUT"
