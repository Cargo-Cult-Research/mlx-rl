#!/bin/bash
# lifecycle: one-off (archive when the papers row is in the lab book)
# Row three: train on arXiv papers, the diagonal cell that has never run under
# the corrected gradient. The 08-18 night batch had papers legs, but they were
# trained under the advantage-pruning sign bias and now live in
# runs/archive-advantage-bug-20260822/ — they are not this row.
#
# Settings identical to the trivia v3 and packages rows on purpose: lr 3e-6,
# rank 16, 12 layers, per-sign advantage pruning, eval every 10. Held-out
# cells are the other two subjects.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"papers","situation":"single","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
CELLS='trivia:single,packages:single'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

OUT=runs/curve/20260824-papers-v1
say "training papers, 60 steps -> $OUT"
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
say "PAPERS RUN DONE -> $OUT"
