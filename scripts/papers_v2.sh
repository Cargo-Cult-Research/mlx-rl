#!/bin/bash
# lifecycle: one-off (archive when the papers row is in the lab book)
# Papers row, second attempt — on the snapshot backend this time.
#
# Why v1 is dead (measured 2026-08-25/26, not guessed):
#  * PapersDomain hardcoded backend="web", so v1 searched live through
#    ddgs-scraped bing/yahoo. 94% of searches for a REAL paper never surfaced
#    it, and a real paper and a fabricated one were indistinguishable through
#    the tool (relevant 5% vs 11%) -- the exact distinction this task teaches.
#    The unknown band collapsed 0.532 -> 0.224 not because the policy got
#    worse (uncapped episodes stayed 1.00 correct throughout) but because it
#    correctly learned that no distinction was observable.
#  * v1 passed no calib_file, so its in-domain eval ran uncalibrated (3 bands)
#    while build_eval_cells injected one for the held-out cells (4 bands).
#    Base scored +0.250 vs +0.762 on "the same" subject. Fixed here by
#    passing calib_file explicitly, so diagonal and off-diagonal agree.
#
# On snapshot, measured over 200 sampled items: post 100% found, known 100%,
# future 0%, fictional 0%. "Empty" finally means something. Near-misses are
# preserved (the index returns related papers), so it is not an oracle.
#
# Snapshot also restores the `future` regime and a VARYING `today` -- a live
# engine cannot hide a paper published after the stated date, so v1 had today
# pinned at 2026-08-18 and never saw that regime at all.
#
# Hyperparameters identical to v1 on purpose. Packages is dropped from the
# eval cells: it is parked (its grader charges -9 for denials and class names).
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"papers","situation":"single","backend":"snapshot","calib_file":"runs/arxiv-calib-20260816/calib-strict.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

OUT=runs/curve/20260826-papers-v2
say "training papers on the SNAPSHOT backend, 60 steps -> $OUT"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 --eval-cells 'trivia:single' --eval-cells-n 64 \
  --checkpoint-every 5 --seed 0 --lease-wait 1800 --required-gb 62 \
  --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "PAPERS V2 DONE -> $OUT"

say "retrofit: score the trivia row's papers cell on the same snapshot instrument"
uv run python scripts/retrofit_eval.py \
  --run runs/curve/20260822-trivia-v3 \
  --also runs/curve/20260822-trivia-v3-to120 \
  --cells papers:single \
  --every 10 --n 64 \
  --out runs/retrofit/papers-snapshot-trivia-v3 2>&1
say "MATRIX READY — trivia diagonal is unchanged from runs/retrofit/matrix2x2-trivia-v3"
