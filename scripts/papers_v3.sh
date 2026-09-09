#!/bin/bash
# lifecycle: one-off (archive when the papers row is in the lab book)
# Papers row on REAL search: the frozen Brave capture (data/serps/).
#
# Why v1 and v2 are in runs/archive-fake-search-20260828/:
#  * v1 searched the anonymous scrapers. 88.6% of the 41,966 cached hits were
#    off-topic; a real paper and a fabricated one were indistinguishable
#    through the tool (5% vs 11% relevant, inverted) -- the exact distinction
#    this task exists to teach.
#  * v2 searched the snapshot, a local title index. Correct for real papers,
#    but it answers "No results found" for anything fabricated, which is a
#    tell no engine gives. A policy learns "empty means decline" and never
#    meets the case that matters.
#
# What serps does instead, measured over the corpus: real papers mean
# relevance 0.997 (100% pass the gate), fictional 0.386 (23% pass). Asked
# about "PolyRecommender-Convolutions Discprecncies towards a Universal
# Length Space for High-Polyp", it returns, ranked first and real:
#     1. PolyRecommender: A Multimodal Recommendation System for Polymer Discovery
#        https://arxiv.org/html/2511.00375v1
# Noticing that is a DIFFERENT paper is the skill. An empty never asks for it.
#
# Instrument checks run before launch (2026-08-29):
#  * corpus covers 1857/1857 real and 300/300 fictional items in the pool
#  * of 1,454 queries the policy actually typed in the v1/v2 runs, 92.2%
#    resolve to the right captured row, 7.7% to none, 0.1% to a wrong row
#  * all four regimes verified end to end: known/uncertain/post find the
#    paper, future returns 0 hits, fictional returns real neighbours and
#    found_target=False
#
# NO trivia eval cell. The trivia tool is the same broken scraper: measured
# over 7,014 cached trivia searches, a gold alias appears in 4.4% of results
# (unknown band) and 6.5% (known). A trivia number from this run would
# measure the scraper, and this project has already lost three generations of
# work to numbers like that. Trivia comes back when it has a real tool.
#
# Hyperparameters identical to v1 and v2 on purpose -- the backend is the
# only thing that changed.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
TK='{"domain":"papers","situation":"single","backend":"serps","calib_file":"runs/arxiv-calib-20260816/calib-strict.jsonl","judge_backend":"local","judge_cache":"runs/judge/local-commitment-cache.jsonl"}'
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

OUT=runs/curve/20260829-papers-v3
say "training papers on the SERPS backend, 60 steps -> $OUT"
uv run python -m mlx_rl.train \
  --profile qwen36 --task honesty --task-kwargs "$TK" \
  --chat-kwargs '{"enable_thinking": false}' \
  --steps 60 --batch-prompts 12 --group-size 8 --group-stage1 4 --stage1-skip saturated \
  --update-adv-frac 0.5 --micro-batch 2 --grad-checkpoint \
  --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1024 --rollout-batch-size 48 \
  --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
  --eval-every 10 --eval-n 160 \
  --checkpoint-every 5 --seed 0 --lease-wait 1800 --required-gb 62 \
  --out "$OUT" 2>&1
uv run python scripts/promote_adapter.py "$OUT" --out "$OUT/promoted" 2>&1 || say "nothing to promote"
say "PAPERS V3 DONE -> $OUT"
