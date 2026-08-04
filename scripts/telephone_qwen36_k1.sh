#!/bin/bash
# lifecycle: one-off (archive when the telephone arc concludes)
# Arm B of the telephone experiment (ears/experiments/2026-08-03-telephone):
# qwen36, k=1 channel, KL 0 (drift is the point). Experiments memlease block
# — must NOT displace :8084. required-gb 45 overrides the CoT-calibrated
# estimator (this run generates 8 tokens/rollout); SwapGuard enforces it.
# After training, the day-0 codebook probe runs on the freed weights.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -m mlx_rl.train \
  --profile qwen36 --task telephone \
  --steps 300 --batch-prompts 8 --group-size 8 \
  --max-new-tokens 8 --kl-coef 0.0 --micro-batch 2 \
  --lease-block experiments --required-gb 45 --lease-wait 900 \
  --out runs/telephone-qwen36-k1

# probe holds its own experiments lease (the trainer released its on exit)
MEMLEASE=~/code/housekeeping/memlease.py
HOLDER="telephone-probe:$$"
python3 "$MEMLEASE" acquire "$HOLDER" --block experiments --ensure-gb 30 \
  --wait 900 --note "telephone day-0 codebook probe (qwen36)"
trap 'python3 "$MEMLEASE" release "$HOLDER"' EXIT
.venv/bin/python scripts/telephone_probe.py \
  --profile qwen36 --vocab-sample 256 --score-chunk 32 \
  --out runs/telephone-qwen36-k1/probe-day0.json
