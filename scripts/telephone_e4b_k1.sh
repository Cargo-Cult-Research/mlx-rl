#!/bin/bash
# lifecycle: one-off (archive when the telephone arc concludes)
# Arm B of the telephone experiment (ears/experiments/2026-08-03-telephone):
# gemma-4-E4B (vlm-loaded), k=1 channel, KL 0 (drift is the point), pure
# p_correct reward. Experiments memlease block — coexists with :8084.
# required-gb 25: ~9 GB weights + rollout/scoring activations + listener
# batch head slabs; SwapGuard enforces the claim.
# Afterwards: the honest day-0 capacity estimate (512-token codebook probe).
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -m mlx_rl.train \
  --profile e4b --task telephone \
  --steps 300 --batch-prompts 8 --group-size 8 \
  --max-new-tokens 8 --kl-coef 0.0 --micro-batch 2 \
  --lease-block experiments --required-gb 25 --lease-wait 1800 \
  --out runs/telephone-e4b-k1

MEMLEASE=~/code/housekeeping/memlease.py
HOLDER="telephone-probe:$$"
# NB no --ensure-gb: that flag displaces :8084 and is exclusive-only
python3 "$MEMLEASE" acquire "$HOLDER" --block experiments \
  --wait 1800 --note "telephone day-0 codebook probe (e4b, 512 tokens)"
trap 'python3 "$MEMLEASE" release "$HOLDER" --block experiments' EXIT
.venv/bin/python scripts/telephone_probe.py \
  --profile e4b --vocab-sample 512 \
  --out runs/telephone-e4b-k1/probe-day0-512.json
