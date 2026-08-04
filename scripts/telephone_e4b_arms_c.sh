#!/bin/bash
# lifecycle: one-off (archive when the telephone arc concludes)
# Arms C of the telephone experiment (ears/experiments/2026-08-03-telephone).
# Arm B died of GRPO entropy collapse at the quality-word plateau (last real
# update step 88/300). Both C arms restore group variance via one injected
# off-policy member per group from the 512-token probe codebook (legal:
# old_lp recomputed, enters the surrogate at ratio 1).
#   C1: k=1 + inject          — can the policy ABSORB a mined code?
#   C2: k=2 + ban + inject    — label words score zero by rule; the policy
#                               must find synonym/mnemonic codes the frozen
#                               twin decodes ('GGG', 'CaO' style are legal).
set -euo pipefail
cd "$(dirname "$0")/.."
CODEBOOK=runs/telephone-e4b-k1/probe-day0-512.json

.venv/bin/python -m mlx_rl.train \
  --profile e4b --task telephone \
  --task-kwargs "{\"inject_codebook\": \"$CODEBOOK\"}" \
  --steps 300 --batch-prompts 8 --group-size 8 --inject-r 1 \
  --max-new-tokens 8 --kl-coef 0.0 --micro-batch 2 \
  --lease-block experiments --required-gb 25 --lease-wait 3600 \
  --out runs/telephone-e4b-c1-inject

.venv/bin/python -m mlx_rl.train \
  --profile e4b --task telephone \
  --task-kwargs "{\"inject_codebook\": \"$CODEBOOK\", \"k_tokens\": 2, \"ban_label_words\": true}" \
  --steps 300 --batch-prompts 8 --group-size 8 --inject-r 1 \
  --max-new-tokens 8 --kl-coef 0.0 --micro-batch 2 \
  --lease-block experiments --required-gb 25 --lease-wait 3600 \
  --out runs/telephone-e4b-c2-ban
