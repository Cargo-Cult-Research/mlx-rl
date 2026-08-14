#!/bin/bash
# lifecycle: one-off (archive when the DeepCoder pilot decides the full-sweep design)
# DeepCoder 200-problem pilot: the SAME seeded sample through both sweep
# models, before committing the full ~19k schedule. Measures (a) qwen36's
# pass/length distributions -> what adaptive-k would save, (b) the small
# model's throughput ratio and (c) whether "4B solves it" predicts "qwen36
# all-pass" (the cascade's validity). :8084 is off at rest, so each leg's
# experiments-block lease has the machine to itself; the sweep's swap guard
# (10 GB, journal+Telegram) is the backstop.
set -u
cd "$(dirname "$0")/.."
NOTE="bash $HOME/code/housekeeping/note.sh"

$NOTE "deepcoder pilot: 200-problem seeded sample x pass@5 T=1.0 cap 4096, qwen36 then Qwen3-4B; log runs/sweeps/deepcoder-pilot.log"

.venv/bin/python scripts/difficulty_sweep.py --task deepcoder --k 5 \
    --temperature 1.0 --max-new-tokens 4096 --batch-prompts 10 \
    --sample 200 --required-gb 38 \
    --out runs/sweeps/deepcoder-pilot-qwen36.jsonl || exit $?

.venv/bin/python scripts/difficulty_sweep.py --task deepcoder --k 5 \
    --temperature 1.0 --max-new-tokens 4096 --batch-prompts 10 \
    --sample 200 --required-gb 12 \
    --model "$HOME/models/mlx/Qwen3-4B-4bit" \
    --out runs/sweeps/deepcoder-pilot-qwen3-4b.jsonl || exit $?

$NOTE "deepcoder pilot: both legs complete"
