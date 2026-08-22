#!/bin/bash
# lifecycle: one-off (archive when its experiment arc is written up)
# Finish what the 08-18 night started:
#   1. the fourth training leg (papers:toolfail), killed at step 6 by the swap guard
#   2. the stacked adapter, which needs all four legs
#   3. eval phases 2 and 3, killed by a Metal OOM
#
# Judge: Opus, deliberately. The three sibling legs were trained under Opus and
# the whole point of the 2x2 is that its corners are comparable; the local judge
# starts with the round AFTER this one.
#
# Memory: the leg died with swap +15.5 GB. Episode cap and rollout batch come
# down (they bound the long tail that lands in the backward pass); the training
# knobs that shape the signal -- steps, batch_prompts, group_size, lr -- are
# left identical to the siblings.
#
# Evals run ONE ARM PER PROCESS: loading several adapters into one process is
# what exhausted GPU memory last time.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
ROOT=runs/night/20260818-night          # same tree: siblings and promoted adapters live here
DAY=runs/day/20260821
mkdir -p "$DAY"
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

LEG="$ROOT/papers-toolfail"
if [ ! -f "$LEG/promoted/adapters.safetensors" ]; then
  say "leg papers-toolfail (retry, smaller footprint)"
  rm -rf "$LEG"
  uv run python -m mlx_rl.train \
    --profile qwen36 --task honesty \
    --task-kwargs '{"domain":"papers","situation":"toolfail","calib_file":"runs/arxiv-calib-20260816/calib-strict.jsonl"}' \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 60 --batch-prompts 4 --group-size 8 --group-stage1 4 --stage1-skip saturated \
    --update-adv-frac 0.25 --micro-batch 1 --grad-checkpoint \
    --max-tool-rounds 4 --max-episode-tokens 6144 --max-new-tokens 1536 \
    --rollout-batch-size 32 \
    --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
    --eval-every 10 --eval-n 16 --checkpoint-every 5 --seed 0 \
    --lease-wait 900 --required-gb 62 --out "$LEG" 2>&1
  uv run python scripts/promote_adapter.py "$LEG" --out "$LEG/promoted" 2>&1 || say "no checkpoint to promote"
fi

STACK="$ROOT/stack-all"
if [ ! -f "$STACK/adapters.safetensors" ] && [ -f "$LEG/promoted/adapters.safetensors" ]; then
  say "stacking four per-square adapters"
  uv run python scripts/stack_adapters.py \
    "$ROOT"/trivia-single/promoted "$ROOT"/trivia-toolfail/promoted \
    "$ROOT"/papers-single/promoted "$ROOT"/papers-toolfail/promoted \
    --out "$STACK" 2>&1
fi

arm_eval () {           # arm_eval <out-dir> <name> <adapter> <cells> [extra...]
  local out=$1 name=$2 adapter=$3 cells=$4; shift 4
  [ -f "$out/$name/results.json" ] && { say "skip $name"; return 0; }
  say "eval $name on $cells"
  uv run python scripts/matrix_eval.py --cells "$cells" --arm "$name=$adapter" \
    --n 32 --k 2 --no-manage-machine --out "$out/$name" "$@" 2>&1
}

TRAINED=trivia:single,trivia:toolfail,papers:single,papers:toolfail
say "phase 2: the trained squares"
arm_eval "$DAY/trained" base "" "$TRAINED"
for a in trivia-single trivia-toolfail papers-single papers-toolfail; do
  [ -f "$ROOT/$a/promoted/adapters.safetensors" ] && arm_eval "$DAY/trained" "$a" "$PWD/$ROOT/$a/promoted" "$TRAINED"
done
[ -f "$STACK/adapters.safetensors" ] && arm_eval "$DAY/trained" stack-all "$PWD/$STACK" "$TRAINED"

say "phase 3: does the second seed land in the same place?"
REPCELLS=papers:single,packages:single,packages:swamp
arm_eval "$DAY/replication" base "" "$REPCELLS" --max-new-tokens 2048
arm_eval "$DAY/replication" papers-single "$PWD/$ROOT/papers-single/promoted" "$REPCELLS" --max-new-tokens 2048
arm_eval "$DAY/replication" papers-single-seed1 "$PWD/$ROOT/papers-single-seed1/promoted" "$REPCELLS" --max-new-tokens 2048

say "ROUND 2 DONE"
