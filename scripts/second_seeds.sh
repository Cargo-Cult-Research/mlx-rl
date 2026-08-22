#!/bin/bash
# lifecycle: one-off (archive when every cell has two seeds)
# Second seed for every cell in the single-situation row.
#
# papers:single already has one, and it is why this is worth doing: the two
# seeds agreed on direction (+0.20 and +0.65 vs base +0.02 on their own square)
# but not on magnitude. One run per cell cannot tell those apart.
#
# Waits for the first row to finish rather than competing with it for the GPU.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
R=runs/night/20260818-night
OUT=runs/day/20260821-single
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

say "waiting for the first row to finish"
while ! grep -q "SINGLE ROW DONE" runs/single-row.log 2>/dev/null; do sleep 120; done
say "first row done, starting second seeds"

leg () {   # leg <name> <domain> <extra task kwargs>
  local name=$1 domain=$2 extra=$3
  local out="$R/$name"
  [ -f "$out/promoted/adapters.safetensors" ] && { say "skip $name"; return 0; }
  say "leg $name (seed 1)"
  uv run python -m mlx_rl.train \
    --profile qwen36 --task honesty \
    --task-kwargs "{\"domain\":\"$domain\",\"situation\":\"single\"$extra}" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 60 --batch-prompts 4 --group-size 8 --group-stage1 4 --stage1-skip saturated \
    --update-adv-frac 0.25 --micro-batch 1 --grad-checkpoint \
    --max-tool-rounds 4 --max-episode-tokens 8192 --max-new-tokens 1536 \
    --rollout-batch-size 32 \
    --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
    --eval-every 10 --eval-n 16 --checkpoint-every 5 --seed 1 \
    --lease-wait 900 --required-gb 62 --out "$out" 2>&1
  uv run python scripts/promote_adapter.py "$out" --out "$out/promoted" 2>&1 || say "$name: nothing to promote"
}

leg trivia-single-seed1   trivia   ',"calib_file":"runs/qa-calib-20260724/calib.jsonl"'
leg packages-single-seed1 packages ''

CELLS=trivia:single,papers:single,packages:single
for a in trivia-single-seed1 packages-single-seed1; do
  [ -f "$OUT/$a/results.json" ] && { say "skip eval $a"; continue; }
  [ -f "$R/$a/promoted/adapters.safetensors" ] || { say "no adapter for $a"; continue; }
  say "eval $a"
  uv run python scripts/matrix_eval.py --cells "$CELLS" --arm "$a=$PWD/$R/$a/promoted" \
    --n 32 --k 2 --no-manage-machine --out "$OUT/$a" 2>&1
done
say "SECOND SEEDS DONE"
