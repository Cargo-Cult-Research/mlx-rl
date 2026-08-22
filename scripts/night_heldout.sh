#!/bin/bash
# Held-out transfer experiment.
#
#   trained squares : trivia:single, trivia:toolfail, papers:single, papers:toolfail
#   held out        : the whole packages row, and the whole swamp column
#
# One adapter per trained square, each starting from the BASE model (never from
# an existing adapter — starting from the web-tools adapter would smuggle arXiv
# training into every leg and destroy the hold-out). Then the full 3x3 grid is
# evaluated, held-out cells first so a short night still yields the experiment.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
STAMP=20260818-night
ROOT=runs/night/$STAMP
mkdir -p "$ROOT"
PAPERS_CALIB='"calib_file":"runs/arxiv-calib-20260816/calib-strict.jsonl"'
TRIVIA_CALIB='"calib_file":"runs/qa-calib-20260724/calib.jsonl"'

say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

# Training writes step checkpoints (adapter-00060.safetensors); a loadable
# adapter dir wants adapters.safetensors. Promote the newest checkpoint.
promote () {
  uv run python scripts/promote_adapter.py "$1" --out "$1/promoted" >/dev/null 2>&1
}

# Brainbow holds the experiments lease while Urs is using it; take the lease
# when it is free, otherwise share the machine (Brainbow idles near 0 GB).
lease_flags () {
  if python3 ~/code/housekeeping/memlease.py status >/dev/null 2>&1; then
    echo "--lease-wait 900 --required-gb 60"
  else
    echo "--no-manage-machine"
  fi
}

leg () {
  local name=$1 domain=$2 situation=$3 calib=$4
  local out="$ROOT/$name"
  if [ -f "$out/promoted/adapters.safetensors" ]; then say "skip $name (already trained)"; return 0; fi
  say "leg $name  ($domain:$situation)"
  # shellcheck disable=SC2046
  uv run python -m mlx_rl.train \
    --profile qwen36 --task honesty \
    --task-kwargs "{\"domain\":\"$domain\",\"situation\":\"$situation\",$calib}" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 60 --batch-prompts 4 --group-size 8 --group-stage1 4 --stage1-skip saturated \
    --update-adv-frac 0.25 --micro-batch 1 --grad-checkpoint \
    --max-tool-rounds 4 --max-episode-tokens 8192 --max-new-tokens 1536 \
    --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
    --eval-every 10 --eval-n 16 --checkpoint-every 10 --seed 0 \
    $(lease_flags) --out "$out" 2>&1
  promote "$out" && say "leg $name done, adapter promoted" || say "leg $name FAILED (no checkpoint)"
}

leg trivia-single   trivia "single"   "$TRIVIA_CALIB"
leg trivia-toolfail trivia "toolfail" "$TRIVIA_CALIB"
leg papers-single   papers "single"   "$PAPERS_CALIB"
leg papers-toolfail papers "toolfail" "$PAPERS_CALIB"

# One stacked adapter carrying all four squares (exact rank concatenation).
STACK="$ROOT/stack-all"
if [ ! -f "$STACK/adapters.safetensors" ]; then
  say "stacking four adapters"
  uv run python scripts/stack_adapters.py \
    "$ROOT"/trivia-single/promoted "$ROOT"/trivia-toolfail/promoted \
    "$ROOT"/papers-single/promoted "$ROOT"/papers-toolfail/promoted \
    --out "$STACK" 2>&1
fi

ARMS=(--arm base=)
for a in trivia-single trivia-toolfail papers-single papers-toolfail; do
  [ -f "$ROOT/$a/promoted/adapters.safetensors" ] && ARMS+=(--arm "$a=$PWD/$ROOT/$a/promoted")
done
[ -f "$STACK/adapters.safetensors" ] && ARMS+=(--arm "stack-all=$PWD/$STACK")

# Phase 1 — the actual experiment: squares no leg was trained on.
say "phase 1: held-out cells"
uv run python scripts/matrix_eval.py \
  --cells packages:single,packages:toolfail,packages:swamp,trivia:swamp,papers:swamp \
  "${ARMS[@]}" --n 32 --k 2 --max-new-tokens 2048 --no-manage-machine \
  --out "$ROOT/eval-heldout" 2>&1

# Phase 2 — the trained squares, for the diagonal of the grid.
say "phase 2: trained cells"
uv run python scripts/matrix_eval.py \
  --cells trivia:single,trivia:toolfail,papers:single,papers:toolfail \
  "${ARMS[@]}" --n 32 --k 2 --no-manage-machine \
  --out "$ROOT/eval-trained" 2>&1

# Bonus, only if the night has room: retrain one square with a different seed.
# Every headline we have rests on a single run; this is the cheapest check that
# the effect is not one lucky seed.
REP="$ROOT/papers-single-seed1"
if [ ! -f "$REP/promoted/adapters.safetensors" ]; then
  say "replication leg: papers:single, seed 1"
  uv run python -m mlx_rl.train \
    --profile qwen36 --task honesty \
    --task-kwargs "{\"domain\":\"papers\",\"situation\":\"single\",$PAPERS_CALIB}" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 60 --batch-prompts 4 --group-size 8 --group-stage1 4 --stage1-skip saturated \
    --update-adv-frac 0.25 --micro-batch 1 --grad-checkpoint \
    --max-tool-rounds 4 --max-episode-tokens 8192 --max-new-tokens 1536 \
    --lr 3e-6 --kl-coef 0.01 --rank 16 --lora-layers 12 \
    --eval-every 10 --eval-n 16 --checkpoint-every 10 --seed 1 \
    $(lease_flags) --out "$REP" 2>&1
  promote "$REP" || say "replication leg produced no checkpoint"
fi
if [ -f "$REP/promoted/adapters.safetensors" ]; then
  say "phase 3: does the replication land in the same place?"
  uv run python scripts/matrix_eval.py \
    --cells papers:single,packages:single,packages:swamp \
    --arm base= --arm "papers-single=$PWD/$ROOT/papers-single/promoted" \
    --arm "papers-single-seed1=$PWD/$REP/promoted" \
    --n 32 --k 2 --max-new-tokens 2048 --no-manage-machine \
    --out "$ROOT/eval-replication" 2>&1
fi

say "ALL DONE"
