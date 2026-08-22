#!/bin/bash
# lifecycle: one-off (archive now — lambda arms were ruled out; kept only for the record)
# Does task-arithmetic scaling rescue the stack?
#   lambda=1.00  plain sum -- what we ran, far below base
#   lambda=0.50 / 0.25     -- 1/k for k=4 is the literature's first guess
# Plus the two failed phase-3 arms, retried now that the eval backs the chunk
# off on an out-of-memory instead of dying.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
R=runs/night/20260818-night
OUT=runs/day/20260821-stack
mkdir -p "$OUT"
CELLS=trivia:single,trivia:toolfail,papers:single,papers:toolfail
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }

arm () {   # arm <dir> <name> <adapter> <cells> [extra...]
  local out=$1 name=$2 adapter=$3 cells=$4; shift 4
  [ -f "$out/$name/results.json" ] && { say "skip $name"; return 0; }
  say "eval $name"
  uv run python scripts/matrix_eval.py --cells "$cells" --arm "$name=$adapter" \
    --n 32 --k 2 --no-manage-machine --out "$out/$name" "$@" 2>&1
}

for lam in 0.50 0.25; do
  arm "$OUT" "stack-lam$lam" "$PWD/$R/stack-lam$lam" "$CELLS"
done

say "phase 3 retry: the seed replication that has failed three times"
REP=papers:single,packages:single,packages:swamp
arm runs/day/20260821/replication papers-single "$PWD/$R/papers-single/promoted" "$REP" --max-new-tokens 2048
arm runs/day/20260821/replication papers-single-seed1 "$PWD/$R/papers-single-seed1/promoted" "$REP" --max-new-tokens 2048

say "also filling the papers:toolfail cell the heavy-cell OOM took out"
for a in trivia-single trivia-toolfail papers-single stack-all; do
  [ -f "$R/$a/promoted/adapters.safetensors" ] && A="$PWD/$R/$a/promoted" || A="$PWD/$R/$a"
  [ -e "$A" ] && arm runs/day/20260821-fill "$a" "$A" papers:toolfail
done
say "STACK SWEEP DONE"
