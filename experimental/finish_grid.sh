#!/bin/bash
# lifecycle: one-off (archive when the seed replication and papers:toolfail column land)
# The two things worth running that have nothing to do with merging:
#   1. the seed replication, failed three times now
#   2. the papers:toolfail column the heavy-cell OOM emptied
# Both now benefit from matrix_eval backing its chunk off on an OOM.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
R=runs/night/20260818-night
say () { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
arm () {
  local out=$1 name=$2 adapter=$3 cells=$4; shift 4
  [ -f "$out/$name/results.json" ] && { say "skip $name"; return 0; }
  say "eval $name on $cells"
  uv run python scripts/matrix_eval.py --cells "$cells" --arm "$name=$adapter" \
    --n 32 --k 2 --no-manage-machine --out "$out/$name" "$@" 2>&1
}
REP=papers:single,packages:single,packages:swamp
arm runs/day/20260821/replication papers-single       "$PWD/$R/papers-single/promoted"       "$REP" --max-new-tokens 2048
arm runs/day/20260821/replication papers-single-seed1 "$PWD/$R/papers-single-seed1/promoted" "$REP" --max-new-tokens 2048
for a in trivia-single trivia-toolfail papers-single; do
  arm runs/day/20260821-fill "$a" "$PWD/$R/$a/promoted" papers:toolfail
done
say "GRID FILL DONE"
