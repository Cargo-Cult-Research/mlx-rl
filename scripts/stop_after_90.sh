#!/bin/bash
# lifecycle: one-off (delete once the trivia curve is closed out)
# Stop the trivia extension at step 90 rather than 120: held-out reward has
# been flat since step 60 (-0.101, -0.180, -0.214) while KL keeps rising, so
# the remaining 30 steps buy drift, not signal. Step 90 is both a checkpoint
# (every 5) and an eval (every 10), so stopping there costs nothing and
# leaves a scored final point. SIGINT was swallowed mid-backward-pass; the
# memlease registers its holder pid and stale holders are reaped, so a TERM
# after the row lands is safe.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
M=runs/curve/20260822-trivia-v3-to120/metrics.jsonl
PID=30593
say () { echo "=== $(date '+%H:%M:%S') $*"; }
say "waiting for the step-90 eval row"
for _ in $(seq 1 240); do          # 240 x 30s = 2h ceiling
  if grep -q '"step": 90.*eval_reward' "$M" 2>/dev/null; then say "step 90 scored"; break; fi
  if ! ps -p $PID >/dev/null 2>&1; then say "trainer already gone"; break; fi
  sleep 30
done
if ps -p $PID >/dev/null 2>&1; then
  say "TERM -> $PID"; kill -TERM $PID; sleep 45
  ps -p $PID >/dev/null 2>&1 && { say "KILL -> $PID"; kill -9 $PID; sleep 10; }
fi
pkill -f extend_v3.sh 2>/dev/null && say "driver stopped (no promote-from-driver)"
uv run python scripts/promote_adapter.py runs/curve/20260822-trivia-v3-to120 \
  --out runs/curve/20260822-trivia-v3-to120/promoted 2>&1 || say "promote failed"
say "STOPPED; newest checkpoint:"; ls -t runs/curve/20260822-trivia-v3-to120/adapters/*.safetensors | head -1
