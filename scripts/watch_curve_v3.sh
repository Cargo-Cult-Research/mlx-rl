#!/bin/bash
# lifecycle: one-off (archive with curve_v3.sh)
# Babysitter for curve v3: if the trainer dies or metrics stall, say so in
# this log AND Telegram Urs (fail-loud rule: a log banner nobody tails is
# silence). Exits after reporting once, or quietly when the run completes.
set -u
cd "${MLX_RL_ROOT:-$HOME/code/mlx-rl}"
M=runs/curve/20260822-trivia-v3/metrics.jsonl
LOG=runs/curve-v3.log
tg () {  # best-effort Telegram; never fails the watcher
  set -a; . ~/code/housekeeping/.env 2>/dev/null; set +a
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -sm 10 \
    "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_USER_ID}" -d text="$1" >/dev/null 2>&1
}
while true; do
  sleep 600
  if ! pgrep -f "mlx_rl.train" >/dev/null; then
    if grep -q "CURVE V3 DONE" "$LOG" 2>/dev/null; then exit 0; fi
    msg="mlx-rl curve v3: TRAINING NOT RUNNING (no DONE marker). Last log: $(tail -2 "$LOG" 2>/dev/null | cut -c1-200)"
    echo "$(date '+%H:%M') $msg"; tg "$msg"; exit 1
  fi
  if [ -f "$M" ]; then
    age=$(( $(date +%s) - $(stat -f %m "$M") ))
    if [ "$age" -gt 2400 ]; then
      msg="mlx-rl curve v3: STALLED — metrics ${age}s old, trainer still alive"
      echo "$(date '+%H:%M') $msg"; tg "$msg"; exit 1
    fi
  fi
done
