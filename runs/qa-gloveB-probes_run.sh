#!/bin/bash
# qa-gloveB-probes_run.sh — 2026-07-31: ARM B's probes, salvaged after the
# night driver's bash-3.2 `local` bug killed it before stage-3 probes.
# Waits for the ARM A retry driver to finish (it holds the machine), then
# probes the promoted qa-gloveB-20260730 adapter glove-ON and glove-OFF.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260730
LOG="runs/qa-glove-night-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
AD="$HOME/models/adapters/qa-gloveB-20260730"
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveB-probes: $*" >/dev/null
    true
}

RETRY_PID="${1:?usage: qa-gloveB-probes_run.sh <retry_driver_pid>}"
say "gloveB probes: waiting for arm-A retry pid $RETRY_PID"
while kill -0 "$RETRY_PID" 2>/dev/null; do sleep 60; done

say "probes[gloveB] chat-probe glove-ON"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" --system honesty \
    --out "runs/qa-chat-gloveB-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[gloveB] papers-recall glove-ON"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$AD" --system honesty \
    --out "runs/papers-recall-gloveB-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
say "probes[gloveB] chat-probe glove-OFF"
$PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
    --adapter "$AD" \
    --out "runs/qa-chat-gloveB-nosys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
tg "arm B probes done"
say "gloveB probes DONE"
