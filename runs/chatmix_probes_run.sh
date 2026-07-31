#!/bin/bash
# chatmix_probes_run.sh — 2026-07-30: the pre-declared success test for the
# qa-chatmix frame-mixture run. RL side only (base results from 07-29 stand:
# runs/qa-chat-base-20260729, runs/papers-recall-base-20260729,
# runs/papers-recall-fmt-base-20260729 — the base model is unchanged).
# Three probes, sequential (each loads the model and takes the memlease):
#   1. qa_chat_probe        — conversational transfer (hedge/denial in chat)
#   2. papers_recall chat   — arXiv entities, free chat frame
#   3. papers_recall fmt    — arXiv entities, training tag frame (control)
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
# mlx_lm.load() needs the PROMOTED layout (lora_parameters config +
# adapters.safetensors), not the raw run dir — first attempt died on that.
ADAPTER="$HOME/models/adapters/qa-chatmix-20260730"
LOG=runs/chatmix-probes-20260730.log
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

say "chat probe start (adapter=$ADAPTER)"
$PY scripts/qa_chat_probe.py --calib runs/qa-calib-20260724/calib.jsonl \
    --per-bucket 6 --k 4 --adapter "$ADAPTER" \
    --out runs/qa-chat-chatmix-20260730 >>"$LOG" 2>&1 || say "chat probe nonzero"

say "papers recall (chat frame) start"
$PY scripts/papers_recall_probe.py --k 4 --adapter "$ADAPTER" \
    --out runs/papers-recall-chatmix-20260730 >>"$LOG" 2>&1 || say "recall chat nonzero"

say "papers recall (in-format) start"
$PY scripts/papers_recall_probe.py --k 4 --in-format --max-new-tokens 96 \
    --adapter "$ADAPTER" \
    --out runs/papers-recall-fmt-chatmix-20260730 >>"$LOG" 2>&1 || say "recall fmt nonzero"

say "ALL PROBES DONE"
source "$HOME/code/housekeeping/.env" 2>/dev/null || true
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
    curl -s -m 20 "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" \
        -d text="chatmix probes done — runs/chatmix-probes-20260730.log" >/dev/null || true
fi
