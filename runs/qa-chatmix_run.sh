#!/bin/bash
# qa-chatmix_run.sh — 2026-07-30: frame-mixture abstention run. 50% tag frames
# (verifiable local grading, keeps inject-r demonstrations working) + 50% free
# chat frames graded by the Opus commitment-parser judge (src/mlx_rl/judge.py,
# headless claude on the plan — one batched call per training step).
#
#   bash runs/qa-chatmix_run.sh sanity   # 3 steps, small eval — integration check
#   bash runs/qa-chatmix_run.sh full     # 200 steps (overnight; ~200-400 judge calls)
#
# Success metric afterwards: scripts/qa_chat_probe.py + papers_recall_probe.py
# (eval-only frames, untouched by training) — chat-frame hedge+denial on
# unknowable entities should move toward the in-format 0.9 while famous-paper
# answering stays ~1.0.
set -uo pipefail
cd "$HOME/code/mlx-rl"
MODE="${1:-sanity}"
STAMP=$(date +%Y%m%d)
if [ "$MODE" = full ]; then STEPS=200; EVALN=200; WATCHDOG="--abort-inactive-window 30"; else STEPS=3; EVALN=24; WATCHDOG=""; fi
OUT="runs/qa-chatmix-$MODE-$STAMP"
LOG="$OUT.log"
mkdir -p "$OUT"
echo "=== $(date '+%F %T') qa-chatmix $MODE start -> $OUT" | tee -a "$LOG"

.venv/bin/mlx-rl-train \
    --profile qwen36 --task qa_abstain \
    --task-kwargs '{"calib_file": "runs/qa-calib-20260724/calib.jsonl",
                    "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
                    "chat_frac": 0.5,
                    "judge_cache": "runs/judge/qa-abstain-cache.jsonl"}' \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps "$STEPS" --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
    --eval-every 10 --eval-n "$EVALN" --checkpoint-every 20 \
    --seed 0 $WATCHDOG \
    --out "$OUT" >>"$LOG" 2>&1
rc=$?
echo "=== $(date '+%F %T') qa-chatmix $MODE done rc=$rc" | tee -a "$LOG"
# fail-loud must reach the human: Telegram the outcome either way
source "$HOME/code/housekeeping/.env" 2>/dev/null || true
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
    curl -s -m 20 "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" \
        -d text="qa-chatmix $MODE done rc=$rc — $OUT$([ -f "$OUT/ABORTED" ] && echo ' (ABORTED)')" >/dev/null || true
fi
