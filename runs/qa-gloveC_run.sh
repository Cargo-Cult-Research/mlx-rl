#!/bin/bash
# qa-gloveC_run.sh — 2026-07-31 evening: ARM C, the known-side-pressure arm.
# A-200+glove crossed the bridge (chat arXiv hedging 0.78-0.88, famous kept)
# but pays 0.29 hedge-on-known in chat; binding shows real per-item signal
# (pearson -0.44, decline known 0.09 vs unknown 0.41) with headroom. Single
# variable vs ARM A: chat_band_mix 0.15/0.35/0.50 -> 0.35/0.35/0.30 — more
# known-band chat groups so hedging on answerable questions is punished as
# often as guessing on unknowable ones. Everything else identical (seed 0,
# lr 3e-6, 200 steps, margin 12). Probes chained, ckpt 160+200 both.
set -uo pipefail
cd "$HOME/code/mlx-rl"
PY=.venv/bin/python
STAMP=20260731
LOG="runs/qa-gloveC-$STAMP.log"
CALIB=runs/qa-calib-20260724/calib.jsonl
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveC: $*" >/dev/null
    true
}

KW_C='{"calib_file": "runs/qa-calib-20260724/calib.jsonl",
 "band_mix": {"known": 0.65, "uncertain": 0.25, "unknown": 0.1},
 "chat_frac": 0.5, "system": "honesty",
 "chat_band_mix": {"known": 0.35, "uncertain": 0.35, "unknown": 0.3},
 "judge_cache": "runs/judge/qa-abstain-cache.jsonl"}'

OUT="runs/qa-gloveC-$STAMP"
mkdir -p "$OUT"
say "ARM C start (known-side pressure, chat bands 0.35/0.35/0.30)"
.venv/bin/mlx-rl-train \
    --profile qwen36 --task qa_abstain \
    --task-kwargs "$KW_C" \
    --chat-kwargs '{"enable_thinking": false}' \
    --steps 200 --batch-prompts 8 --group-size 8 --micro-batch 4 \
    --max-new-tokens 192 --lora-layers 12 --grad-checkpoint \
    --lr 3e-6 --kl-coef 0.01 --inject-r 1 \
    --eval-every 10 --eval-n 200 --checkpoint-every 20 \
    --seed 0 --abort-inactive-window 30 --swap-guard-margin 12 \
    --out "$OUT" >>"$LOG" 2>&1
rc=$?
say "ARM C rc=$rc"
if [ $rc -ne 0 ]; then
    tg "ARM C FAILED rc=$rc $([ -f "$OUT/ABORTED" ] && echo ABORTED) — $LOG"
    exit 1
fi
for CK in 160 200; do
    AD="$HOME/models/adapters/qa-gloveC-$CK-$STAMP"
    mkdir -p "$AD"
    cp "$OUT/adapters/adapter-00$CK.safetensors" "$AD/adapters.safetensors"
    cp "$HOME/models/adapters/qa-chatmix-20260730/adapter_config.json" "$AD/adapter_config.json"
    echo "promoted $OUT ckpt $CK (ARM C: known-side pressure)" > "$AD/MANIFEST.md"
    say "probes[C-$CK] chat glove-ON"
    $PY scripts/qa_chat_probe.py --calib "$CALIB" --per-bucket 6 --k 4 \
        --adapter "$AD" --system honesty \
        --out "runs/qa-chat-gloveC$CK-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
    say "probes[C-$CK] papers-recall glove-ON"
    $PY scripts/papers_recall_probe.py --k 4 --adapter "$AD" --system honesty \
        --out "runs/papers-recall-gloveC$CK-sys-$STAMP" >>"$LOG" 2>&1 || say "probe nonzero"
done
say "binding[C-200]"
$PY scripts/qa_binding_correlation.py \
    --adapter "$HOME/models/adapters/qa-gloveC-200-$STAMP" --n 200 --k 8 \
    --out "runs/qa-binding-C200-$STAMP" >>"$LOG" 2>&1 || say "binding nonzero"
tg "ARM C DONE rc=0 — probes + binding in runs/ (qa-gloveC-$STAMP)"
say "ARM C DONE"
