#!/bin/bash
# qa-gloveC-patient.sh — 2026-07-31: ARM C attempts 1-3 failed the memory
# guard (68.1 GB needed; migration file-cache holds the margin). Instead of
# bouncing the serving backend on every blind retry, pre-check headroom
# WITHOUT displacement: available_gb() + 22 (the lens's wired Metal pages,
# freed on displacement) must clear the guard's 68.1/0.9 = 75.7 with a
# little slack. Poll every 10 min as the cache decays; give up after 8 h.
set -uo pipefail
cd "$HOME/code/mlx-rl"
LOG="runs/qa-gloveC-patient.log"
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
tg(){
    source "$HOME/code/housekeeping/.env" 2>/dev/null || true
    [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s -m 20 \
        "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d chat_id="$TELEGRAM_USER_ID" -d text="gloveC-patient: $*" >/dev/null
    true
}

NEED=76.5   # 68.1/0.9 + slack
DEADLINE=$(( $(date +%s) + 8*3600 ))
while :; do
    AVAIL=$(.venv/bin/python -c "from mlx_rl.memory import available_gb; print(available_gb())")
    LENS=0
    lsof -iTCP:8084 -sTCP:LISTEN -n 2>/dev/null | grep -q . && LENS=22
    OK=$(.venv/bin/python -c "print(1 if $AVAIL + $LENS >= $NEED else 0)")
    say "headroom check: avail=$AVAIL lens_reclaim=$LENS need=$NEED ok=$OK"
    [ "$OK" = 1 ] && break
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        tg "gave up after 8h — headroom never cleared $NEED GB (avail $AVAIL + lens $LENS). Manual look needed."
        exit 1
    fi
    sleep 600
done
say "headroom cleared — launching ARM C"
tg "headroom cleared — ARM C launching now"
bash "$HOME/code/housekeeping/note.sh" "gloveC-patient: headroom cleared, ARM C attempt 4 launching" || true
exec bash runs/qa-gloveC_run.sh
