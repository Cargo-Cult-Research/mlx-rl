#!/bin/bash
# qa-gloveC-relaunch.sh — 2026-07-31: ARM C failed its memory guard (needs
# 68 GB, 49 "available") because a detached rsync disk migration
# (/Volumes/data -> /Volumes/data-new) is churning ~32 GB of speculative
# file cache that psutil-available discounts. The migration is user
# infra — do not touch it. Wait for rsync to exit, give the cache a
# settle window, then relaunch the ARM C driver.
set -uo pipefail
LOG="$HOME/code/mlx-rl/runs/qa-gloveC-relaunch.log"
say(){ echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
say "waiting for rsync disk migration to finish"
while pgrep -x rsync >/dev/null 2>&1; do sleep 120; done
say "rsync gone — settling 5 min for cache reclaim"
sleep 300
say "relaunching ARM C"
bash "$HOME/code/housekeeping/note.sh" "rsync migration finished — relaunching ARM C" || true
exec bash "$HOME/code/mlx-rl/runs/qa-gloveC_run.sh"
