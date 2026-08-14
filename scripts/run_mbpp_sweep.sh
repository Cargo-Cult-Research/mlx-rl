#!/bin/bash
# lifecycle: one-off (archive when the MBPP difficulty atlas is complete)
# MBPP pass@5 sweep driver: 3 temperature legs under ONE exclusive memlease.
#
# Why exclusive (not the experiments block): the sweep needs ~30 GB beside a
# 22 GB lens server the wedge-watchdog force-loads every 5 min, plus the
# colima VM — that co-residency swapped 11.8 GB and tripped the guard on
# 2026-08-13. With Moss decommissioned, displacing :8084 for the run is the
# sanctioned path (research priority). The lease is PID-tied to this driver:
# lens is displaced once, restored once at the end, and the restore is
# VERIFIED (the known memlease failure leaves the backend disabled+down).
set -u
cd "$(dirname "$0")/.."
MEMLEASE="python3 $HOME/code/housekeeping/memlease.py"
NOTE="bash $HOME/code/housekeeping/note.sh"

$MEMLEASE acquire mbpp-sweep-driver --block exclusive --ensure-gb 42 \
    --pid $$ --note "MBPP pass@5 sweep, 3 temp legs, displaces :8084" || exit 1

rc=0
for T in 1.0 0.8 0.6; do
    .venv/bin/python scripts/difficulty_sweep.py --task code --k 5 \
        --temperature "$T" --max-new-tokens 4096 --batch-prompts 10 \
        --save-texts --required-gb 38 --no-manage-machine || { rc=$?; break; }
done

$MEMLEASE release mbpp-sweep-driver --block exclusive

# Verify the restore actually brought :8084 back (up to 3 min for model load).
for i in $(seq 36); do
    curl -s -m 5 http://127.0.0.1:8084/v1/models >/dev/null && break
    sleep 5
done
if curl -s -m 5 http://127.0.0.1:8084/v1/models >/dev/null; then
    $NOTE "MBPP sweep driver done (rc=$rc): lease released, :8084 restore verified up"
else
    $NOTE "MBPP sweep driver done (rc=$rc): :8084 still DOWN after release — forcing switch-backend default"
    bash "$HOME/code/housekeeping/switch-backend.sh" default
fi
exit "$rc"
