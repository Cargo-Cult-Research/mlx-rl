#!/usr/bin/env bash
# lifecycle: core
#
# Package the QA/abstention program's artifacts for the internal release.
#
# Why this exists: the adapters, the calibration file and the judge cache all
# live under gitignored paths (`runs/`, `~/models/adapters`). They existed only
# on one disk, so a collaborator reproducing the result had to re-derive every
# input from scratch — a day of judge traffic to rebuild a cache we already had.
# This script turns that state into a downloadable release asset.
#
# Usage:  bash scripts/bundle_artifacts.sh [OUTDIR]
# Then:   gh release create qa-glove-artifacts-<date> --repo <your-org>/<your-repo> ...
#
# Public counterpart: only the C-200 + prompt pair ships externally (Hugging
# Face). Everything here is internal — full arm family, raw rollouts, caches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER_STORE="${ADAPTER_STORE:-$HOME/models/adapters}"
OUT="${1:-$REPO_ROOT/dist}"
STAGE="$OUT/stage"

# The glove program's trained arms, in the order the results doc presents them.
ADAPTERS=(
  qa-abstain-20260726        # tag-frame flagship (pre-glove)
  qa-chatmix-20260730        # naive frame mixture, no prompt — the flat control
  qa-gloveB-20260730         # arm B: prompt + tag-tuned bands
  qa-gloveA-20260731         # arm A ckpt 200 — the transfer claim's checkpoint
  qa-gloveA-160-20260731     # arm A ckpt 160 — eval-best
  qa-gloveC-160-20260731     # arm C ckpt 160
  qa-gloveC-200-20260731     # arm C ckpt 200 — THE DELIVERABLE (ships with GLOVE.txt)
  qa-gloveC-seed1-20260801   # arm C, second seed — the replication
  qa-glovec1-20260801        # c=1 penalty-sweep point
)

# Training runs whose config/metrics/rollouts back those adapters.
TRAIN_RUNS=(
  qa-gloveA-20260731 qa-gloveB-20260730 qa-gloveC-20260731
  qa-gloveC-seed1-20260801 qa-glovec1-20260801 qa-pilot-20260726
)

rm -rf "$STAGE"
mkdir -p "$STAGE"/{adapters,runs,calib,judge,probes}

echo "==> adapters"
for a in "${ADAPTERS[@]}"; do
  if [[ ! -d "$ADAPTER_STORE/$a" ]]; then
    echo "MISSING adapter: $ADAPTER_STORE/$a" >&2; exit 1
  fi
  cp -R "$ADAPTER_STORE/$a" "$STAGE/adapters/"
  printf '    %-28s %s\n' "$a" "$(head -1 "$ADAPTER_STORE/$a/MANIFEST.md" 2>/dev/null)"
done

echo "==> training runs (config + metrics + rollouts; checkpoints excluded, promoted copies are in adapters/)"
for r in "${TRAIN_RUNS[@]}"; do
  src="$REPO_ROOT/runs/$r"
  [[ -d "$src" ]] || { echo "MISSING run: $src" >&2; exit 1; }
  mkdir -p "$STAGE/runs/$r"
  cp "$src"/config*.json "$src"/metrics.jsonl "$src"/samples.jsonl "$STAGE/runs/$r/" 2>/dev/null || true
done
# The run scripts are already in git, but ship them alongside so the bundle
# stands alone.
cp "$REPO_ROOT"/runs/qa-glove*_run.sh "$REPO_ROOT"/runs/qa-gloveA-retry_run.sh \
   "$REPO_ROOT"/runs/papers_recall_run.sh "$REPO_ROOT"/runs/papers_probe_run.sh \
   "$STAGE/runs/" 2>/dev/null || true

echo "==> calibration probe (the band assignment every run trained against)"
cp -R "$REPO_ROOT/runs/qa-calib-20260724" "$STAGE/calib/"

echo "==> judge cache (Opus commitment verdicts — replaces live judge traffic on a re-run)"
cp "$REPO_ROOT"/runs/judge/qa-abstain-cache.jsonl \
   "$REPO_ROOT"/runs/judge/qa-abstain-cache.calls.jsonl "$STAGE/judge/"

echo "==> probe outputs (the measurements in the results tables)"
for d in "$REPO_ROOT"/runs/papers-recall-* "$REPO_ROOT"/runs/qa-chat-*; do
  [[ -d "$d" ]] && cp -R "$d" "$STAGE/probes/"
done

cp "$REPO_ROOT/docs/artifacts.md" "$STAGE/README.md"

echo "==> tarballs"
tar -C "$STAGE" -czf "$OUT/qa-glove-adapters.tar.gz" adapters README.md
tar -C "$STAGE" -czf "$OUT/qa-glove-repro-inputs.tar.gz" runs calib judge probes README.md
( cd "$OUT" && shasum -a 256 qa-glove-adapters.tar.gz qa-glove-repro-inputs.tar.gz > SHA256SUMS )

rm -rf "$STAGE"
ls -lh "$OUT"
echo
echo "done. verify with:  cd $OUT && shasum -a 256 -c SHA256SUMS"
