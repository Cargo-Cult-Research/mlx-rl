#!/usr/bin/env bash
# Reapply the vendored mlx-lm patches in patches/ to the active venv.
#
# The patched files live in .venv/ (build output, not source control), so any
# `uv sync` / venv rebuild silently reverts them. Run this after either.
# Idempotent: already-applied patches are detected and skipped.
#
#   ./scripts/apply_patches.sh            # apply to ./.venv
#   VENV=/path/to/venv ./scripts/apply_patches.sh
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
WANT="0.31.3"
HAVE="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("mlx-lm"))')"

if [ "$HAVE" != "$WANT" ]; then
    echo "REFUSING: patches target mlx-lm $WANT but $HAVE is installed." >&2
    echo "Re-verify each patch against the new version before forcing." >&2
    exit 1
fi

rc=0
for p in patches/mlx-lm-$WANT-*.patch; do
    name="$(basename "$p")"
    if patch -p1 -d "$SP" -R --dry-run -s -f <"$p" >/dev/null 2>&1; then
        echo "already applied: $name"
    elif patch -p1 -d "$SP" --dry-run -s -f <"$p" >/dev/null 2>&1; then
        patch -p1 -d "$SP" -s <"$p"
        echo "APPLIED:         $name"
    else
        echo "FAILED (does not apply cleanly): $name" >&2
        rc=1
    fi
done
exit $rc
