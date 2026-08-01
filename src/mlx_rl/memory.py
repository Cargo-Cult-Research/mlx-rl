"""Memory guard — the in-process backstop.

Machine-level room-making can go through an optional memory-lease command
(see machine.py); when one is configured the trainer acquires it and it
frees/restores whatever else you run. This guard is the last line of defense
behind that — if a run STILL doesn't fit (no lease configured, or something
else is eating memory), it refuses to load rather than risk a kernel-panic
scenario (two big models on 96 GB).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

SAFETY_FRACTION = 0.9

# Exit code the swap guard uses when it hard-aborts a thrashing run. Distinct
# from 1 so wrappers/logs can tell "we killed it for swapping" from a crash.
SWAP_ABORT_EXIT = 137


class MemoryGuardError(RuntimeError):
    pass


def model_disk_gb(model_path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors")) / 1e9


def estimate_run_gb(weights_gb: float, headroom_gb: float = 4.0) -> float:
    # Training multiplier measured, not guessed: the qwen36 rank-16 probe
    # (22 GB weights, micro_batch 2, ~160-token seqs) peaked at 63 GB —
    # ~2.9x weights once dequant transients, autodiff graph, and Adam state
    # are in. Generation alone is ~1.35x; we guard for the training peak.
    return weights_gb * 2.9 + headroom_gb


def available_gb() -> float:
    """psutil available + macOS speculative pages.

    On macOS, psutil's `available` excludes *speculative* pages — read-ahead
    file cache the kernel reclaims instantly under allocation pressure. A
    bulk file copy (e.g. the 2026-07-31 rsync disk migration) parks tens of
    GB there and made the guard refuse a run that genuinely fit, twice.
    Speculative pages are as reclaimable as free ones for our purposes;
    count them. (Purgeable is already inside psutil's available.)"""
    avail = psutil.virtual_memory().available / 1e9
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 timeout=10).stdout
            page = 16384
            m = re.search(r"page size of (\d+) bytes", out)
            if m:
                page = int(m.group(1))
            m = re.search(r"Pages speculative:\s+(\d+)", out)
            if m:
                avail += int(m.group(1)) * page / 1e9
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass  # fall back to the conservative number
    return avail


def write_abort_marker(out_dir_or_path: Path | None, reason: str) -> None:
    """Drop runs/<name>/ABORTED so death is visible in-band (the dashboard
    raises an error on it). One swap-guard kill was only discoverable by
    noticing the run had silently stopped progressing — not loud enough."""
    if out_dir_or_path is None:
        return
    p = Path(out_dir_or_path)
    marker = p if p.name == "ABORTED" else p / "ABORTED"
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {reason}\n"
        )
    except OSError:
        pass  # never let the death rattle raise


def assert_fits(required_gb: float, available: float | None = None) -> None:
    avail = available_gb() if available is None else available
    budget = avail * SAFETY_FRACTION
    if required_gb > budget:
        raise MemoryGuardError(
            f"Run needs ~{required_gb:.1f} GB but only {avail:.1f} GB is available "
            f"({budget:.1f} GB after safety margin) — even after the host memory "
            "lease made room (or the run opted out of it). Check what else is "
            "resident and free some, then retry."
        )


def swap_used_gb() -> float:
    return psutil.swap_memory().used / 1e9


class SwapGuard:
    """Background swap watchdog: fail LOUD and FAST instead of thrashing.

    On a 96 GB box a long-sequence MoE backward can push activation memory past
    physical RAM; macOS then pages to SSD and a ~6 s step silently becomes
    40 min–2 h (measured). That slowness is worse than a crash — you
    can't tell it from a hang. This samples system swap on a daemon thread and,
    if swap grows more than `margin_gb` above the baseline captured at start(),
    prints a banner and hard-exits (`os._exit`, so the thrashing mx.eval can't
    swallow the signal). A PID-aware external lease command (see machine.py)
    can still detect the dead holder and restore whatever it displaced.
    """

    def __init__(
        self,
        margin_gb: float = 3.0,
        interval_s: float = 3.0,
        abort_marker: Path | None = None,
    ):
        self.margin_gb = margin_gb
        self.interval_s = interval_s
        self.abort_marker = abort_marker
        self.baseline_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SwapGuard":
        if self.margin_gb <= 0:  # disabled
            return self
        self.baseline_gb = swap_used_gb()
        self._thread = threading.Thread(target=self._run, name="swap-guard", daemon=True)
        self._thread.start()
        print(
            f"[swap-guard] armed: baseline {self.baseline_gb:.1f} GB, "
            f"abort if +{self.margin_gb:.1f} GB (every {self.interval_s:.0f}s)",
            flush=True,
        )
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            grown = swap_used_gb() - self.baseline_gb
            if grown > self.margin_gb:
                self._abort(grown)

    def _abort(self, grown_gb: float) -> None:
        now = swap_used_gb()
        msg = (
            "\n" + "=" * 72 + "\n"
            "[swap-guard] ABORT — the run started SWAPPING.\n"
            f"  swap now {now:.1f} GB, up {grown_gb:.1f} GB from baseline "
            f"{self.baseline_gb:.1f} GB (margin {self.margin_gb:.1f} GB).\n"
            "  A backward pass spilled past physical RAM; paging to SSD would\n"
            "  make this run 10-100x slower. Failing loud instead of crawling.\n"
            "  FIX: --grad-checkpoint (recompute cuts a 1536-token backward\n"
            "  83 -> 37 GiB), lower --max-new-tokens, or free resident memory\n"
            "  (free resident memory / stop other big jobs on the box).\n"
            + "=" * 72 + "\n"
        )
        sys.stderr.write(msg)
        sys.stderr.flush()
        write_abort_marker(
            self.abort_marker,
            f"swap-guard abort: swap {now:.1f} GB, +{grown_gb:.1f} GB over "
            f"baseline {self.baseline_gb:.1f} GB (margin {self.margin_gb:.1f})",
        )
        os._exit(SWAP_ABORT_EXIT)
