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

    Two detectors, because swap VOLUME and swap THRASHING are different
    things. macOS allocates swap in 1 GB files on demand and will happily add
    a few while tens of GB of RAM sit free — that costs nothing, but a
    level-only guard reads it as danger and kills healthy runs (observed: a
    150-step run died at step 69 on +3.4 GB drift with 68 GB RAM available).
    What actually turns a 6 s step into 40 minutes is sustained paging, so the
    primary detector is the page-in/page-out RATE; the level check stays as a
    slower backstop with a wider default margin.

    The rate detector needs a second opinion too (2026-08-17). Paging traffic
    is not by itself proof of trouble: on a nearly-full swap file macOS
    produces bursts of ~270 MB/s that the run never feels, and the rate
    detector killed a correctly-configured 200-step run on one. The reason we
    care about paging at all is that it destroys step times, so step times are
    the ground truth — feed them in with `note_step()` and a paging burst can
    only abort the run if the run has ALSO slowed to `slow_factor`x its own
    established pace (or is currently overdue by that much).

    The gate deliberately does not apply before a baseline exists. Thrashing
    during model load or the first backward is the case where paging is the
    only signal available, and it is a real one: an over-sized LoRA
    configuration that thrashed at its first backward is the rate detector's
    one true positive so far. Withheld kills are logged, never silent.
    """

    def __init__(
        self,
        margin_gb: float = 8.0,
        interval_s: float = 3.0,
        abort_marker: Path | None = None,
        rate_mb_s: float = 200.0,
        rate_samples: int = 3,
        slow_factor: float = 3.0,
        baseline_steps: int = 5,
    ):
        self.margin_gb = margin_gb
        self.interval_s = interval_s
        self.abort_marker = abort_marker
        self.rate_mb_s = rate_mb_s
        self.rate_samples = rate_samples
        self.slow_factor = slow_factor
        self.baseline_steps = baseline_steps
        self._step_lock = threading.Lock()
        self._step_dts: list[float] = []   # durations seen while establishing pace
        self._step_baseline_s: float | None = None
        self._last_step_dt: float | None = None
        self._last_step_end: float | None = None  # monotonic, for the overdue check
        self.baseline_gb = 0.0
        self._hot = 0  # consecutive over-threshold samples
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SwapGuard":
        if self.margin_gb <= 0 and self.rate_mb_s <= 0:  # disabled
            return self
        self.baseline_gb = swap_used_gb()
        self._thread = threading.Thread(target=self._run, name="swap-guard", daemon=True)
        self._thread.start()
        print(
            f"[swap-guard] armed: baseline {self.baseline_gb:.1f} GB, "
            f"abort if +{self.margin_gb:.1f} GB or paging >= "
            f"{self.rate_mb_s:.0f} MB/s for {self.rate_samples} samples "
            f"(every {self.interval_s:.0f}s)",
            flush=True,
        )
        return self

    def stop(self) -> None:
        self._stop.set()

    def note_step(self, dt_s: float, now: float | None = None) -> None:
        """Report a completed training step, so the rate detector can tell
        'the machine is paging' from 'the run is being hurt by paging'.

        The first `baseline_steps` durations set the run's healthy pace. Step 0
        carries compilation and first-touch costs, so it is dropped rather than
        allowed to inflate the baseline into uselessness.
        """
        now = time.monotonic() if now is None else now
        with self._step_lock:
            self._last_step_dt = dt_s
            self._last_step_end = now
            if self._step_baseline_s is not None:
                return
            self._step_dts.append(dt_s)
            if len(self._step_dts) > self.baseline_steps:
                usable = sorted(self._step_dts[1:])  # drop the warm-up step
                self._step_baseline_s = usable[len(usable) // 2]
                print(f"[swap-guard] step pace established: "
                      f"{self._step_baseline_s:.1f}s/step — paging bursts now "
                      f"need a {self.slow_factor:.0f}x slowdown to abort",
                      flush=True)

    def _run_is_hurting(self, now: float | None = None) -> tuple[bool, str]:
        """Is the run actually slowed? -> (verdict, one-line evidence).

        Unknown counts as hurting: with no pace to compare against, paging is
        the only evidence there is, and load-time thrashing is real.
        """
        now = time.monotonic() if now is None else now
        with self._step_lock:
            base, last_dt, last_end = (
                self._step_baseline_s, self._last_step_dt, self._last_step_end)
        if base is None:
            return True, "no step pace established yet (load or first steps)"
        threshold = base * self.slow_factor
        if last_dt is not None and last_dt >= threshold:
            return True, (f"last step {last_dt:.1f}s vs {base:.1f}s baseline "
                          f"(>= {self.slow_factor:.0f}x)")
        overdue = now - last_end if last_end is not None else 0.0
        if overdue >= threshold:
            return True, (f"current step overdue {overdue:.0f}s vs "
                          f"{base:.1f}s baseline")
        return False, (f"steps healthy: last {last_dt:.1f}s, "
                       f"{overdue:.0f}s into the current one, "
                       f"baseline {base:.1f}s")

    def classify(self, moved_bytes: float, dt_s: float, used_gb: float,
                 now: float | None = None):
        """One sample -> ('rate', mb_s) | ('level', grown_gb) | None.

        Factored out of the sampling loop so the policy can be tested without
        spawning a thread: a stray daemon thread outliving a test reads REAL
        swap and can hard-exit the test runner.
        """
        rate = max(moved_bytes, 0.0) / max(dt_s, 1e-6) / 1e6
        if self.rate_mb_s > 0 and rate >= self.rate_mb_s:
            self._hot += 1
            if self._hot >= self.rate_samples:
                hurting, why = self._run_is_hurting(now)
                if hurting:
                    return "rate", rate
                # Paging, but the run is fine. Say so — a guard that withholds
                # a kill silently is indistinguishable from one that is broken.
                print(f"[swap-guard] paging {rate:.0f} MB/s for "
                      f"{self._hot} samples, NOT aborting — {why}", flush=True)
                self._hot = 0
        else:
            self._hot = 0
        if self.margin_gb > 0:
            grown = used_gb - self.baseline_gb
            if grown > self.margin_gb:
                return "level", grown
        return None

    def _run(self) -> None:
        prev, prev_t = psutil.swap_memory(), time.monotonic()
        while not self._stop.wait(self.interval_s):
            cur, now = psutil.swap_memory(), time.monotonic()
            # Counters are cumulative; clamp in case they wrap or reset.
            moved = max(cur.sin - prev.sin, 0) + max(cur.sout - prev.sout, 0)
            verdict = self.classify(moved, now - prev_t, cur.used / 1e9)
            prev, prev_t = cur, now
            if verdict is None:
                continue
            kind, value = verdict
            if kind == "rate":
                self._abort_rate(value, self._hot)
            else:
                self._abort(value)

    def _abort_rate(self, rate_mb_s: float, samples: int) -> None:
        msg = (
            "\n" + "=" * 72 + "\n"
            "[swap-guard] ABORT — sustained paging (thrashing).\n"
            f"  {rate_mb_s:.0f} MB/s swapped for {samples} consecutive "
            f"{self.interval_s:.0f}s samples "
            f"(threshold {self.rate_mb_s:.0f} MB/s).\n"
            "  This is the failure mode that makes a 6 s step take 40 min.\n"
            "  FIX: --grad-checkpoint, lower --max-new-tokens/--batch-prompts,\n"
            "  or free resident memory (stop other big jobs on the box).\n"
            + "=" * 72 + "\n"
        )
        self._die(msg, f"swap-guard abort: paging {rate_mb_s:.0f} MB/s for "
                       f"{samples} samples (threshold {self.rate_mb_s:.0f})")

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
        self._die(
            msg,
            f"swap-guard abort: swap {now:.1f} GB, +{grown_gb:.1f} GB over "
            f"baseline {self.baseline_gb:.1f} GB (margin {self.margin_gb:.1f})",
        )

    def _die(self, msg: str, marker: str | None = None) -> None:
        sys.stderr.write(msg)
        sys.stderr.flush()
        write_abort_marker(self.abort_marker, marker or msg.strip())
        os._exit(SWAP_ABORT_EXIT)
