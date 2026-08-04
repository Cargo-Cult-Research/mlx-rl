"""Optional integration with an external memory-lease command.

On a single box a big training run may need most of the memory, and you may
have another memory-hungry process (e.g. a local inference server) you want to
pause for the duration and restore afterwards. If you have such a coordinator,
point ``MLX_RL_MEMLEASE_CMD`` at it and mlx-rl will call it to acquire a lease
before loading the model and release it when the run ends — even on crash, if
your command is PID-aware.

When ``MLX_RL_MEMLEASE_CMD`` is unset, a machine-local fallback is probed:
if ``~/code/housekeeping/memlease.py`` exists it is used, so on a box that
has the coordinator every launcher is managed by default — an unmanaged run
next to a 22 GB inference server is exactly the co-residency swap abort of
2026-07-29, and it happened because the one run script that forgot the
export silently opted out. Set the env var (or pass --no-manage-machine)
to override; on machines without the coordinator the run simply proceeds
unmanaged and the in-process memory guard (memory.py) is still the safety
net that refuses runs which do not fit.

The command is invoked as::

    <cmd> acquire <holder> --pid <pid> --ensure-gb <N> --wait <S> --note <text>
    <cmd> release <holder>

where ``<cmd>`` is the (shell-split) value of ``MLX_RL_MEMLEASE_CMD``. A zero
exit from ``acquire`` means the lease was granted; non-zero means it is held by
someone else. Anything your command prints is passed straight through so its
progress is visible.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

def _default_cmd() -> list[str]:
    path = os.path.expanduser("~/code/housekeeping/memlease.py")
    if os.path.exists(path):
        return [sys.executable, path]
    return []


_env = os.environ.get("MLX_RL_MEMLEASE_CMD")
MEMLEASE_CMD = shlex.split(_env) if _env is not None else _default_cmd()


class MachineBusyError(RuntimeError):
    pass


def _enabled() -> bool:
    return bool(MEMLEASE_CMD)


def _run(args: list[str]) -> int:
    return subprocess.run(
        [*MEMLEASE_CMD, *args],
        stdout=None,  # pass through: acquire/release messages must be visible
        stderr=None,
    ).returncode


def holder_name() -> str:
    return f"mlx-rl:{os.getpid()}"


def acquire(required_gb: float, wait_s: float = 0, note: str = "",
            block: str = "exclusive") -> str | None:
    """Acquire the memory lease and make room (frees other jobs if the command does).

    Returns the holder name (pass it to release), or None when no lease command
    is configured (the in-process memory guard still protects the run). Raises
    MachineBusyError if the lease is held by someone else.

    ensure-gb is inflated to required/SAFETY_FRACTION: assert_fits() later
    demands required <= available * SAFETY_FRACTION, so the lease must make
    that much room or the guard refuses a run the lease thought it had
    cleared (a run died exactly this way: the lease saw 65.7 >= 65.6 needed
    and left the room short; the guard then required 65.6/0.9).
    """
    if not _enabled():
        print(
            "machine: no memory-lease command configured (MLX_RL_MEMLEASE_CMD) "
            "— running unmanaged",
            file=sys.stderr,
        )
        return None
    from .memory import SAFETY_FRACTION

    holder = holder_name()
    rc = _run(
        [
            "acquire",
            holder,
            "--block",
            block,
            "--pid",
            str(os.getpid()),
            "--ensure-gb",
            str(round(required_gb / SAFETY_FRACTION, 1)),
            "--wait",
            str(wait_s),
            "--note",
            note,
        ]
    )
    if rc != 0:
        raise MachineBusyError(
            "the memory lease is held by another job; retry with --lease-wait "
            "or after it releases"
        )
    return holder


def release(holder: str | None) -> None:
    """Release the lease; your command is responsible for restoring anything it paused."""
    if holder is None or not _enabled():
        return
    _run(["release", holder])
