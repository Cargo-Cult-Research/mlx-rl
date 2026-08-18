#!/usr/bin/env python3
# lifecycle: core
"""Launch a multi-hour run so that nothing but the run itself can kill it.

    uv run python scripts/launch_detached.py --log runs/myrun.log -- \
        python -m mlx_rl.train --profile qwen36 --task qa_abstain ...

Prints the pid and a copy-pasteable `tail -f` line, then exits. The child
survives the parent, the terminal, the ssh session, and whatever tooling
launched any of them.

Why this exists rather than a line of shell in each run script: a 200-step
training run was killed at step 17 by the teardown of the wrapper that started
it, and the obvious fix — `setsid` — **does not exist on macOS**. BSD userland
ships no setsid(1) (same family as the missing timeout(1), where the answer is
`gtimeout`). There is no drop-in binary; the call has to come from the process
itself, which is what `start_new_session=True` does here (it is setsid(2), via
posix_spawn). `nohup cmd &` detaches from the terminal but leaves the child in
the launcher's process group, so a group-directed teardown still reaches it.

The three failure modes this closes:
  1. the launching terminal/session goes away          -> new session
  2. the wrapper that launched it is torn down as a group -> new process group
  3. the log is on a pipe nobody is reading anymore    -> log goes to a file
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run a command fully detached; print pid and log path.")
    p.add_argument("--log", required=True, type=Path,
                   help="file to append stdout+stderr to (parents created)")
    p.add_argument("--pidfile", type=Path,
                   help="write the child pid here (default: <log>.pid)")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="the command, after a literal --")
    a = p.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        p.error("no command given (put it after a literal --)")

    a.log.parent.mkdir(parents=True, exist_ok=True)
    pidfile = a.pidfile or a.log.with_suffix(a.log.suffix + ".pid")

    # Append, never truncate: a relaunch must not destroy the log of the run
    # it is replacing.
    with open(a.log, "ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,   # a run that waits on stdin is a hang
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,     # setsid(2) — no macOS setsid(1) needed
            cwd=os.getcwd(),
        )

    pidfile.write_text(f"{proc.pid}\n")
    print(f"pid {proc.pid}  (session leader, survives this shell)")
    print(f"log {a.log}")
    print(f"pid file {pidfile}")
    print()
    print(f"    tail -f {a.log}")
    print()
    print(f"stop it with:  kill {proc.pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
