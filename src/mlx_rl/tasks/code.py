"""MBPP code task — real, verifiable coding signal (disjoint from SWE-bench).

Each example is a self-contained Python function spec + hidden unit-test asserts.
Reward = 1.0 iff the model's function passes every assert, else 0.0. A fixed
seeded split holds out an eval set that sample() never draws — so the trainer's
evaluate() (which uses eval_sample) never leaks.

SECURITY: by default (sandbox=True) candidate code runs under macOS
sandbox-exec (Seatbelt) with network denied and file writes confined to the
candidate's own temp dir, plus POSIX rlimits (CPU, file size, fds, procs,
data) and a scrubbed environment. Pass sandbox=False (task_kwargs) to run in
a plain subprocess with only the timeout + rlimits — model-generated code
then executes with this process's privileges. The rlimits guard against
resource accidents, not hostile code; use a container/VM for untrusted
prompts or third-party models. See the warning in README.md.

Dataset: sanitized MBPP (427 problems), data/mbpp_sanitized.json — see
data/README.md for provenance and license (CC BY 4.0, Google Research).
"""
from __future__ import annotations

import json
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Example, RewardResult, register

_DATA = Path(__file__).resolve().parents[3] / "data" / "mbpp_sanitized.json"
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_N_EVAL = 80          # held-out for eval; the rest is trainable
_TIMEOUT_S = 8

# Seatbelt profile: allow-default, then deny the two things that matter —
# network, and writes outside the candidate's own temp dir (/dev stays
# writable; CPython opens /dev/null and /dev/dtracehelper on startup).
_SB_PROFILE = (
    '(version 1)'
    '(allow default)'
    '(deny network*)'
    '(deny file-write*)'
    '(allow file-write* (subpath "{tmp}") (subpath "/dev"))'
)


def _limit_resources():
    """preexec_fn for the candidate subprocess. Accident containment, not a
    security boundary: caps CPU (backstops grandchildren the wall-clock kill
    can't reach), file size, fds, and process count (fork bombs). Heap is
    best-effort only — Darwin rejects RLIMIT_DATA/RLIMIT_AS changes outright
    on some releases, so a memory blowup is bounded by the timeout, not by
    an rlimit."""
    resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_S + 2,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (1024,) * 2)
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (4 * 2**30,) * 2)
    except (ValueError, OSError):
        pass


def _resolve_sandbox_exec(sandbox: bool) -> str | None:
    """Path to sandbox-exec, None when sandboxing is off. Raises when the
    sandbox was requested but isn't available — never silently downgrades."""
    if not sandbox:
        return None
    exe = shutil.which("sandbox-exec")
    if not exe:
        raise RuntimeError(
            "code task: sandbox=True but sandbox-exec was not found "
            "(macOS only). Run inside a container/VM instead, or pass "
            "task_kwargs '{\"sandbox\": false}' to execute candidate "
            "code unsandboxed."
        )
    return exe


def sandbox_run(files: dict[str, str], argv: list[str],
                sandbox_exec: str | None, timeout: int, stdin: str | None = None):
    """Run `argv` in a fresh temp dir seeded with `files` (relative paths in
    argv resolve there), under Seatbelt when sandbox_exec is set, always with
    rlimits and a scrubbed env. Returns the CompletedProcess, or None on
    wall-clock timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp).resolve()  # Seatbelt subpath needs the real path
        for name, content in files.items():
            (d / name).write_text(content)
        if sandbox_exec:
            argv = [sandbox_exec, "-p", _SB_PROFILE.format(tmp=d), *argv]
        env = {"PATH": "/usr/bin:/bin", "HOME": str(d),
               "TMPDIR": str(d), "LC_ALL": "C.UTF-8"}
        try:
            return subprocess.run(argv, cwd=d, env=env, input=stdin,
                                  preexec_fn=_limit_resources,
                                  capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def _extract_code(completion: str) -> str:
    """Model output -> candidate source. Drop the think block, take the last
    fenced python block, else the raw post-think text."""
    if "</think>" in completion:
        completion = completion.split("</think>", 1)[1]
    blocks = _CODE_BLOCK.findall(completion)
    return (blocks[-1] if blocks else completion).strip()


@register
class CodeTask:
    name = "code"

    def __init__(self, seed: int = 12345, sandbox: bool = True, **_):
        self._sandbox_exec = _resolve_sandbox_exec(sandbox)
        rows = json.loads(_DATA.read_text())
        rng = random.Random(seed)
        rng.shuffle(rows)
        self._eval = rows[:_N_EVAL]
        self._train = rows[_N_EVAL:]

    def _example(self, row) -> Example:
        tests = "\n".join(row["test_list"][:3])
        prompt = (
            f"{row['prompt'].strip()}\n\n"
            "Write a single self-contained Python function that satisfies the "
            "tests below. Reply with your reasoning, then the function in one "
            "```python code block. Do not include the tests.\n\n"
            f"Tests it must pass:\n{tests}"
        )
        return Example(messages=[{"role": "user", "content": prompt}],
                       meta={"task_id": row["task_id"],
                             "test_list": row["test_list"],
                             "test_imports": row.get("test_imports", [])})

    def sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._train))

    def all_examples(self) -> list[Example]:
        """Every problem exactly once, split-tagged — for offline sweeps
        (base-model difficulty labeling), never for training draws."""
        out = []
        for split, rows in (("train", self._train), ("eval", self._eval)):
            for row in rows:
                ex = self._example(row)
                ex.meta["split"] = split
                out.append(ex)
        return out

    def eval_sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._eval))

    def reward(self, example: Example, completion: str) -> RewardResult:
        code = _extract_code(completion)
        if not code or "def " not in code:
            return RewardResult(0.0, {"correct": 0.0, "code": 1.0, "nopatch": 1.0})
        ok = run_asserts(code, example.meta, self._sandbox_exec)
        return RewardResult(1.0 if ok else 0.0, {"correct": float(ok), "code": 1.0})


def run_asserts(code: str, meta: dict, sandbox_exec: str | None) -> bool:
    """MBPP-shaped grading: candidate + the row's assert list in one script,
    passed iff it runs to the sentinel. Shared with the kodcode task's MBPP
    eval rows."""
    script = "\n".join([
        *meta.get("test_imports", []),
        code, "",
        *meta["test_list"],
        "print('ALL_TESTS_PASSED')",
    ])
    p = sandbox_run({"cand.py": script}, [sys.executable, "cand.py"],
                    sandbox_exec, timeout=_TIMEOUT_S)
    return p is not None and p.returncode == 0 and "ALL_TESTS_PASSED" in p.stdout
