"""MBPP code task — real, verifiable coding signal (disjoint from SWE-bench).

Each example is a self-contained Python function spec + hidden unit-test asserts.
Reward = 1.0 iff the model's function passes every assert, else 0.0. A fixed
seeded split holds out an eval set that sample() never draws — so the trainer's
evaluate() (which uses eval_sample) never leaks.

⚠️ SECURITY: candidate code runs in a plain subprocess with a timeout — it is
NOT sandboxed. Model-generated code executes with this process's privileges
(filesystem, network). Run this task inside a container/VM if that matters
to you. See the warning in README.md.

Dataset: sanitized MBPP (427 problems), data/mbpp_sanitized.json — see
data/README.md for provenance and license (CC BY 4.0, Google Research).
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Example, RewardResult, register

_DATA = Path(__file__).resolve().parents[3] / "data" / "mbpp_sanitized.json"
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_N_EVAL = 80          # held-out for eval; the rest is trainable
_TIMEOUT_S = 8


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

    def __init__(self, seed: int = 12345, **_):
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

    def eval_sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._eval))

    def reward(self, example: Example, completion: str) -> RewardResult:
        code = _extract_code(completion)
        if not code or "def " not in code:
            return RewardResult(0.0, {"correct": 0.0, "code": 1.0, "nopatch": 1.0})
        script = "\n".join([
            *example.meta.get("test_imports", []),
            code, "",
            *example.meta["test_list"],
            "print('ALL_TESTS_PASSED')",
        ])
        try:
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "cand.py"
                f.write_text(script)
                p = subprocess.run([sys.executable, str(f)], cwd=d,
                                   capture_output=True, text=True,
                                   timeout=_TIMEOUT_S)
            ok = p.returncode == 0 and "ALL_TESTS_PASSED" in p.stdout
        except subprocess.TimeoutExpired:
            ok = False
        return RewardResult(1.0 if ok else 0.0,
                            {"correct": float(ok), "code": 1.0})
