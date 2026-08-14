"""DeepCoder code task — competition-style problems, stdin/stdout judged.

Data: data/deepcoder/{train,test}.jsonl.gz, produced by
scripts/fetch_deepcoder.py from agentica-org/DeepCoder-Preview-Dataset
(TACO-verified + SYNTHETIC-1 + pre-cutoff LiveCodeBench, curated for RL:
every problem's tests verified against a reference solution). v1 is
stdin/stdout problems only — one unambiguous judge.

Reward = 1.0 iff a single emitted Python program passes every stored test
case (<=16, evenly thinned at fetch time; first failure short-circuits),
else 0.0. Output comparison: per-line rstrip, trailing blank lines dropped,
exact match. Problems whose canonical answer allows multiple formats will
under-credit — visible as an n_pass=0 mass in difficulty sweeps, to be
root-caused there rather than papered over with a fuzzy judge.

⚠️ SECURITY: same as tasks/code.py — candidate code runs in a plain
subprocess with a timeout, NOT sandboxed.

Curriculum: pass labels_file= (a difficulty_sweep JSONL for this task) plus
min_pass=/max_pass= to restrict training draws to a difficulty band;
eval draws are never filtered. Step the band up/down between runs as the
policy improves — the labels are measured once, reused forever.
"""
from __future__ import annotations

import gzip
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Example, RewardResult, register

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "deepcoder"
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_CASE_TIMEOUT_S = 10
_PROMPT_SUFFIX = (
    "\n\nRead input from standard input and write the answer to standard "
    "output. Reply with your reasoning, then a complete Python program in "
    "one ```python code block."
)


def _extract_code(completion: str) -> str:
    if "</think>" in completion:
        completion = completion.split("</think>", 1)[1]
    blocks = _CODE_BLOCK.findall(completion)
    return (blocks[-1] if blocks else completion).strip()


def _norm_out(s: str) -> str:
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _load(split: str) -> list[dict]:
    path = _DATA_DIR / f"{split}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/fetch_deepcoder.py first")
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


@register
class DeepCoderTask:
    name = "deepcoder"

    def __init__(self, labels_file: str | None = None,
                 min_pass: int = 0, max_pass: int = 10**9, **_):
        self._train = _load("train")
        self._eval = _load("test")
        if labels_file:
            labels = {}
            for line in Path(labels_file).expanduser().read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                labels[row["task_id"]] = row["n_pass"]
            before = len(self._train)
            self._train = [
                r for r in self._train
                if r["id"] in labels and min_pass <= labels[r["id"]] <= max_pass
            ]
            if not self._train:
                raise ValueError(
                    f"difficulty band [{min_pass},{max_pass}] from "
                    f"{labels_file} leaves no training problems")
            print(f"deepcoder: difficulty band [{min_pass},{max_pass}] -> "
                  f"{len(self._train)}/{before} training problems", flush=True)

    def _example(self, row: dict) -> Example:
        prompt = row["problem"].rstrip() + _PROMPT_SUFFIX
        if row.get("starter_code"):
            prompt += ("\n\nYou may start from this code:\n```python\n"
                       f"{row['starter_code']}\n```")
        return Example(messages=[{"role": "user", "content": prompt}],
                       meta={"task_id": row["id"], "source": row["source"],
                             "tests": row["tests"]})

    def sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._train))

    def eval_sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._eval))

    def all_examples(self) -> list[Example]:
        out = []
        for split, rows in (("train", self._train), ("eval", self._eval)):
            for row in rows:
                ex = self._example(row)
                ex.meta["split"] = split
                out.append(ex)
        return out

    def reward(self, example: Example, completion: str) -> RewardResult:
        code = _extract_code(completion)
        if not code:
            return RewardResult(0.0, {"correct": 0.0, "code": 1.0, "nopatch": 1.0})
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "cand.py"
            f.write_text(code)
            for case in example.meta["tests"]:
                try:
                    p = subprocess.run(
                        [sys.executable, str(f)], cwd=d,
                        input=case["input"], capture_output=True,
                        text=True, timeout=_CASE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    return RewardResult(0.0, {"correct": 0.0, "code": 1.0,
                                              "timeout": 1.0})
                if p.returncode != 0 or _norm_out(p.stdout) != _norm_out(case["output"]):
                    return RewardResult(0.0, {"correct": 0.0, "code": 1.0})
        return RewardResult(1.0, {"correct": 1.0, "code": 1.0})
