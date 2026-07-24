"""Math task — the paper's actual training domain, from a public corpus.

Dataset: agentica-org/DeepScaleR-Preview-Dataset (MIT, 40,315 competition math
problems — AMC/AIME/Olympiad/MATH-train provenance; the corpus DeepScaleR, one
of the SAGE paper's subject models, was RL-trained on). Fetched via the HF
cache on first use, filtered to numerically verifiable answers (~25k rows:
ints, decimals, fractions) so grading needs no CAS. A fixed seeded split holds
out an eval set that sample() never draws.

Reward: 1.0 iff the last \boxed{...} in the visible reply matches the
reference numerically, else 0.0 (no format reward — the truncation-hack
lesson from arithmetic).
"""
from __future__ import annotations

import json
import random
import re
from fractions import Fraction

from .base import Example, RewardResult, register

_REPO = "agentica-org/DeepScaleR-Preview-Dataset"
_FILE = "deepscaler.json"
_N_EVAL = 200
# int / decimal / fraction, optional commas-as-thousands, optional $…$ wrap
_NUMERIC = re.compile(r"-?[\d,]+(?:\.\d+)?(?:\s*/\s*-?\d+)?")


def _to_number(s: str) -> Fraction | None:
    """Normalize an answer string to an exact Fraction, or None if non-numeric."""
    s = s.strip().strip("$").replace("\\!", "").replace("\\,", "").strip()
    m = _NUMERIC.fullmatch(s)
    if not m:
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        if "/" in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _last_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...}, with balanced-brace parsing."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out.append(c)
        i += 1
    return "".join(out) if not depth else None


def _boxed_number(text: str) -> Fraction | None:
    """Last boxed value as a number; falls back to \\frac{a}{b} inside the box."""
    box = _last_boxed(text)
    if box is None:
        return None
    n = _to_number(box)
    if n is not None:
        return n
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", box.strip())
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    return None


@register
class MathTask:
    name = "math"

    def __init__(self, seed: int = 12345, **_):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_REPO, _FILE, repo_type="dataset")
        rows = json.loads(open(path).read())
        rows = [
            {"problem": r["problem"], "answer": str(r["answer"]).strip()}
            for r in rows
            if _to_number(str(r["answer"])) is not None
        ]
        rng = random.Random(seed)
        rng.shuffle(rows)
        self._eval = rows[:_N_EVAL]
        self._train = rows[_N_EVAL:]

    def _example(self, row) -> Example:
        prompt = (
            f"{row['problem'].strip()}\n\n"
            "Reason step by step, then give the final answer as a single "
            "number inside \\boxed{}."
        )
        return Example(
            messages=[{"role": "user", "content": prompt}],
            meta={"answer": row["answer"]},
        )

    def sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._train))

    def eval_sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._eval))

    def reward(self, example: Example, completion: str) -> RewardResult:
        got = _boxed_number(completion)
        want = _to_number(example.meta["answer"])
        if got is None:
            return RewardResult(0.0, {"format": 0.0, "correct": 0.0})
        correct = want is not None and got == want
        return RewardResult(
            1.0 if correct else 0.0, {"format": 1.0, "correct": float(correct)}
        )
