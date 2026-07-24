"""RFCS — Ratio of First-Correct-Step (SAGE paper, arXiv 2602.08354).

RFCS = (1-based index of the first reasoning step whose text contains the
reference answer) / (total reasoning steps), computed over the think block of
a completion that was graded CORRECT. A step is a `\n\n`-delimited segment,
matching the paper's step definition. RFCS ≪ 1 means the model reached the
answer early and kept thinking; the paper's claim is that SAGE-RL drives
RFCS toward 1 (stop right after the answer appears).

Only numeric answers are matched (arithmetic / math tasks) — code diffs have
no well-defined "the answer appears in step k" notion, so those return None
and are excluded from the aggregate rather than guessed at.
"""
from __future__ import annotations

import re
from fractions import Fraction

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_RATIO = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_FRAC = re.compile(r"\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}")


def to_number(s: str) -> Fraction | None:
    """Answer string -> exact Fraction, or None if non-numeric."""
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    try:
        if "/" in s:
            num, den = s.split("/")
            return Fraction(int(num), int(den))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def split_steps(think: str) -> list[str]:
    return [s for s in think.split("\n\n") if s.strip()]


def _step_contains(step: str, want: Fraction) -> bool:
    for a, b in _FRAC.findall(step) + _RATIO.findall(step):
        try:
            if Fraction(int(a), int(b)) == want:
                return True
        except ZeroDivisionError:
            pass
    for tok in _NUM.findall(step):
        try:
            if Fraction(tok.replace(",", "")) == want:
                return True
        except ValueError:
            pass
    return False


def rfcs(think: str, answer: str) -> float | None:
    """RFCS for one completion's think block, or None when undefined
    (non-numeric answer, empty think, or the answer never appears in the
    chain — the model can legitimately first state it in the visible reply)."""
    want = to_number(answer)
    if want is None:
        return None
    steps = split_steps(think)
    if not steps:
        return None
    for i, step in enumerate(steps):
        if _step_contains(step, want):
            return (i + 1) / len(steps)
    return None
