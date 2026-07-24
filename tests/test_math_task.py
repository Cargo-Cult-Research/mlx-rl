"""Math task: boxed-answer extraction + numeric grading (no dataset download —
pure-function tests plus a registry smoke test that skips offline)."""

from fractions import Fraction

import pytest

from mlx_rl.tasks.math import _boxed_number, _last_boxed, _to_number


def test_last_boxed_balanced_braces():
    assert _last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert _last_boxed(r"\boxed{3} then \boxed{7}") == "7"
    assert _last_boxed(r"\boxed{unclosed") is None
    assert _last_boxed("no box") is None


@pytest.mark.parametrize(
    "s,want",
    [
        ("42", Fraction(42)),
        ("-13.5", Fraction("-13.5")),
        ("1,006", Fraction(1006)),
        ("7/3", Fraction(7, 3)),
        ("$5$", Fraction(5)),
        ("x+1", None),
        ("", None),
        ("1/0", None),
    ],
)
def test_to_number(s, want):
    assert _to_number(s) == want


def test_boxed_number_frac_fallback_and_numeric_eq():
    assert _boxed_number(r"\boxed{\dfrac{7}{3}}") == Fraction(7, 3)
    assert _boxed_number(r"\boxed{2.0}") == _to_number("2")
    assert _boxed_number(r"\boxed{x+1}") is None


def test_task_reward_end_to_end():
    from mlx_rl.tasks.base import Example
    from mlx_rl.tasks.math import MathTask

    task = MathTask.__new__(MathTask)  # skip __init__ (no download in tests)
    ex = Example(messages=[], meta={"answer": "204"})
    assert task.reward(ex, r"... so \boxed{204}").total == 1.0
    assert task.reward(ex, r"... so \boxed{205}").total == 0.0
    assert task.reward(ex, "no box").total == 0.0
    assert task.reward(ex, "no box").parts["format"] == 0.0
