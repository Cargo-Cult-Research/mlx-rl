"""RFCS: step splitting, numeric containment, ratio semantics."""

from mlx_rl.rfcs import rfcs, split_steps, to_number


def test_split_steps_drops_blank_segments():
    assert split_steps("a\n\n\n\nb\n\nc") == ["a", "b", "c"]
    assert split_steps("") == []


def test_first_correct_step_ratio():
    think = "Let me add.\n\n10 + 32 gives 42.\n\nDouble-check: yes 42.\n\nDone."
    assert rfcs(think, "42") == 2 / 4  # answer first appears in step 2 of 4


def test_answer_in_last_step_is_one():
    assert rfcs("hmm\n\nso the result is 7", "7") == 1.0


def test_no_substring_false_positive():
    # 342 must not match answer 42; maximal-munch tokenization prevents it
    assert rfcs("consider 342\n\nanswer is 42", "42") == 1.0


def test_fraction_and_ratio_forms_match():
    assert rfcs(r"half, i.e. \frac{1}{2}\n\nmore", "0.5") is not None
    assert rfcs("the ratio 1/2 appears\n\nmore", "0.5") == 0.5


def test_undefined_cases_return_none():
    assert rfcs("no numbers here\n\nat all", "42") is None  # never appears
    assert rfcs("step", "x+1") is None                      # non-numeric answer
    assert rfcs("", "42") is None                           # empty think


def test_to_number():
    assert to_number("1,006") == 1006
    assert to_number("7/3") is not None
    assert to_number("nope") is None
