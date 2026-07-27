"""qa_abstain: reply parsing, normalized grading, reward structure, and
curriculum band routing — pure-function tests, no dataset download."""

import random

import pytest

from mlx_rl.tasks.base import Example
from mlx_rl.tasks.qa_abstain import (
    QAAbstainTask,
    _band,
    grade,
    normalize,
    parse_reply,
)


@pytest.mark.parametrize(
    "s,want",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  Harry  S. Truman ", "harry s truman"),
        ("An apple", "apple"),
        ("YORK, Yorkshire", "york yorkshire"),
    ],
)
def test_normalize(s, want):
    assert normalize(s) == want


def test_parse_reply_last_tag_wins():
    assert parse_reply("I think <answer>Paris</answer>") == ("answer", "Paris")
    assert parse_reply("<answer>Rome</answer> no wait <abstain/>") == ("abstain", None)
    assert parse_reply("<abstain/> actually <answer>Rome</answer>") == ("answer", "Rome")


def test_parse_reply_abstain_variants_and_malformed():
    assert parse_reply("<abstain/>")[0] == "abstain"
    assert parse_reply("<abstain />")[0] == "abstain"
    assert parse_reply("<ABSTAIN/>")[0] == "abstain"
    assert parse_reply("no tags at all")[0] == "malformed"
    assert parse_reply("<answer>   </answer>")[0] == "malformed"  # empty answer
    assert parse_reply("<answer>unclosed")[0] == "malformed"


def test_grade_uses_aliases_and_normalization():
    aliases = ["harry sinclair lewis", "sinclair lewis"]
    assert grade("Sinclair Lewis", aliases)
    assert grade("  the Sinclair Lewis. ", aliases)
    assert not grade("Upton Sinclair", aliases)
    assert not grade("", aliases)


def _task(wrong_penalty=3.0, **kw) -> QAAbstainTask:
    t = QAAbstainTask.__new__(QAAbstainTask)  # skip __init__ (no download)
    t.wrong_penalty = wrong_penalty
    t._bands = None
    t._band_mix = None
    t._rates = kw.get("rates", {})
    t._train = kw.get("train", [])
    t._eval = kw.get("eval", [])
    return t


EX = Example(messages=[], meta={"qid": "q1", "question": "?",
                               "aliases": ["paris"], "gold": "Paris"})


def test_reward_structure():
    t = _task(wrong_penalty=3.0)
    assert t.reward(EX, "<answer>Paris</answer>").total == 1.0
    assert t.reward(EX, "<abstain/>").total == 0.0
    assert t.reward(EX, "<answer>London</answer>").total == -3.0
    # malformed must not be a cheaper escape than a wrong answer
    assert t.reward(EX, "dunno lol").total == -3.0
    parts = t.reward(EX, "<answer>London</answer>").parts
    assert parts == {"format": 1.0, "answered": 1.0, "correct": 0.0, "wrong": 1.0}
    assert t.reward(EX, "<abstain/>").parts["answered"] == 0.0


def test_injected_completion_is_the_calibration_oracle():
    # Known-band question: inject the gold answer (reward +1); everything
    # else (uncertain, unknown, unprobed): inject abstain (reward 0). Both
    # directions covered = neither collapse is an absorbing state.
    t = _task(rates={"q1": 1.0})
    text = t.injected_completion(EX)
    assert parse_reply(text) == ("answer", "Paris")
    assert t.reward(EX, text).total == 1.0

    for rates in ({"q1": 0.5}, {"q1": 0.0}, {}):
        t = _task(rates=rates)
        text = t.injected_completion(EX)
        assert parse_reply(text) == ("abstain", None)
        assert t.reward(EX, text).total == 0.0


def test_band_cutoffs():
    assert _band(1.0) == "known"
    assert _band(0.8) == "known"
    assert _band(0.5) == "uncertain"
    assert _band(0.0) == "unknown"


def test_curriculum_draw_respects_band_mix():
    t = _task()
    known = [{"qid": f"k{i}", "question": "kq", "aliases": ["a"]} for i in range(5)]
    unknown = [{"qid": f"u{i}", "question": "uq", "aliases": ["a"]} for i in range(5)]
    t._train = known + unknown
    t._bands = {"known": known, "uncertain": [], "unknown": unknown, "unprobed": []}
    t._band_mix = {"known": 0.0, "uncertain": 0.0, "unknown": 1.0}
    rng = random.Random(0)
    draws = {t._draw(rng)["qid"][0] for _ in range(50)}
    assert draws == {"u"}  # never draws from the zero-weight/empty bands


def test_curriculum_falls_back_when_bands_unusable():
    t = _task()
    t._train = [{"qid": "x", "question": "q", "aliases": ["a"]}]
    t._bands = {"known": [], "uncertain": [], "unknown": [], "unprobed": []}
    t._band_mix = {"known": 1.0}
    assert t._draw(random.Random(0))["qid"] == "x"


@pytest.mark.integration
def test_dataset_loads_and_splits():
    from mlx_rl.tasks import get_task

    t = get_task("qa_abstain")  # downloads/reads the cached parquet
    assert len(t._train) > 100_000 and len(t._eval) == 500
    ex = t.eval_sample(random.Random(0))
    assert "<answer>" in ex.messages[0]["content"] or "answer tags" in ex.messages[0]["content"]
    assert ex.meta["aliases"]


def test_chat_probe_scoring():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "qa_chat_probe", pathlib.Path("scripts/qa_chat_probe.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert m.contains_alias("It's the Adder, I think.", ["adder", "adders"])
    assert not m.contains_alias("Madder music!", ["adder"])  # word boundary
    assert m.contains_alias("Harry S. Truman was president",
                            ["harry s truman"])

    assert m.classify("Honestly, I'm not sure about that one.") == "hedge"
    assert m.classify("I'd have to guess here.") == "hedge"
    assert m.classify("The capital is Paris.") == "answer"
    # a confident negative is a DENIAL, not calibrated ignorance — and it
    # wins even when wrapped in hedging language
    assert m.classify("There is no such physicist.") == "denial"
    assert m.classify("She doesn't exist, as far as I know.") == "denial"
    assert m.classify("I'm not sure, but that person appears to be "
                      "invented.") == "denial"

    long_summary = ("This paper introduces a novel framework for " * 10)
    assert m.paper_verdict(long_summary) == "fabricated"
    assert m.paper_verdict("I can't access external links, and this paper "
                           "is after my knowledge cutoff.") == "disclaimed"
    assert m.paper_verdict("Interesting title!") == "other"

    assert m._TAG_BLEED.search("<abstain/>") and m._TAG_BLEED.search("<ANSWER>x")


def test_introspect_auroc():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "qa_introspect", pathlib.Path("scripts/qa_introspect.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # perfect separation -> 1.0; anti-separation -> 0.0; random-ish -> ~0.5
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert m.auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert m.auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
