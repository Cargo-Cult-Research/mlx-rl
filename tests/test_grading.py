"""Think-aware grading + SAGE budget regressions.

Both bug classes were observed in runs/sage2-codemix (2026-07-11):
1. Step-3 arithmetic: rollouts that hit the token cap INSIDE an unclosed
   <think> block were graded as passes off draft <answer> tags written in
   the CoT (reward 1.0 with no visible reply at all).
2. A SAGE completion recorded think_len 2069 with max_new_tokens 2048 —
   the batched reasoning loop was not bounded by the token budget, and the
   answer phase got zero tokens.
"""
import random

import pytest

from mlx_rl.tasks import get_task
from mlx_rl.tasks.base import Example
from mlx_rl.train import _visible_reply

THINK = "</think>"


def _arith_example(answer=-1156):
    return Example(messages=[], meta={"expr": "x", "answer": answer})


def _grade(text, think_close=THINK):
    task = get_task("arithmetic", format_reward=0.0)
    visible, closed = _visible_reply(text, think_close)
    res = task.reward(_arith_example(), visible)
    return res.total, closed


# --------- the sage2-codemix step-3 failure shapes, distilled ---------

def test_draft_tags_inside_unclosed_think_grade_zero():
    # Hit the cap mid-CoT, never closed think, but drafted the right tags:
    # the old grader called this a pass. It must be 0.
    text = ("Let me draft the reply: <answer>-1156</answer>\n"
            "Wait, let me double-check the arithmetic once more: 813 - 694 ...")
    total, closed = _grade(text)
    assert not closed
    assert total == 0.0


def test_closed_think_with_answer_after_still_passes():
    text = ("I drafted <answer>-1156</answer> above; confirming." + THINK +
            "\nThe result is -1156.\n\n<answer>-1156</answer>")
    total, closed = _grade(text)
    assert closed
    assert total == 1.0


def test_closed_think_draft_only_inside_grades_zero():
    # Closes think but the tags exist ONLY inside the CoT: no visible answer.
    text = "Draft: <answer>-1156</answer> looks right." + THINK + "\nDone."
    total, closed = _grade(text)
    assert closed
    assert total == 0.0


def test_zero_answer_tokens_after_forced_close_grades_zero():
    # The think_len==len shape: reasoning consumed the whole budget, the
    # completion ends AT the forced close with nothing visible after it.
    text = "5. Subtract 636: -520 - 636 = -1156.\n\nFinal Answer: -1156.\n" + THINK
    total, closed = _grade(text)
    assert closed
    assert total == 0.0


def test_non_thinking_mode_grades_full_text():
    total, closed = _grade("The answer is <answer>-1156</answer>",
                           think_close=None)
    assert closed
    assert total == 1.0


# ------------------------- SAGE budget invariant -------------------------

integration = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny():
    from huggingface_hub import snapshot_download

    from mlx_rl.config import LoraConfig
    from mlx_rl.models import load_policy
    from mlx_rl.profiles import get_profile

    prof = get_profile("tiny")
    try:
        snapshot_download(prof.model, local_files_only=True)
    except Exception:
        pytest.skip("tiny model not in local HF cache")
    model, tokenizer, _ = load_policy(prof.model, LoraConfig(rank=8))
    return model, tokenizer


@integration
@pytest.mark.parametrize("batched", [True, False])
def test_sage_never_exceeds_budget_and_leaves_answer_room(tiny, batched):
    from mlx_rl.engine import sage_completion
    from mlx_rl.rollout import eos_ids
    model, tokenizer = tiny
    nl = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    rng = random.Random(0)
    for _ in range(3):
        n = rng.randint(2, 99)
        comp = sage_completion(
            model, tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Compute {n} + {n}. Think first."}],
                add_generation_prompt=True),
            think_end=999, eos=eos_ids(tokenizer), step_delim={nl},
            m=2, tr=0.5, max_new_tokens=64, max_reasoning_steps=8,
            max_step_tokens=16, step_tokens=16, think_temperature=1.0,
            answer_temperature=0.0, answer_reserve=32, batched=batched)
        assert len(comp.tokens) <= 64, "completion overran max_new_tokens"
        assert comp.think_len is not None
        # reasoning capped at max_new_tokens - answer_reserve (+1 forced close)
        assert comp.think_len <= 64 - 32 + 1, "reasoning ate the answer reserve"
