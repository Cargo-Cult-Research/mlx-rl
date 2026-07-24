import random

from mlx_rl.tasks import get_task
from mlx_rl.tasks.arithmetic import CORRECT_REWARD, FORMAT_REWARD


def test_registry():
    task = get_task("arithmetic")
    assert task.name == "arithmetic"


def test_sample_is_seeded_and_answer_is_correct():
    task = get_task("arithmetic")
    a = task.sample(random.Random(7))
    b = task.sample(random.Random(7))
    assert a.meta == b.meta
    assert eval(a.meta["expr"]) == a.meta["answer"]


def test_reward_correct():
    task = get_task("arithmetic")
    ex = task.sample(random.Random(1))
    res = task.reward(ex, f"Sure! <answer>{ex.meta['answer']}</answer>")
    assert res.total == CORRECT_REWARD
    assert res.parts == {"format": 1.0, "correct": 1.0}


def test_reward_wrong_value_gets_format_credit():
    task = get_task("arithmetic")
    ex = task.sample(random.Random(1))
    res = task.reward(ex, f"<answer>{ex.meta['answer'] + 1}</answer>")
    assert res.total == FORMAT_REWARD
    assert res.parts["correct"] == 0.0


def test_reward_malformed():
    task = get_task("arithmetic")
    ex = task.sample(random.Random(1))
    for text in ["the answer is 42", "<answer>forty</answer>", "<answer></answer>", ""]:
        assert task.reward(ex, text).total == 0.0


def test_reward_tolerates_whitespace_and_negatives():
    task = get_task("arithmetic")
    ex = task.sample(random.Random(1))
    ans = ex.meta["answer"]
    res = task.reward(ex, f"<answer>  {ans}  </answer>")
    assert res.total == CORRECT_REWARD
