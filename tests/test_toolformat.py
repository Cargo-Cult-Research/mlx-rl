import random

import pytest

from mlx_rl.tasks import get_task
from mlx_rl.tasks.toolformat import render_call


@pytest.fixture
def task():
    return get_task("toolformat")


def _ex(task, seed=1, scenario=None):
    rng = random.Random(seed)
    for _ in range(200):
        ex = task.sample(rng)
        if scenario is None or ex.meta["scenario"] == scenario:
            return ex
    raise AssertionError(f"no {scenario} example in 200 draws")


def test_sample_scenarios_exist(task):
    assert _ex(task, scenario="cold").meta["scenario"] == "cold"
    post = _ex(task, scenario="post_tool")
    roles = [m["role"] for m in post.messages]
    assert roles == ["user", "assistant", "tool", "user"]


def test_canonical_call_scores_full(task):
    ex = _ex(task)
    text = render_call(ex.meta["tool"], ex.meta["args"])
    res = task.reward(ex, text)
    assert res.total == 1.0
    assert res.parts == {
        "canonical": 1.0, "name": 1.0, "args": 1.0, "has_function_tag": 1.0,
    }


def test_reasoning_prefix_allowed_suffix_forbidden(task):
    ex = _ex(task)
    call = render_call(ex.meta["tool"], ex.meta["args"])
    assert task.reward(ex, "I'll check that file.\n" + call).total == 1.0
    res = task.reward(ex, call + "\nLet me know if you need more!")
    assert res.total == 0.0  # "NO suffix" is part of the contract


def test_drift_variants_score_zero(task):
    """The exact drift zoo that cost three parser patches."""
    ex = _ex(task)
    name, args = ex.meta["tool"], ex.meta["args"]
    k, v = next(iter(args.items()))
    drifts = [
        # bare <function=> without <tool_call> wrapper (patch #597)
        f"<function={name}>\n<parameter={k}>\n{v}\n</parameter>\n</function>",
        # missing </function> close
        f"<tool_call>\n<function={name}>\n<parameter={k}>\n{v}\n</parameter>\n</tool_call>",
        # name as a parameter tag
        f"<tool_call>\n<function=>\n<parameter=name>\n{name}\n</parameter>\n</function>\n</tool_call>",
        # JSON-style call
        f'<tool_call>{{"name": "{name}", "arguments": {{"{k}": "{v}"}}}}</tool_call>',
        # inline single-line form (values must sit on their own lines)
        f"<tool_call><function={name}><parameter={k}>{v}</parameter></function></tool_call>",
    ]
    for text in drifts:
        assert task.reward(ex, text).total == 0.0, text


def test_wrong_tool_gets_partial(task):
    ex = _ex(task)
    other = "bash" if ex.meta["tool"] != "bash" else "web_search"
    arg = {"command": "ls"} if other == "bash" else {"query": "x"}
    assert task.reward(ex, render_call(other, arg)).total == 0.4


def test_right_tool_wrong_args_gets_partial(task):
    ex = _ex(task)
    bad_args = {k: v + "XXX" for k, v in ex.meta["args"].items()}
    assert task.reward(ex, render_call(ex.meta["tool"], bad_args)).total == 0.7


def test_unknown_extra_param_not_full_credit(task):
    ex = _ex(task)
    args = dict(ex.meta["args"], bogus_param="1")
    assert task.reward(ex, render_call(ex.meta["tool"], args)).total == 0.7


def test_no_call_scores_zero(task):
    ex = _ex(task)
    assert task.reward(ex, "The file contains system hostnames.").total == 0.0
