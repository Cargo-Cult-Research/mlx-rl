"""KodCode task: prompt formats, exact-coverage eval cycling, and both
sandboxed reward paths (pytest-style KodCode tests + MBPP assert scripts).
Dataset-construction test skips offline; everything else uses __new__."""

import shutil

import pytest

from mlx_rl.tasks.base import Example
from mlx_rl.tasks.kodcode import _EVAL_INSTRUCTION, KodCodeTask

HAVE_SEATBELT = shutil.which("sandbox-exec") is not None


def _task():
    t = KodCodeTask.__new__(KodCodeTask)  # skip __init__ (no download)
    t._sandbox_exec = shutil.which("sandbox-exec")
    t._eval_i = 0
    t._sampling = "replace"
    t._order = []
    t.epoch = 0
    return t


def _completion(body):
    return f"reasoning...\n```python\n{body}\n```"


KOD_ROW = {
    "question_id": "Filter_1_I",
    "question": "Return the sum of a list of numbers.",
    "test_info": [{"function_declaration": "def total(nums):"}],
    "test": ("from solution import total\n\n"
             "def test_total():\n"
             "    assert total([1, 2, 3]) == 6\n\n"
             "def test_total_empty():\n"
             "    assert total([]) == 0\n"),
}

MBPP_ROW = {
    "task_id": 2,
    "prompt": "Write a function to find the shared elements from the given two lists.",
    "test_list": ["assert set(similar_elements((3, 4), (4, 10))) == set((4,))"],
    "test_imports": [],
}


def test_train_prompt_includes_signature_and_contract():
    ex = _task()._train_example(KOD_ROW)
    p = ex.messages[0]["content"]
    assert "def total(nums):" in p
    assert "```python" in p
    assert ex.meta["kind"] == "kodcode"


def test_train_formats_render_distinctly():
    t = _task()
    t._formats = list(("fenced", "bare", "instruct"))
    fenced = t._train_example(KOD_ROW, "fenced").messages[0]["content"]
    bare = t._train_example(KOD_ROW, "bare").messages[0]["content"]
    instruct = t._train_example(KOD_ROW, "instruct").messages[0]["content"]
    assert fenced.startswith(_EVAL_INSTRUCTION) and "```python" in fenced
    assert bare.startswith(_EVAL_INSTRUCTION) and "```python" not in bare
    assert instruct.startswith(KOD_ROW["question"])
    assert len({fenced, bare, instruct}) == 3
    # the question text survives every dialect
    assert all(KOD_ROW["question"] in p for p in (fenced, bare, instruct))


def test_sample_mixes_formats_deterministically():
    import random

    t = _task()
    t._formats = ["fenced", "bare", "instruct"]
    t._train = [KOD_ROW]
    seen = [t.sample(random.Random(s)).meta["format"] for s in range(30)]
    assert len(set(seen)) == 3  # all dialects appear
    assert seen == [t.sample(random.Random(s)).meta["format"]
                    for s in range(30)]  # rng-determined, reproducible


def test_bad_train_format_rejected():
    with pytest.raises(ValueError, match="unknown"):
        KodCodeTask(sandbox=HAVE_SEATBELT, train_formats="fenced,klingon")


def test_eval_prompt_bytematches_evalplus_openai_backend():
    # evalplus.provider.openai: instruction_prefix + f"\n```python\n{p}\n```"
    # where p is the docstring-wrapped problem + first assert. Any drift here
    # silently breaks leaderboard comparability (see 2026-08-13 echo finding).
    ex = _task()._eval_example(MBPP_ROW)
    doc = (f'"""\n{MBPP_ROW["prompt"]}\n{MBPP_ROW["test_list"][0]}\n"""')
    assert ex.messages[0]["content"] == (
        f"{_EVAL_INSTRUCTION}\n```python\n{doc}\n```")
    assert ex.meta["kind"] == "mbpp"


def test_eval_sample_cycles_exact_coverage():
    t = _task()
    t._eval = [dict(MBPP_ROW, task_id=i) for i in range(3)]
    ids = [t.eval_sample(None).meta["task_id"] for _ in range(6)]
    assert ids == [0, 1, 2, 0, 1, 2]  # in order, no replacement


def test_kodcode_reward_pass_and_fail():
    t = _task()
    ex = _task()._train_example(KOD_ROW)
    good = "def total(nums):\n    return sum(nums)"
    bad = "def total(nums):\n    return 0"
    assert t.reward(ex, _completion(good)).total == 1.0
    assert t.reward(ex, _completion(bad)).total == 0.0


def test_kodcode_reward_wrong_function_name_fails():
    t = _task()
    ex = _task()._train_example(KOD_ROW)
    r = t.reward(ex, _completion("def summed(nums):\n    return sum(nums)"))
    assert r.total == 0.0  # tests import `total` from solution.py


def test_mbpp_reward_pass_and_fail():
    t = _task()
    ex = _task()._eval_example(MBPP_ROW)
    good = ("def similar_elements(a, b):\n"
            "    return tuple(set(a) & set(b))")
    assert t.reward(ex, _completion(good)).total == 1.0
    assert t.reward(ex, _completion("def similar_elements(a, b):\n    return ()")).total == 0.0


def test_no_code_scores_zero():
    r = _task().reward(Example(messages=[], meta={"kind": "mbpp"}), "no idea")
    assert r.total == 0.0
    assert r.parts["nopatch"] == 1.0


@pytest.mark.skipif(not HAVE_SEATBELT, reason="sandbox-exec not available")
def test_kodcode_reward_sandboxed_hostile_test_blocked(tmp_path):
    # The sandbox covers the pytest path too: a candidate writing outside its
    # temp dir fails and leaves no side effects.
    t = _task()
    target = tmp_path / "escape.txt"
    ex = Example(messages=[], meta={
        "kind": "kodcode",
        "test": ("from solution import f\n"
                 "def test_f():\n"
                 "    assert f()\n")})
    body = (f"def f():\n"
            f"    open({str(target.resolve())!r}, 'w').write('x')\n"
            f"    return True")
    assert t.reward(ex, _completion(body)).total == 0.0
    assert not target.exists()


def test_real_datasets_load_and_are_disjoint():
    try:
        t = KodCodeTask(sandbox=HAVE_SEATBELT)
    except Exception as e:  # offline / HF unreachable
        pytest.skip(f"dataset fetch failed: {e}")
    assert len(t._eval) == 378  # EvalPlus MBPP+ v0.2 task count
    assert len(t._train) > 3000
    assert all(r["gpt_difficulty"] == "easy" for r in t._train)
    assert all(r["subset"] not in ("Package", "Docs") for r in t._train)


def test_val_frac_splits_disjointly_and_deterministically():
    try:
        a = KodCodeTask(sandbox=HAVE_SEATBELT, subsets="Filter",
                        val_frac=0.1)
    except Exception as e:
        pytest.skip(f"dataset fetch failed: {e}")
    tr = {r["question_id"] for r in a._train}
    va = {r["question_id"] for r in a._val}
    assert va and not (tr & va)                       # non-empty, disjoint
    assert 0.08 < len(va) / (len(tr) + len(va)) < 0.12
    b = KodCodeTask(sandbox=HAVE_SEATBELT, subsets="Filter", val_frac=0.1)
    assert {r["question_id"] for r in b._val} == va    # same seed, same split
    c = KodCodeTask(sandbox=HAVE_SEATBELT, subsets="Filter", val_frac=0.1,
                    seed=999)
    assert {r["question_id"] for r in c._val} != va    # seed actually varies
    ex = a.val_examples()
    assert len(ex) == len(a._val) and ex[0].meta["kind"] == "kodcode"


def test_epoch_sampling_covers_pool_without_replacement():
    import random

    t = _task()
    t._formats = ["instruct"]
    t._sampling = "epoch"
    t._order = []
    t.epoch = 0
    t._train = [dict(KOD_ROW, question_id=f"q{i}") for i in range(20)]
    rng = random.Random(0)
    first = [t.sample(rng).meta["question_id"] for _ in range(20)]
    assert len(set(first)) == 20 and t.epoch == 1     # every problem, once
    second = [t.sample(rng).meta["question_id"] for _ in range(20)]
    assert len(set(second)) == 20 and t.epoch == 2    # next epoch, full again
    assert first != second                             # reshuffled


def test_epoch_sampling_is_deterministic():
    import random

    def run(seed):
        t = _task()
        t._formats = ["instruct"]
        t._sampling = "epoch"
        t._order = []
        t.epoch = 0
        t._train = [dict(KOD_ROW, question_id=f"q{i}") for i in range(10)]
        rng = random.Random(seed)
        return [t.sample(rng).meta["question_id"] for _ in range(15)]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_bad_sampling_mode_rejected():
    # validated before the dataset download, so this never touches the network
    with pytest.raises(ValueError, match="sampling"):
        KodCodeTask(sandbox=HAVE_SEATBELT, subsets="Filter", sampling="wat")


def test_subsets_include_filter():
    try:
        t = KodCodeTask(sandbox=HAVE_SEATBELT, subsets="Filter,Prefill")
    except Exception as e:
        pytest.skip(f"dataset fetch failed: {e}")
    assert {r["subset"] for r in t._train} == {"Filter", "Prefill"}
    assert len(t._train) > 2000
