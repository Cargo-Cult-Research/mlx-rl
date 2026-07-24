"""SAGE decoding tests (faithful to arXiv 2602.08354; see docs/sage-paper-notes.md).

Fast unit tests cover the confidence-gate control flow (Φ, top-h acceptance)
with no model. Integration tests exercise the step-wise beam end-to-end on the
cached tiny model.
"""
import math
import random

import pytest

from mlx_rl.engine import _accept, _Cand


# ----------------------------- fast unit tests -----------------------------
def _cand(phi: float, ended: str | None, ntok: int = 4) -> _Cand:
    return _Cand(cache=[], pending=0, tokens=[0] * ntok,
                 logprobs=[phi] * ntok, sum_lp=phi * ntok, ended=ended)


def test_phi_is_mean_logprob():
    assert math.isclose(_Cand(cache=[], pending=0, tokens=[1, 2],
                              logprobs=[-1.0, -3.0], sum_lp=-4.0).phi, -2.0)


def test_accept_think_in_band():
    # </think> is the highest-Φ candidate -> accepted at any h>=1.
    ranked = [_cand(-0.5, "think"), _cand(-1.0, "delim"), _cand(-2.0, "delim")]
    kind, c = _accept(ranked, h=1)
    assert kind == "think" and c is ranked[0]


def test_low_confidence_think_below_band_is_rejected():
    # THE FIX: a </think> ranked below top-h is NOT accepted (the old eager
    # tol-stop would have taken it). h=2 -> band is the first two.
    ranked = [_cand(-0.2, "delim"), _cand(-0.4, "delim"), _cand(-3.0, "think")]
    assert _accept(ranked, h=2) == (None, None)
    # widen the band to include it -> now accepted.
    assert _accept(ranked, h=3)[0] == "think"


def test_highest_phi_think_wins_within_band():
    ranked = [_cand(-0.3, "think"), _cand(-0.5, "think"), _cand(-0.6, "delim")]
    _, c = _accept(ranked, h=3)
    assert c is ranked[0]  # ranked is Φ-descending


def test_eos_takes_precedence_in_band():
    # a model-closed turn (eos) in the band beats a think in the band.
    ranked = [_cand(-0.4, "think"), _cand(-0.5, "eos"), _cand(-0.6, "delim")]
    assert _accept(ranked, h=3)[0] == "eos"


def test_no_terminal_in_band_returns_none():
    ranked = [_cand(-0.4, "delim"), _cand(-0.5, "cap"), _cand(-0.6, "think")]
    assert _accept(ranked, h=2) == (None, None)


def test_tr_maps_to_h():
    # h = round(TR * 2m); TR=0.5,m=2 -> h=2 ; TR=1.0,m=2 -> h=4
    assert max(1, round(0.5 * 2 * 2)) == 2
    assert max(1, round(1.0 * 2 * 2)) == 4


# --------------------------- integration (tiny) ----------------------------
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


def _prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True)


@integration
def test_step_wise_runs_end_to_end(tiny):
    from mlx_rl.engine import sage_completion
    from mlx_rl.rollout import eos_ids
    model, tokenizer = tiny
    nl = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    comp = sage_completion(
        model, _prompt(tokenizer, "Compute 2 + 3. Think, then answer."),
        think_end=999, eos=eos_ids(tokenizer), step_delim={nl},
        m=2, tr=0.5, max_new_tokens=64, max_reasoning_steps=8,
        max_step_tokens=32, think_temperature=1.0, answer_temperature=0.0)
    assert comp.tokens and len(comp.logprobs) == len(comp.tokens)
    assert all(lp <= 1e-6 for lp in comp.logprobs)
    assert comp.think_len is not None and 1 <= comp.think_len <= len(comp.tokens)
    assert comp.finish_reason in ("stop", "length")


@integration
def test_step_budget_forces_stop(tiny):
    # A think_end the model won't emit + tiny step budget -> forced commit.
    from mlx_rl.engine import sage_completion
    from mlx_rl.rollout import eos_ids
    model, tokenizer = tiny
    nl = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    comp = sage_completion(
        model, _prompt(tokenizer, "Write a long story about the sea."),
        think_end=999, eos=eos_ids(tokenizer), step_delim={nl},
        m=2, tr=0.5, max_new_tokens=48, max_reasoning_steps=2,
        max_step_tokens=8, think_temperature=0.0, answer_temperature=0.0,
        batched=False)  # this asserts the unbatched force mechanic specifically
    assert comp.think_len is not None
    assert comp.tokens[comp.think_len - 1] == 999  # forced </think> at the boundary
    assert len(comp.tokens) <= 48


@integration
def test_hybrid_collect_rollouts(tiny):
    from mlx_rl.config import TrainConfig
    from mlx_rl.tasks import get_task
    from mlx_rl.train import collect_rollouts
    model, tokenizer = tiny
    cfg = TrainConfig(
        group_size=3, max_new_tokens=48, sage_r=1, sage_m=2, sage_tr=0.5,
        sage_max_reasoning_steps=3, sage_max_step_tokens=8,
        think_end=tokenizer.encode("\n\n", add_special_tokens=False)[-1])
    task = get_task("arithmetic", n_operands=2, max_operand=99)
    examples = [task.sample(random.Random(0)) for _ in range(2)]
    rollouts, _, skipped = collect_rollouts(model, tokenizer, examples, cfg, task)
    assert len(rollouts) == 6
    assert skipped == 0  # two-stage off by default
    for i, r in enumerate(rollouts):
        is_sage = i % 3 == 2
        assert r.sage == is_sage
        assert (r.think_len is not None) == is_sage
        assert r.completion_tokens
