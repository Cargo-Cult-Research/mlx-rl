"""Integration tests on the cached tiny model (skipped when not cached)."""

import random

import pytest

pytestmark = pytest.mark.integration

from mlx_rl.profiles import get_profile


@pytest.fixture(scope="module")
def tiny():
    from huggingface_hub import snapshot_download

    from mlx_rl.config import LoraConfig
    from mlx_rl.models import load_policy

    prof = get_profile("tiny")
    try:
        snapshot_download(prof.model, local_files_only=True)
    except Exception:
        pytest.skip("tiny model not in local HF cache")
    model, tokenizer, _ = load_policy(prof.model, LoraConfig(rank=8))
    return model, tokenizer


def _prompt(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def test_batched_greedy_matches_sequential(tiny):
    from mlx_rl.engine import rollout_groups
    from mlx_rl.rollout import sample_completion

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Compute 6 + 7. Answer with just the number.")

    seq_toks, seq_lps, _ = sample_completion(model, tokenizer, prompt, 24, 0.0)
    groups, stats = rollout_groups(
        model, tokenizer, [prompt], group_size=3, max_new_tokens=24, temperature=0.0
    )
    for comp in groups[0]:
        assert comp.tokens == seq_toks, "greedy batched decode must equal sequential"
        assert comp.finish_reason in ("stop", "length")
        assert len(comp.logprobs) == len(comp.tokens)
        assert all(lp <= 0.0 for lp in comp.logprobs)
    assert stats.generation_tokens > 0


def test_share_prompt_equals_full_prefill(tiny):
    from mlx_rl.engine import rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Name the capital of France in one word.")
    a, _ = rollout_groups(
        model, tokenizer, [prompt], 2, 24, 0.0, share_prompt=True
    )
    b, _ = rollout_groups(
        model, tokenizer, [prompt], 2, 24, 0.0, share_prompt=False
    )
    assert a[0][0].tokens == b[0][0].tokens


def test_multi_prompt_groups_and_eos(tiny):
    from mlx_rl.engine import rollout_groups
    from mlx_rl.rollout import eos_ids

    model, tokenizer = tiny
    prompts = [
        _prompt(tokenizer, "Say the single word: hello"),
        _prompt(tokenizer, "Compute 2 + 2. Answer with just the number."),
    ]
    groups, _ = rollout_groups(model, tokenizer, prompts, 2, 32, 0.0)
    eos = eos_ids(tokenizer)
    assert len(groups) == 2 and all(len(g) == 2 for g in groups)
    for g in groups:
        for comp in g:
            assert comp.tokens, "every completion must produce tokens"
            if comp.finish_reason == "stop":
                assert comp.tokens[-1] in eos


def test_sampled_rollouts_have_spread(tiny):
    from mlx_rl.engine import rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Write one short sentence about the sea.")
    groups, _ = rollout_groups(
        model, tokenizer, [prompt], 4, 24, 1.0
    )
    texts = {tuple(c.tokens) for c in groups[0]}
    assert len(texts) > 1, "temp=1 group should not be degenerate"
