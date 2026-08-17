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


def test_episode_cache_splice_matches_fresh_prefill(tiny):
    """A row that stops on a tool token, gets a block prefilled onto its
    own KV cache and resumes must continue exactly as a fresh prefill of
    prompt + segment + block would (greedy). This is the invariant that
    makes cache-spliced tool rounds identical to a re-rendered transcript."""
    from mlx_rl.engine import rollout_episodes, rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Write three short sentences about the sea.")
    # First find a token the greedy continuation actually emits, to use as
    # the "tool stop": the first token of the plain greedy completion after
    # position 3 (so segment 1 is non-trivial).
    plain, _ = rollout_groups(model, tokenizer, [prompt], 1, 24, 0.0,
                              share_prompt=False)
    toks = plain[0][0].tokens
    assert len(toks) > 6
    stop_tok = toks[4]
    inject = tokenizer.encode(" Also:", add_special_tokens=False)
    calls = []

    def on_tool(pi, gi, ep, text):
        calls.append(text)
        return list(inject)

    groups, _ = rollout_episodes(
        model, tokenizer, [prompt], 1, 12, 0.0,
        on_tool=on_tool, tool_stop_ids=(stop_tok,), max_tool_rounds=1,
        share_prompt=False)
    ep = groups[0][0]
    assert len(calls) == 1
    assert ep.rounds == 1
    assert [s.generated for s in ep.segments] == [True, False, True]
    seg1, inj, seg2 = ep.segments
    assert seg1.tokens == toks[:5] and seg1.finish_reason == "tool"
    assert inj.tokens == inject
    assert ep.gen_mask == [1] * 5 + [0] * len(inject) + [1] * len(seg2.tokens)
    assert len(seg2.tokens) > 0
    # Fresh prefill of the spliced prefix must greedy-continue identically.
    fresh, _ = rollout_groups(model, tokenizer, [prompt + seg1.tokens + inject],
                              1, len(seg2.tokens), 0.0, share_prompt=False)
    assert fresh[0][0].tokens == seg2.tokens


def test_episode_shared_prompt_and_round_cap(tiny):
    """Group members share the prompt cache; a stop on the tool token with
    no rounds left ends the episode as 'tool_cap' rather than injecting."""
    from mlx_rl.engine import rollout_episodes, rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Count from one to ten in words.")
    plain, _ = rollout_groups(model, tokenizer, [prompt], 1, 16, 0.0)
    stop_tok = plain[0][0].tokens[3]
    seen = []
    groups, _ = rollout_episodes(
        model, tokenizer, [prompt], 3, 16, 0.0,
        on_tool=lambda *a: seen.append(a) or None,
        tool_stop_ids=(stop_tok,), max_tool_rounds=0)
    assert not seen
    for ep in groups[0]:
        assert ep.finish_reason == "tool_cap"
        assert ep.segments[0].tokens == plain[0][0].tokens[:4]


def test_episode_tools_run_off_thread_and_rows_reenter(tiny):
    """A slow tool must not stall the batch: with a 0.4 s tool and 3 rows
    that all call it, wall time is ~one tool latency, not three, and every
    row resumes with the injected block in its transcript."""
    import time
    from mlx_rl.engine import rollout_episodes, rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "Name three colours.")
    plain, _ = rollout_groups(model, tokenizer, [prompt], 1, 16, 0.0)
    stop_tok = plain[0][0].tokens[2]
    inject = tokenizer.encode(" ok", add_special_tokens=False)

    def slow_tool(pi, gi, ep, text):
        time.sleep(0.4)
        ep.tool_calls.append({"name": "t"})
        return list(inject)

    t0 = time.perf_counter()
    groups, _ = rollout_episodes(model, tokenizer, [prompt], 3, 12, 0.0,
                                 on_tool=slow_tool, tool_stop_ids=(stop_tok,),
                                 max_tool_rounds=1, tool_workers=4)
    dt = time.perf_counter() - t0
    for ep in groups[0]:
        assert ep.rounds == 1 and ep.tool_calls == [{"name": "t"}]
        assert [s.generated for s in ep.segments][:2] == [True, False]
        assert ep.segments[1].tokens == inject
        assert ep.finish_reason in ("stop", "length")
    assert dt < 0.4 * 3 + 2.0  # not serialized: 3 rows, ~1 latency (+ decode)
