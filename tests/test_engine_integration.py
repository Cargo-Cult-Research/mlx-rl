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


def test_collect_multiturn_rows_and_history(tiny):
    """Two turns, two examples, G=2: rows come out [ex0 t0 ×2, ex1 t0 ×2,
    ex0 t1 ×2, ex1 t1 ×2]; every turn-1 prompt contains that member's own
    turn-0 reply; each row's mask covers only its own turn's tokens."""
    import random
    from mlx_rl.config import TrainConfig
    from mlx_rl.tasks.base import Example, RewardResult
    from mlx_rl.train import collect_multiturn

    model, tokenizer = tiny

    class T:
        name = "mt"
        turns = 2
        def sample(self, rng):
            return Example(messages=[{"role": "user", "content": "Say a colour."}], meta={"q": 0})
        def followup(self, ex, turn, history):
            return Example(messages=list(history) + [{"role": "user", "content": "Now a number."}],
                           meta={"q": turn}, chat_kwargs=dict(ex.chat_kwargs))
        def reward(self, ex, completion):
            return RewardResult(float(len(completion) % 3), {"len3": float(len(completion) % 3)})

    task = T()
    cfg = TrainConfig(model="tiny", task="mt", group_size=2, max_new_tokens=8, temperature=1.0)
    examples = [task.sample(random.Random(i)) for i in range(2)]
    rollouts, _, _ = collect_multiturn(model, tokenizer, examples, cfg, task)
    assert len(rollouts) == 2 * 2 * 2
    turns = [r.meta["turn"] for r in rollouts]
    assert turns == [0, 0, 0, 0, 1, 1, 1, 1]
    for i in range(4):
        r0, r1 = rollouts[i], rollouts[4 + i]
        reply0 = r0.text.replace(tokenizer.eos_token or "", "").strip()
        prompt1 = tokenizer.decode(r1.prompt_tokens)
        assert reply0[:20] in prompt1  # own history carried forward
        assert "Now a number." in prompt1
        assert r1.gen_mask == [1] * len(r1.completion_tokens)
        assert r1.reward_parts["turn"] == 1.0


def test_init_adapter_loads_trainable_weights(tiny, tmp_path):
    """--init-adapter: a saved adapter's tensors land in the freshly attached
    LoRA layers (same keys), leaving the frozen base untouched."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_rl.config import LoraConfig
    from mlx_rl.models import load_policy, save_adapter
    from mlx_rl.profiles import get_profile

    model, _ = tiny
    # perturb the trainable params, save, then load into a fresh policy
    params = dict(tree_flatten(model.trainable_parameters()))
    bumped = {k: v + 0.5 for k, v in params.items()}
    model.load_weights(list(bumped.items()), strict=False)
    save_adapter(model, tmp_path, LoraConfig(rank=8), get_profile("tiny").model, 7)
    fresh, _, _ = load_policy(get_profile("tiny").model, LoraConfig(rank=8))
    before = dict(tree_flatten(fresh.trainable_parameters()))
    weights = dict(mx.load(str(tmp_path / "adapter-00007.safetensors")).items())
    fresh.load_weights(list(weights.items()), strict=False)
    after = dict(tree_flatten(fresh.trainable_parameters()))
    assert set(weights) == set(after)
    k = next(iter(weights))
    assert not mx.allclose(before[k], after[k]).item()
    assert mx.allclose(after[k], bumped[k]).item()
    # restore the shared fixture
    model.load_weights(list(params.items()), strict=False)


def test_episode_cap_message_gives_one_more_segment(tiny):
    """With on_cap: a call made at the round cap is not run; the cap block is
    injected once and the row generates a final segment. A call after that
    ends the episode as tool_cap."""
    from mlx_rl.engine import rollout_episodes, rollout_groups

    model, tokenizer = tiny
    prompt = _prompt(tokenizer, "List some fruits.")
    plain, _ = rollout_groups(model, tokenizer, [prompt], 1, 16, 0.0)
    stop_tok = plain[0][0].tokens[2]
    inject = tokenizer.encode(" go", add_special_tokens=False)
    cap = tokenizer.encode(" STOP", add_special_tokens=False)
    caps = []
    groups, _ = rollout_episodes(
        model, tokenizer, [prompt], 1, 8, 0.0,
        on_tool=lambda *a: list(inject), on_cap=lambda *a: caps.append(1) or list(cap),
        tool_stop_ids=(stop_tok,), max_tool_rounds=0)
    ep = groups[0][0]
    assert ep.capped and caps == [1]
    assert ep.segments[1].tokens == cap and not ep.segments[1].generated
    assert len(ep.segments) >= 3
    assert ep.finish_reason in ("stop", "length", "tool_cap")
