"""A spliced tool round must render byte-for-byte like the chat template's own
multi-turn render: the engine keeps the prompt cache across the round, so any
drift here is a silent train/serve mismatch."""
from pathlib import Path

import pytest

QWEN = Path.home() / "models/mlx/Qwen3.6-35B-A3B-4bit"


@pytest.fixture(scope="module")
def qtok():
    if not QWEN.exists():
        pytest.skip("qwen36 not on disk")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(QWEN))


@pytest.mark.parametrize("thinking", [False, True])
def test_spliced_tool_round_matches_template_render(qtok, thinking):
    from mlx_rl.rollout import tool_response_ids
    from mlx_rl.toolcall import format_tool_call
    from mlx_rl.webtools import WEB_SEARCH_TOOL

    kw = {"enable_thinking": thinking, "tools": [WEB_SEARCH_TOOL]}
    msgs = [{"role": "system", "content": "S. Today's date is 2026-08-16."},
            {"role": "user", "content": "Who wrote 'X'?"}]
    call = format_tool_call("web_search", query="X")
    result = "- X (2026-01-01)\n  authors: A, B"
    base = qtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    gen_text = ("thinking...\n</think>\n\n" if thinking else "") + call
    full = qtok.apply_chat_template(
        msgs + [{"role": "assistant", "content": gen_text},
                {"role": "tool", "content": result}],
        add_generation_prompt=True, tokenize=False, **kw)
    if not thinking:
        # thinking off: template renders the assistant call verbatim after
        # the empty think block, so base + call is a literal prefix
        assert full.startswith(base + call)
        spliced_tail = full[len(base) + len(call):]
        assert qtok.encode(spliced_tail, add_special_tokens=False) == \
            tool_response_ids(qtok, result, **kw)
    # In both modes the injected block must decode to exactly what follows
    # the call in the template's own render.
    tail = full.rsplit("</tool_call>", 1)[1]
    assert qtok.decode(tool_response_ids(qtok, result, **kw)) == tail
    # and </tool_call> is the single stop token the engine keys on
    assert qtok.encode("</tool_call>", add_special_tokens=False) == [248059]
