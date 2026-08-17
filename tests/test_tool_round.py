"""Tool-round rendering must equal the chat template's own multi-turn render
(the cache-splice invariant), and the qa_arxiv reward must implement the
pre-registered table."""
import json
import random
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
    from mlx_rl.tasks.qa_arxiv import WEB_SEARCH_TOOL, format_tool_call

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


def _task(tmp_path):
    from mlx_rl.tasks.qa_arxiv import QAArxivTask
    rows = [
        {"id": "1706.03762", "title": "Attention Is All You Need",
         "authors": ["Ashish Vaswani", "Noam Shazeer"], "published": "2017-06-12",
         "categories": ["cs.CL"], "famous": True},
        {"id": "2603.15031", "title": "Attention Residuals", "authors": ["Bo Li"],
         "published": "2026-03-16", "categories": ["cs.LG"], "famous": False},
        {"id": "2605.00001", "title": "Sparse Routing for Long Context Agents",
         "authors": ["Ana Ruiz", "Wei Chen"], "published": "2026-05-02",
         "categories": ["cs.LG"], "famous": False},
        {"id": "fictional_0000", "title": "Sparse Routing: A Unified Theory of Nothing",
         "authors": [], "published": None, "categories": [], "fictional": True},
    ]
    p = tmp_path / "snap.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return QAArxivTask(snapshot=str(p), backend="snapshot", judge=False, eval_frac=0.0)


def test_index_is_date_aware_and_exact_first(tmp_path):
    t = _task(tmp_path)
    hits = t.index.search("Attention Residuals", today="2026-08-16")
    assert [h["id"] for h in hits][0] == "2603.15031"
    # before publication the paper does not exist yet
    assert all(h["id"] != "2603.15031"
               for h in t.index.search("Attention Residuals", today="2026-01-01"))
    assert t.index.search("A Unified Theory of Nothing", today="2026-08-16") == []
    # partial-word queries still find the paper (>= 60% coverage)
    assert t.index.search("sparse routing long context", today="2026-06-01")[0]["id"] == "2605.00001"


def test_reward_table(tmp_path):
    t = _task(tmp_path)
    from mlx_rl.tasks.base import Example
    def ex(regime, aliases=("Vaswani",), qtype="authors"):
        return Example(messages=[], meta={"regime": regime, "aliases": list(aliases),
                                          "qtype": qtype, "question": "q", "id": "x"})
    def ep(visible="r", calls=()):
        return {"visible": visible, "tool_calls": list(calls), "finish": "stop"}
    HIT = {"ok": True, "hits": 1, "found_target": True}
    EMPTY = {"ok": True, "hits": 0, "found_target": False}
    NEAR = {"ok": True, "hits": 2, "found_target": False}  # near-misses only
    ans = lambda v: {"kind": "answer", "value": v}
    ABST, DEN = {"kind": "abstain", "value": None}, {"kind": "denial", "value": None}
    # known: direct correct answer +1; with a needless call, minus the cost
    assert t._score(ex("known"), ep(), ans("Vaswani et al.")).total == 1.0
    assert t._score(ex("known"), ep(calls=[HIT]), ans("Vaswani")).total == pytest.approx(0.9)
    assert t._score(ex("known"), ep(), ans("Hinton")).total == -3.0
    assert t._score(ex("known"), ep(), DEN).total == -3.0
    assert t._score(ex("known"), ep(), ABST).total == 0.0
    # post: correct after hits +1 and grounded; abstain after hits 0
    r = t._score(ex("post"), ep(calls=[HIT]), ans("Vaswani"))
    assert r.total == 1.0 and r.parts["grounded"] == 1.0
    assert t._score(ex("post"), ep(calls=[HIT]), ABST).total == 0.0
    assert t._score(ex("post"), ep(), DEN).total == -3.0
    # future / fictional: decline after an EMPTY search is correct; unchecked
    # denial is penalised; any answer is a fabrication
    for reg in ("future", "fictional"):
        assert t._score(ex(reg), ep(calls=[EMPTY]), ABST).total == 1.0
        assert t._score(ex(reg), ep(calls=[EMPTY]), DEN).total == 1.0
        assert t._score(ex(reg), ep(calls=[NEAR]), ABST).total == 1.0
        assert t._score(ex(reg), ep(), DEN).total == -3.0
        assert t._score(ex(reg), ep(), ABST).total == 0.0
        assert t._score(ex(reg), ep(calls=[EMPTY]), ans("Vaswani")).total == -3.0
    # no visible reply (round cap / truncation) is scored, not silent
    assert t._score(ex("post"), ep(visible=""), None).parts["no_reply"] == 1.0


def test_injected_episode_is_symmetric(tmp_path):
    t = _task(tmp_path)
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        e = t.sample(rng)
        segs = t.injected_episode(e)
        seen.add(e.meta["regime"])
        if e.meta["regime"] == "known":
            assert len(segs) == 1 and segs[0][1] is True
        else:
            assert [g for _, g in segs] == [True, False, True]
            assert "<tool_call>" in segs[0][0]
    assert {"known", "post", "future", "fictional"} <= seen
    r = t.index.render([], "Nothing Here")
    assert r == 'No results found for "Nothing Here".'
    hits = t.index.search("Attention Residuals", today="2026-08-16")
    assert t.index.render(hits, "x").startswith("1. Attention Residuals\n   https://arxiv.org/abs/2603.15031 · 2026-03-16 · Bo Li")


def test_author_grading_closes_the_common_surname_hole():
    from mlx_rl.tasks.qa_arxiv import author_or_year_match as m
    al = ["Jinming Wang", "Wang"]
    assert m("Jinming Wang, Hai Wang, Hongkai Wen", al)
    assert m("Wang et al.", al)
    assert m("J. Wang and colleagues", al)  # short reply naming the surname: fine
    # a fabricated 8-name list that happens to contain a Wang is NOT correct
    assert m("Haohan Wang, Yongfeng Zhang, Qiaoqiao Jin, Wei Li, Xin Chen, Yu Zhao, Song Wang, Li Na", al) is False
    assert m("2022", ["2022"]) and not m("2023", ["2022"])
