# captured SERPs

Real search results, captured once through the Brave Search API on 2026-08-26/27
and frozen. Served by `mlx_rl.serps.SerpIndex` with fuzzy query matching, so the
paraphrases a policy types resolve to the capture made for that item.

## why this exists

The honesty task needs a search tool whose results separate a real paper from
a fabricated one. Anonymous scraping does not: over 41,966 cached hits, 88.6%
were off-topic, and real and fictional papers were statistically
indistinguishable through the tool (5% vs 11% relevant), erasing the
distinction the task exists to teach. A frozen capture through a search API
is deterministic and keeps that separation, measured below.

## what is here

| | rows | mean relevance | pass gate (>=0.5) |
|---|---|---|---|
| real papers | 1857 | 0.997 | 100% |
| fictional papers | 300 | 0.386 | 23% |

That separation is the point. A fictional title returns five REAL papers on
adjacent topics, confidently ranked; noticing that the result is a *different*
paper is the skill being trained.

Rendered tool output: median 1909 chars, cap 2500,
per-result body cap 300. Bodies are HTML-unescaped and cuts land on a word
boundary with an ellipsis, so a truncation is visible rather than read as fact.

## regimes

`SerpIndex(dates=...)` hides every hit for an item published after the stated
`today`, which is what enforces the future regime. Per-hit arXiv-id filtering
alone is not enough: a capture for a 2019 paper carries undated ACL and
NeurIPS rows that would sail past a 2018 `today`.

## cost

2157 queries at $5/1000 = $10.79, one time: the corpus is frozen and costs
nothing at training time. The trivia domain has no capture and uses live web
tools; `scripts/capture_serps.py --corpus trivia` builds one.
