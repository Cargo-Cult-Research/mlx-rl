# captured SERPs

Real search results, captured once through the Brave Search API on 2026-08-26/27
and frozen. Served by `mlx_rl.serps.SerpIndex` with fuzzy query matching, so the
paraphrases a policy types resolve to the capture made for that item.

## why this exists

Anonymous scraping produced a tool worse than no tool. Measured over the 41,966
cached "hits" in `runs/webcache/`: **88.6% were off-topic**. Bing and yahoo --
the only two engines that did not block us -- passed the relevance gate on 11%
of paper titles, and real vs fictional papers were statistically
indistinguishable through the tool (5% vs 11% relevant), erasing the exact
distinction the honesty task exists to teach.

## what is here

| | rows | mean relevance | pass gate (>=0.5) |
|---|---|---|---|
| real papers | 1857 | 0.997 | 100% |
| fictional papers | 300 | 0.386 | 23% |

That separation is the point. A fictional title returns five REAL papers on
adjacent topics, confidently ranked -- noticing the result is a *different*
paper is the skill. The snapshot index returns a bare "No results found" there,
which is a free tell.

Rendered tool output: median 1909 chars, cap 2500,
per-result body cap 300. Bodies are HTML-unescaped and cuts land on a word
boundary with an ellipsis, so a truncation is visible rather than read as fact.

## regimes

`SerpIndex(dates=...)` hides every hit for an item published after the stated
`today`. Per-hit arXiv-id filtering alone is not enough -- a capture for a 2019
paper carries undated ACL and NeurIPS rows that would sail past a 2018 `today`.
With the item date supplied, the future regime is fully enforced here, which
until now only the snapshot backend could do.

## cost

2157 queries at $5/1000 = $10.79. One-time; the corpus is
frozen and costs nothing at training time. Trivia (2,000 items) is NOT yet
captured.
