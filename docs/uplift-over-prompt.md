# Uplift over prompt-only — every phase on the same footing

*For anyone asking "how much does the RL add over just the prompt?"
One table per phase; every row compares the **same base model under the
same system prompt and the same eval**, adapter present vs absent. Numbers
are copied from the run summaries named in each section, not re-derived.
Last updated 2026-08-17.*

The prompt in question is `HONESTY_SYSTEM` (`tasks/qa_abstain.py`), the
one-paragraph honesty-about-uncertainty prompt that ships with every
adapter; from arm 1 on it also carries the stated date. The adapters are
inert without it (measured: prompt-off probes sit at base everywhere), so
"prompt-only" is the honest control and "prompt + adapter" is the artifact.

## Phase 1 — calibrated abstention in free chat (C-200, 2026-07-31)

Free-chat probes, k=4 samples per question, judge-graded commitment
(hedge = decline; denial = asserts the thing does not exist; confident-wrong =
answered wrongly with no hedge). Source: `runs/qa-chat-base-glove-sys-20260730`
vs `runs/qa-chat-gloveC200-sys-20260731`; arXiv recall from
`runs/papers-recall-*`; full context in `docs/qa-glove-results.md`.

| bucket (n) | metric | prompt-only | prompt + C-200 | Δ |
|---|---|---|---|---|
| fictional people (80) | hedge+denial ↑ | 0.56 | **0.95** | +0.39 |
| fictional people (80) | confident-wrong ↓ | 0.44 | **0.05** | −0.39 |
| fictional people (80) | denial ↓ | 0.29 | 0.16 | −0.13 |
| post-cutoff papers, summarize frame (80) | hedge+denial ↑ | 0.79 | **0.92** | +0.13 |
| post-cutoff papers — *who wrote it* (80) | hedge+denial ↑ | 0.26 | **0.65** | +0.39 |
| post-cutoff papers — *who wrote it* (80) | confident-wrong ↓ | 0.74 | **0.35** | −0.39 |
| post-cutoff papers — *what year* (80) | hedge+denial ↑ | 0.46 | **0.88** | +0.42 |
| post-cutoff papers — *what year* (80) | confident-wrong ↓ | 0.54 | **0.12** | −0.42 |
| famous papers — authors (40) | correct (kept) | 1.00 | 1.00 | 0 |
| famous papers — year (40) | correct (kept) | 0.93 | 0.95 | +0.02 |
| real-obscure (12) | hedge+denial ↑ | 0.08 | 0.33 | +0.25 |
| chat-known (24) | correct (cost) | 0.42 | 0.46 | +0.04 |
| chat-known (24) | hedge+denial (cost) ↓ | 0.17 | 0.21 | +0.04 |
| agentic coding, SWE-bench lv-72 | solved | 45 | 45 | 0 |

Base without any prompt, for scale: fictional-people hedge 0.20, papers-post
0.50, post-cutoff author hedge 0.03. So the prompt alone does real work
(0.20 → 0.56, 0.03 → 0.26) and RL on top roughly doubles it again while
keeping the known side flat. Per-item binding (does the policy *read* its own
uncertainty rather than hedge more everywhere): decline 0.02 on questions the
base knows vs 0.35 on ones it doesn't, AUROC 0.74 on held-out never-trained
questions — a 16× separation the prompt alone does not have (prompt-only:
0.17 hedge on knowns). Replicated on a second seed. Prompt-off inertness
confirmed for C-200 (at/below base every bucket).

## Phase 2 — tools and the stated date (arm 1, 2026-08-16/17)

The prompt now also states today's date; the model is offered real
`web_search` + `fetch_url`. Eval: `scripts/arxiv_transfer_eval.py`, 64
held-out questions × 2 samples, **live web tools** (fallback share reported),
judge-graded episode reward (+1 correct / 0 decline / −3 wrong, denial-
without-search, or no reply; decline after a search that found nothing = +1).
Source: `runs/arxiv-transfer-20260817b`. Two adapters, trained on different
search backends (see design doc §9.1).

| slice (n) | prompt-only | + sandbox-v3-60 | + web-v4-120 |
|---|---|---|---|
| all (128) | 0.40 | **0.92** (+0.52) | 0.89 (+0.49) |
| findable paper (82) | 0.85 | 0.91 | **0.95** |
| known paper (28) | 0.99 | 0.90 (always calls) | 0.94 |
| fictional title (18) | −2.61 | **1.00** (declines after checking 0.94) | 0.56 |
| calls the tool | 0.68 | 1.00 | 0.91 |

Reading: with real tools the prompt-only base already searches (0.68) and
answers findable papers well (0.85); what it cannot do is read five
plausible junk results for a made-up title and decline — it fabricates
(−2.61). That is where the +0.5 comes from. Caveat: the fictional slice is
n=18 and both adapters were evaluated under the day's tool weather.

Two prompt-dependent footnotes worth knowing before quoting a number:
- With the *earlier* demo tool (`search_arxiv`, whose description said "use
  this whenever asked about a paper you don't know") plus a hand-written
  "use tools before declining" clause, prompt-only already called the tool
  100% and scored 0.65 in the sandbox — most of the behaviour was in the
  tool description. With a plain `web_search` and no clause the prompt-only
  call rate fell to 0.06 in the sandbox. Uplift numbers depend on which
  prompt you call "the prompt"; the table above uses the served one
  (honesty prompt + date, no clause, generic tools).
- In the sandbox (snapshot index), the same comparison read −1.25
  prompt-only vs 0.92 trained; those numbers are not comparable to the real-
  web table and are kept only in the design doc.

## Phase 3 — multi-turn (arm 2, running 2026-08-17)

Same eval, 3-turn transcripts (each member carries its own history), greedy,
n=32 questions × 3 turns. Source: `runs/arxiv-transfer-mt3-20260817`.

| | all | turn 0 | turn 1 | turn 2 | fictional | findable |
|---|---|---|---|---|---|---|
| prompt-only | 0.00 | 0.59 | −0.13 | −0.47 | −1.85 | 0.38 |
| + sandbox-v3-60 (single-turn trained) | **0.67** | 0.97 | 0.71 | 0.34 | 0.55 | 0.67 |
| + web-v4-120 (single-turn trained) | 0.50 | 0.98 | 0.34 | 0.19 | −0.15 | 0.62 |
| + arm 2 (multi-turn trained) | *pending* | | | | | |

Reading: prompt-only decays to below zero by turn 2; the single-turn adapters
lift every turn but decay too (0.97 → 0.34). Arm 2 trains against exactly
that; its in-run n=16 eval read 0.99 / 0.74 / 0.97 at step 40 (baseline
0.99 / 0.43 / −0.34) — the held-out n=32 number lands here when it finishes.

## How to reproduce a row

```sh
# phase 1 probes (prompt-only vs adapter): scripts/qa_chat_probe.py, scripts/papers_recall_probe.py
# phase 2/3:
uv run python scripts/arxiv_transfer_eval.py --n 64 --k 2 --batch 64 \
    --arm base= --arm sandbox=~/models/adapters/qa-arxiv-sandbox-v3-60 \
    --arm web=~/models/adapters/qa-arxiv-web-v4-120            # add --turns 3 for phase 3
```
`base=` is prompt-only: the task builds the system prompt (+ date) for every
arm; only the adapter differs.
