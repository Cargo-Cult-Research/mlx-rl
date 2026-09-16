# How much the training adds over the prompt alone

Every row compares the **same base model under the same system prompt and the
same eval**, adapter present versus absent. The prompt is `HONESTY_SYSTEM`
(`src/mlx_rl/tasks/qa_abstain.py`), the one-paragraph honesty-about-uncertainty
prompt that ships with the adapter. The trivia-trained adapters are inert
without it (prompt-off probes sit at base everywhere), so "prompt-only" is the
honest control and "prompt + adapter" is the artifact.

Subject: **Qwen3.6-35B-A3B** at 4-bit, adapter C-200. Free-chat probes, k=4
samples per question, judge-graded commitment (hedge = decline; denial =
asserts the thing does not exist; confident-wrong = answered wrongly with no
hedge). Full context, including the arm-A rows and the tool-using adapters:
[qa-glove-results.md](qa-glove-results.md).

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

Base without any prompt, for scale: fictional-people hedge 0.20, post-cutoff
papers 0.50, post-cutoff author hedge 0.03. The prompt alone does real work
(0.20 → 0.56, 0.03 → 0.26); RL on top roughly doubles it again while keeping
the known side flat.

Per-item binding — does the policy *read* its own uncertainty rather than
hedge more everywhere: decline 0.02 on questions the base knows versus 0.35 on
ones it does not, AUROC 0.74 on held-out never-trained questions. That is a
16× separation the prompt alone does not have (prompt-only: 0.17 hedge on
knowns). Replicated on a second seed; prompt-off inertness confirmed for C-200
at or below base on every bucket.

## Reproducing a row

The chat buckets come from `scripts/qa_chat_probe.py`, run twice with the same
seed and calibration file — once without `--adapter` for the prompt-only
control:

```sh
uv run python scripts/qa_chat_probe.py --calib data/labels/trivia-pass@8-qwen36.jsonl \
    --system honesty --k 4 --out runs/uplift-prompt-only
uv run python scripts/qa_chat_probe.py --calib data/labels/trivia-pass@8-qwen36.jsonl \
    --system honesty --adapter <adapter-dir> --k 4 --out runs/uplift-c200
```

The paper-recall buckets (famous versus post-cutoff authors and years) were
measured with a separate arXiv recall probe that is not shipped in this repo;
`scripts/matrix_eval.py --cells papers` scores the same material through the
honesty task's `papers` domain.
