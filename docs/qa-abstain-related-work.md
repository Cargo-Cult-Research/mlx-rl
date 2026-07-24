# qa_abstain — prior work and what is (and isn't) new here

Positioning notes for the calibrated-factuality task (`tasks/qa_abstain.py`):
teach a model to answer short factual questions when it knows and emit
`<abstain/>` when it doesn't, with a fully verifiable reward. This note maps
the prior work so the contribution is stated honestly — the core idea is
**not** novel, and this repo does not claim it is.

## The framing result: guessing is an incentive problem

**Why Language Models Hallucinate** (Kalai, Nachum, Vempala et al.,
[arXiv:2509.04664](https://arxiv.org/abs/2509.04664)) argues hallucination
persists because binary-graded training and evals reward guessing: an
abstention scores 0 while a guess has positive expected value, so models
that always answer dominate leaderboards. Their proposed fix is explicit
**confidence-threshold scoring** — answer only if you are more than t
confident; wrong answers cost t/(1−t) points.

The `qa_abstain` reward is that scoring rule used directly as an RL
objective: correct +1, abstain 0, wrong −c, so the reward-optimal policy
answers exactly when p(correct) > c/(1+c) (default c=3 → t=0.75). The
penalty is not a heuristic; it *is* the calibration target.

## Models already contain the needed signal

- **Language Models (Mostly) Know What They Know** (Kadavath et al.,
  [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)): P(IK) — models can
  largely predict whether they know the answer; the signal RL must surface.
- **Do LLMs Know What They Don't Know?** (Yin et al.,
  [arXiv:2305.18153](https://arxiv.org/abs/2305.18153)): the SelfAware
  benchmark; self-knowledge exists but is far from ceiling.

## Teaching abstention by SFT / preference optimization

- **R-Tuning** (Zhang et al.,
  [arXiv:2311.09677](https://arxiv.org/abs/2311.09677)): refusal-aware
  instruction tuning — split the training set by whether the model itself
  answers correctly, append "I am sure/unsure", SFT on that.
- **Can AI Assistants Know What They Don't Know?** (Cheng et al., ICML 2024,
  [arXiv:2401.13275](https://arxiv.org/abs/2401.13275)): builds a
  **model-specific Idk dataset** on open-domain QA (incl. TriviaQA) from the
  assistant's own known/unknown split, then aligns with SFT/PO. Our
  `scripts/qa_calibrate.py` pass@k probe is the same move, used here to
  drive a *curriculum* rather than labels.

## RL for truthfulness — the closest prior work

- **TruthRL** (Meta,
  [arXiv:2509.25760](https://arxiv.org/abs/2509.25760),
  [code](https://github.com/facebookresearch/TruthRL)): **GRPO with a
  ternary reward** distinguishing correct / hallucination / abstention —
  algorithmically the same recipe as this task. Reports −28.9%
  hallucination and +21.1% truthfulness vs vanilla RL across four
  knowledge-intensive benchmarks, on Qwen/Llama backbones, with and without
  retrieval. **This is the wheel; we are deliberately rolling it on
  different terrain** (see below).
- **Beyond Binary Rewards (RLCR)** (Damani et al.,
  [arXiv:2507.16806](https://arxiv.org/abs/2507.16806)): RL where the model
  also emits a numeric confidence, rewarded by correctness + Brier score —
  calibrated *verbalized* confidence rather than a discrete answer/abstain
  policy. The natural extension if the discrete version works here.
- **Abstain-R1** ([arXiv:2604.17073](https://arxiv.org/abs/2604.17073)):
  calibrated abstention via verifiable RL — recent, directly on-topic;
  worth a close read before claiming any empirical novelty.

## RL's known side effect — why the regression gates exist

**The Hallucination Tax of Reinforcement Finetuning** (Song et al.,
[arXiv:2505.13988](https://arxiv.org/abs/2505.13988)): standard RFT
*degrades* refusal (−80% refusal rate) and mixing ~10% unanswerable
problems restores it. The mirror-image risk for us: abstention-RL could tax
general capability. Hence the promotion gates in this repo: an adapter must
hold its SWE-bench/HumanEval slices (README "Adapter lifecycle") before the
calibration gain counts.

## What this repo's run adds (and what it doesn't)

**Not new:** the ternary abstention reward under GRPO (TruthRL), the
threshold-scoring rationale (Kalai et al.), model-specific knowledge
probing (Cheng et al.).

**The delta, honestly stated:**

1. **An open, single-machine existence proof.** Prior work runs on
   datacenter fleets; this trains a 35B-A3B MoE **on one 96 GB Apple
   Silicon box**, in-process, with the whole recipe (task, probe,
   curriculum, gates) reproducible from this repo.
2. **Penalty as explicit threshold.** We expose wrong_penalty as the
   calibration knob (c ⇒ t = c/(1+c)) and can sweep it to trace a
   risk-coverage frontier, rather than fixing one symmetric penalty.
3. **Curriculum for group variance.** On one box, rollouts are the budget.
   Uniformly drawn QA groups are dominated by always-right/always-wrong
   questions whose GRPO groups carry zero gradient; the pass@k band mix
   (`calib_file`/`band_mix`) targets sampling where the decision boundary
   actually is.
4. **Transfer + capability gates as first-class endpoints.** Train on
   TriviaQA; report the risk-coverage curve on held-out TriviaQA **and
   out-of-distribution PopQA**; require agentic-coding slices to hold.
   The claim to beat is a *prompted* abstention baseline, not a no-abstain
   strawman.

If the run works, the honest headline is "TruthRL-style calibration
training, reproduced end-to-end on consumer hardware, with a curriculum
that makes it sample-efficient enough to be practical there" — not a new
algorithm.
