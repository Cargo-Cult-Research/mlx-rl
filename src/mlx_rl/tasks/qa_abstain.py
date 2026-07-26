"""Calibrated factuality task — answer or abstain, with a verifiable reward.

The model answers a short factual question inside <answer></answer> tags, or
replies <abstain/> when it does not know. Reward:

    correct answer   +1.0
    abstain           0.0
    wrong answer     -wrong_penalty   (default 3.0)
    malformed reply  -wrong_penalty   (a non-reply must not dodge the penalty)

The penalty IS the calibration target: the reward-optimal policy answers
exactly when its own p(correct) > wrong_penalty / (1 + wrong_penalty)
(3.0 -> 0.75). Training this with GRPO is an RL existence proof for
"know when you don't know" — the verifiable core of hallucination reduction.

Datasets (fetched from the HF Hub on first use, never redistributed here):
- "triviaqa" (default): TriviaQA rc.nocontext (Apache-2.0, ~138k train).
  Grading = normalized exact match against the dataset's alias sets.
- "popqa": PopQA (MIT, ~14k entity-centric questions) — intended as the
  OUT-OF-DISTRIBUTION transfer eval: train on triviaqa, then check that the
  learned answer/abstain policy is calibrated here too, not just in-domain.

Difficulty curriculum (optional): pass `calib_file` (jsonl written by
scripts/qa_calibrate.py: {"qid", "pass_rate"}) and `band_mix` to control the
mix of known / uncertain / unknown questions per batch. Group-relative
advantages need decision variance inside a group; an uncontrolled pool is
dominated by questions the model always gets right or always misses, which
yield zero-variance groups (no gradient).

Known reward-hack surface (defended): drafting tags inside <think> is void —
the trainer grades only the visible post-think reply; hedged prose fails
strict tag parsing; garbage output scores like a wrong answer, not like an
abstention. The residual risk to WATCH in samples.jsonl is abstain-collapse
(abstaining on everything) — visible immediately as frac_answered -> 0.
"""
from __future__ import annotations

import json
import random
import re
import string

from .base import Example, RewardResult, register

_N_EVAL = 500
# Last tag wins: models may deliberate before committing.
_TAG_RE = re.compile(r"<answer>(.*?)</answer>|<abstain\s*/?\s*>", re.DOTALL | re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    """SQuAD-style answer normalization: lowercase, drop punctuation and
    articles, collapse whitespace."""
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def parse_reply(text: str) -> tuple[str, str | None]:
    """-> ("answer", value) | ("abstain", None) | ("malformed", None).
    The LAST well-formed tag in the reply decides."""
    last = None
    for m in _TAG_RE.finditer(text):
        last = m
    if last is None:
        return "malformed", None
    if last.group(1) is not None:
        val = last.group(1).strip()
        return ("answer", val) if val else ("malformed", None)
    return "abstain", None


def grade(value: str, aliases: list[str]) -> bool:
    norm = normalize(value)
    return bool(norm) and norm in {normalize(a) for a in aliases}


def _band(pass_rate: float) -> str:
    if pass_rate >= 0.8:
        return "known"
    if pass_rate <= 0.0:
        return "unknown"
    return "uncertain"


def load_triviaqa() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "mandarjoshi/trivia_qa",
        "rc.nocontext/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path, columns=["question", "question_id", "answer"])
    rows = []
    for r in table.to_pylist():
        aliases = list(r["answer"]["normalized_aliases"] or [])
        if r["answer"].get("value"):
            aliases.append(r["answer"]["value"])
        if r["question"] and aliases:
            rows.append({"qid": r["question_id"], "question": r["question"].strip(),
                         "aliases": aliases})
    return rows


def load_popqa() -> list[dict]:
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            aliases = json.loads(r["possible_answers"])
            if r["question"] and aliases:
                rows.append({"qid": f"popqa_{r['id']}",
                             "question": r["question"].strip(),
                             "aliases": aliases})
    return rows


PROMPT = (
    "Answer this question with a short factual answer, wrapped in answer "
    "tags like <answer>Paris</answer>. If you do not reliably know the "
    "answer, reply with exactly <abstain/> instead of guessing.\n\n"
    "Question: {q}"
)


@register
class QAAbstainTask:
    name = "qa_abstain"

    def __init__(
        self,
        dataset: str = "triviaqa",
        wrong_penalty: float = 3.0,
        seed: int = 12345,
        calib_file: str | None = None,
        band_mix: dict | None = None,
        **_,
    ):
        self.wrong_penalty = wrong_penalty
        rows = {"triviaqa": load_triviaqa, "popqa": load_popqa}[dataset]()
        rng = random.Random(seed)
        rng.shuffle(rows)
        self._eval = rows[:_N_EVAL]
        self._train = rows[_N_EVAL:]
        # Optional curriculum: bucket the train pool by probed pass rate and
        # draw bands by weight. Questions absent from the calib file stay in
        # a shared "unprobed" pool drawn with the residual weight mass.
        self._bands: dict[str, list[dict]] | None = None
        self._band_mix = band_mix
        if calib_file:
            rates = {}
            with open(calib_file) as f:
                for line in f:
                    r = json.loads(line)
                    rates[r["qid"]] = float(r["pass_rate"])
            self._bands = {"known": [], "uncertain": [], "unknown": [], "unprobed": []}
            for row in self._train:
                key = _band(rates[row["qid"]]) if row["qid"] in rates else "unprobed"
                self._bands[key].append(row)
            self._band_mix = dict(band_mix or {"known": 0.25, "uncertain": 0.5, "unknown": 0.25})

    def _example(self, row: dict) -> Example:
        return Example(
            messages=[{"role": "user", "content": PROMPT.format(q=row["question"])}],
            meta={"qid": row["qid"], "question": row["question"], "aliases": row["aliases"]},
        )

    def _draw(self, rng: random.Random) -> dict:
        if not self._bands:
            return rng.choice(self._train)
        names = [n for n in self._band_mix if self._bands.get(n)]
        weights = [self._band_mix[n] for n in names]
        if not names:  # calib file matched nothing usable
            return rng.choice(self._train)
        return rng.choice(self._bands[rng.choices(names, weights=weights, k=1)[0]])

    def sample(self, rng: random.Random) -> Example:
        return self._example(self._draw(rng))

    def eval_sample(self, rng: random.Random) -> Example:
        return self._example(rng.choice(self._eval))

    def injected_completion(self, example: Example) -> str:
        """Off-policy demonstration for --inject-r: the base policy almost
        never samples an abstention (measured 0/152 pilot rollouts), so
        without injection there is no gradient toward it."""
        return "<abstain/>"

    def reward(self, example: Example, completion: str) -> RewardResult:
        kind, value = parse_reply(completion)
        if kind == "abstain":
            return RewardResult(0.0, {"format": 1.0, "answered": 0.0,
                                      "correct": 0.0, "wrong": 0.0})
        if kind == "malformed":
            return RewardResult(-self.wrong_penalty,
                                {"format": 0.0, "answered": 0.0,
                                 "correct": 0.0, "wrong": 1.0})
        ok = grade(value, example.meta["aliases"])
        return RewardResult(
            1.0 if ok else -self.wrong_penalty,
            {"format": 1.0, "answered": 1.0,
             "correct": float(ok), "wrong": float(not ok)},
        )
