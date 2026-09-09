"""KodCode train pool + EvalPlus-MBPP eval — leaderboard-comparable code task.

Train: KodCode-Light-RL-10K (Xu et al. 2025, arXiv:2503.02951 — 10k
execution-verified question/solution/pytest triplets curated for RL; the
authors decontaminated against MBPP/HumanEval). Filtered to easy problems by
default (0.5B-class models need the variance) and to subsets whose tests are
stdlib-only. ⚠️ License: CC BY-NC 4.0 (non-commercial) — fine for research,
not for anything commercial.

Eval: evalplus/mbppplus — the exact task set behind the EvalPlus leaderboard
(https://evalplus.github.io/leaderboard.html), MBPP+ v0.2 with 378 tasks.
None of them are trained on (the whole point). eval_sample() CYCLES through
the tasks in dataset order instead of drawing with replacement, so
--eval-n 378 covers every task exactly once and the reported eval_correct is
a leaderboard-comparable "MBPP" (base-test) pass@1 under greedy decoding.
The eval prompt BYTE-MATCHES EvalPlus's openai chat backend (instruction +
```python-fenced docstring with the problem and first assert) — an earlier
version presented the docstring unfenced, and the resulting policy collapsed
to prompt-echoing under the official harness's fenced format (~0.06). Train
prompts use the identical shape. Remaining caveats vs the official harness:
extraction here is "last fenced code block" (theirs is AST-based
sanitization), and the leaderboard's MBPP+ column additionally uses ~5x
augmented tests — for a headline number, run the real EvalPlus harness
against mlx_lm.server with the promoted adapter.

Both reward paths run in the same Seatbelt sandbox + rlimits as the `code`
task (see code.py). KodCode tests are pytest-style with
`from solution import ...`, so the candidate is written to solution.py and
graded by pytest's exit code.

Datasets are fetched via the HF cache on first use (no redistribution).
"""
from __future__ import annotations

import random
import sys

from .base import Example, RewardResult, register
from .code import _TIMEOUT_S, _extract_code, _resolve_sandbox_exec, run_asserts, sandbox_run

_KODCODE_REPO = "KodCode/KodCode-Light-RL-10K"
_MBPP_PLUS_REPO = "evalplus/mbppplus"
# Tests in these subsets import non-stdlib packages: any candidate fails for
# environment reasons, which poisons the reward with always-zero groups.
_EXCLUDE_SUBSETS = {"Package", "Docs"}
# pytest startup adds ~1s of interpreter+plugin overhead on top of the tests
_PYTEST_TIMEOUT_S = _TIMEOUT_S + 4

_EVAL_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
_INSTRUCT_SUFFIX = (
    "Write a self-contained Python solution. Reply with your reasoning, then "
    "the complete solution in one ```python code block. Do not include tests."
)
# Train-prompt dialects. Eval is ALWAYS fenced (the official protocol) — this
# only varies what the policy is trained against. Measured 2026-08-13: a
# policy trained on one dialect scores near zero on the fenced one
# (souplate1: 0.392 bare / 0.032 fenced), and training on fenced alone spends
# itself escaping the echo attractor (souplate3: 0.265 fenced ≈ base's 0.283
# bare — format repair, no capability gain). Mixing aims for both.
_TRAIN_FORMATS = ("fenced", "bare", "instruct")


def _dataset_rows(repo: str, columns: list[str] | None = None) -> list[dict]:
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    fname = next(f for f in sorted(list_repo_files(repo, repo_type="dataset"))
                 if f.endswith(".parquet"))
    path = hf_hub_download(repo, fname, repo_type="dataset")
    return pq.read_table(path, columns=columns).to_pylist()


@register
class KodCodeTask:
    name = "kodcode"

    def __init__(self, sandbox: bool = True, difficulties: str = "easy",
                 subsets: str = "", train_formats: str = "fenced",
                 val_frac: float = 0.0, seed: int = 12345,
                 sampling: str = "replace", **_):
        self._sandbox_exec = _resolve_sandbox_exec(sandbox)
        self._formats = [f.strip() for f in train_formats.split(",") if f.strip()]
        bad = set(self._formats) - set(_TRAIN_FORMATS)
        if bad or not self._formats:
            raise ValueError(f"train_formats: unknown {sorted(bad)}; "
                             f"choose from {list(_TRAIN_FORMATS)}")
        if sampling not in ("replace", "epoch"):  # before the dataset download
            raise ValueError(f"sampling: expected 'replace' or 'epoch', got {sampling!r}")
        want = {d.strip() for d in difficulties.split(",") if d.strip()}
        # Optional include-list on top of the hard excludes. Calibration
        # (scripts/kodcode_calibrate.py, 2026-08 probe on the 4-bit 0.5B):
        # Filter/Prefill sit in the model's learning band (53% of groups
        # carry gradient vs 39% pool-wide); Algorithm/Data_Structure/
        # Leetcode/Code_Contests are ~85%+ unsolved at k=8.
        keep = {s.strip() for s in subsets.split(",") if s.strip()}
        rows = _dataset_rows(
            _KODCODE_REPO,
            ["question", "solution", "test", "test_info",
             "gpt_difficulty", "subset", "question_id"],
        )
        self._train = [r for r in rows
                       if r["gpt_difficulty"] in want
                       and r["subset"] not in _EXCLUDE_SUBSETS
                       and (not keep or r["subset"] in keep)]
        if not self._train:
            raise ValueError(
                f"no KodCode rows for difficulties={sorted(want)}"
                f" subsets={sorted(keep) or 'all'}")
        # Source-distribution holdout for checkpoint selection. Selecting on
        # the MBPP eval would be target-domain (oracle) model selection: this
        # is a transfer setup — train KodCode, test MBPP — so letting
        # target-distribution data pick the checkpoint weakens the claim from
        # "KodCode training transfers" to "the best-on-MBPP checkpoint, chosen
        # using MBPP". Validate on held-out KodCode instead; it is a noisier
        # predictor of test score, and that is the honest cost.
        # val_frac < 1 is a fraction of the pool; >= 1 is an absolute count.
        self._val: list[dict] = []
        if val_frac > 0:
            rows = list(self._train)
            random.Random(seed).shuffle(rows)
            n = int(val_frac) if val_frac >= 1 else max(1, int(len(rows) * val_frac))
            if n >= len(rows):
                raise ValueError(f"val_frac={val_frac} would leave no training rows")
            self._val, self._train = rows[:n], rows[n:]
        self._sampling = sampling
        self._order: list[int] = []   # epoch mode: remaining indices, popped
        self.epoch = 0
        self._eval = _dataset_rows(_MBPP_PLUS_REPO)
        self._eval_i = 0

    def _train_example(self, row, fmt: str | None = None) -> Example:
        fmt = fmt or getattr(self, "_formats", ["fenced"])[0]
        decls = [i.get("function_declaration", "").strip()
                 for i in (row.get("test_info") or [])]
        decls = [d for d in decls if d]
        sig = ("\nYour solution must define: " + "; ".join(decls)
               if decls else "")
        body = f"{row['question'].strip()}{sig}"
        doc = f'"""\n{body}\n"""'
        if fmt == "fenced":
            prompt = f"{_EVAL_INSTRUCTION}\n```python\n{doc}\n```"
        elif fmt == "bare":
            prompt = f"{_EVAL_INSTRUCTION}\n\n{doc}\n"
        else:  # instruct
            prompt = f"{body}\n\n{_INSTRUCT_SUFFIX}"
        return Example(
            messages=[{"role": "user", "content": prompt}],
            meta={"kind": "kodcode", "question_id": row["question_id"],
                  "format": fmt, "test": row["test"]},
        )

    def _eval_example(self, row) -> Example:
        # BYTE-MATCHES evalplus.provider.openai.OpenAIChatDecoder.codegen:
        # instruction + "\n```python\n{prompt.strip()}\n```" where their
        # dataset prompt is the docstring-wrapped description + first assert.
        # The 4-bit 0.5B collapses to echoing under this fenced format unless
        # trained on it (2026-08-13 A/B) — do NOT "improve" the format here:
        # matching the official harness is the point. Scored on the ORIGINAL
        # test_list (the leaderboard's "MBPP" column; the "+" column needs
        # their augmented tests).
        doc = (f'"""\n{row["prompt"].strip()}\n'
               f'{row["test_list"][0].strip()}\n"""')
        prompt = f"{_EVAL_INSTRUCTION}\n```python\n{doc}\n```"
        return Example(
            messages=[{"role": "user", "content": prompt}],
            meta={"kind": "mbpp", "task_id": row["task_id"],
                  "test_list": list(row["test_list"]),
                  "test_imports": list(row.get("test_imports") or [])},
        )

    def sample(self, rng: random.Random) -> Example:
        """sampling='replace' (default) draws i.i.d. with replacement — the
        historical behaviour, and why a 150-step run touched only 581 of 7901
        problems. sampling='epoch' draws WITHOUT replacement, reshuffling when
        the pool is exhausted, so N draws cover min(N, pool) distinct
        problems: full coverage needs one pass instead of the ~n·ln(n) draws
        the coupon-collector bound demands with replacement."""
        if self._sampling == "epoch":
            if not self._order:
                self._order = list(range(len(self._train)))
                rng.shuffle(self._order)
                self.epoch += 1
            row = self._train[self._order.pop()]
        else:
            row = rng.choice(self._train)
        return self._train_example(row, rng.choice(self._formats))

    def get_state(self) -> dict:
        """Sampler position, for deterministic resume. Without this an epoch
        run would restart its shuffle mid-epoch and re-draw problems it had
        already used."""
        return {"order": list(self._order), "epoch": self.epoch}

    def set_state(self, state: dict) -> None:
        self._order = list(state["order"])
        self.epoch = state["epoch"]

    def val_examples(self, fmt: str | None = None) -> list[Example]:
        """Held-out source-distribution problems (needs val_frac > 0). One
        fixed dialect so scores stay comparable across checkpoints; graded by
        the same sandboxed pytest reward as training."""
        return [self._train_example(r, fmt or self._formats[0])
                for r in self._val]

    def eval_sample(self, rng: random.Random) -> Example:
        # Deliberately ignores rng: cycles the eval set in dataset order so
        # eval_n == len(set) is exact full coverage, and any smaller eval_n
        # is the same stable prefix every evaluation.
        row = self._eval[self._eval_i % len(self._eval)]
        self._eval_i += 1
        return self._eval_example(row)

    def reward(self, example: Example, completion: str) -> RewardResult:
        code = _extract_code(completion)
        if not code or "def " not in code:
            return RewardResult(0.0, {"correct": 0.0, "code": 1.0, "nopatch": 1.0})
        if example.meta["kind"] == "mbpp":
            ok = run_asserts(code, example.meta, self._sandbox_exec)
        else:
            p = sandbox_run(
                {"solution.py": code, "test_solution.py": example.meta["test"]},
                [sys.executable, "-m", "pytest", "-x", "-q",
                 "-p", "no:cacheprovider", "test_solution.py"],
                self._sandbox_exec, timeout=_PYTEST_TIMEOUT_S)
            ok = p is not None and p.returncode == 0
        return RewardResult(1.0 if ok else 0.0,
                            {"correct": float(ok), "code": 1.0})
