"""Toy verifiable task: multi-operand integer arithmetic with a format contract.

Difficulty is tuned so a ~0.5B instruct model is mediocre (mixed rewards
within a group = gradient signal). Reward: 1.0 exact answer in tags, 0.2 for
well-formed tags with a wrong value, 0.0 otherwise.
"""

from __future__ import annotations

import random
import re

from .base import Example, RewardResult, register

_ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")

FORMAT_REWARD = 0.2
CORRECT_REWARD = 1.0


@register
class ArithmeticTask:
    name = "arithmetic"

    def __init__(
        self,
        n_operands: int = 3,
        max_operand: int = 999,
        format_reward: float = FORMAT_REWARD,
    ):
        self.n_operands = n_operands
        self.max_operand = max_operand
        # 0.0 = pure-correctness reward. The 0.2 default proved dangerous
        # under a tight token budget (truncation reward hacking: the model
        # learned to skip reasoning and guess inside tags).
        self.format_reward = format_reward

    def sample(self, rng: random.Random) -> Example:
        operands = [rng.randint(10, self.max_operand) for _ in range(self.n_operands)]
        signs = [rng.choice(["+", "-"]) for _ in range(self.n_operands - 1)]
        expr = str(operands[0])
        answer = operands[0]
        for sign, operand in zip(signs, operands[1:]):
            expr += f" {sign} {operand}"
            answer = answer + operand if sign == "+" else answer - operand
        # Invite reasoning BEFORE the tags. The earlier wording ("reply with
        # only the final integer...") made reasoning non-compliant, so RL
        # collapsed to terse guessing — the instruction fought the CoT.
        return Example(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Compute {expr}. Think step by step, then end your "
                        "reply with the final integer wrapped in answer tags, "
                        "like <answer>7</answer>."
                    ),
                }
            ],
            meta={"expr": expr, "answer": answer},
        )

    def reward(self, example: Example, completion: str) -> RewardResult:
        # LAST match: thinking-mode completions may draft tags inside the
        # CoT; the contract is that the reply ENDS with the answer tags.
        m = None
        for m in _ANSWER_RE.finditer(completion):
            pass
        if m is None:
            return RewardResult(0.0, {"format": 0.0, "correct": 0.0})
        correct = int(m.group(1)) == example.meta["answer"]
        return RewardResult(
            CORRECT_REWARD if correct else self.format_reward,
            {"format": 1.0, "correct": float(correct)},
        )
