"""Task interface: a task supplies prompts and a verifiable reward."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Example:
    messages: list[dict]
    meta: dict = field(default_factory=dict)
    # Per-example chat-template kwargs (e.g. tools=[...]) merged over the
    # run's global chat_kwargs when the prompt is rendered.
    chat_kwargs: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    """What a task's run_tool() hands back: the text injected into the
    transcript as the tool response, plus bookkeeping the reward may use
    (hit counts, error flags) that the policy never sees."""
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class RewardResult:
    total: float
    parts: dict = field(default_factory=dict)


class Task(Protocol):
    name: str

    def sample(self, rng: random.Random) -> Example: ...

    def reward(self, example: Example, completion: str) -> RewardResult: ...

    # Optional — tool-using tasks (see qa_arxiv). When `tools` is set the
    # trainer samples segmented episodes (engine.rollout_episodes): the
    # policy's tool calls are executed by run_tool() and the response is
    # injected as context; grading goes through episode_reward().
    #   tools: list[dict]                       # offered via the chat template
    #   def run_tool(self, name, args, example) -> ToolResult
    #   def episode_reward(self, examples, episodes) -> list[RewardResult]
    #       episodes[i] = {"visible": str, "tool_calls": [dict], "finish": str}


_REGISTRY: dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_task(name: str, **kwargs) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
