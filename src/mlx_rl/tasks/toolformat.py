"""Tool-call format task — a synthetic end-to-end exercise for the RL loop.

The model's own chat template specifies exactly one canonical call format
(newline-delimited <tool_call> / <function=..> / <parameter=..> blocks,
optional reasoning BEFORE the call, "NO suffix" after). This task rewards
exactly that form via a STRICT parser, plus correct tool choice and args.

Besides being a trainable task, this doubles as a format-drift *detector*
(e.g. for regression-testing promoted adapters): a sub-1.0 canonical
baseline on a given model/serving stack is a regression signal, to be
root-caused at the serving layer, not trained away here. (qwen36 measures
100% canonical at baseline.)

Reward: 1.0 canonical + right tool + required args match;
        0.7 canonical + right tool, wrong/missing args;
        0.4 canonical form, wrong tool;
        0.0 anything else (incl. drift the tolerant parser would accept).
"""

from __future__ import annotations

import random
import re

from .base import Example, RewardResult, register

# Canonical block, exactly as the qwen3.x chat template teaches it.
_CALL_RE = re.compile(
    r"<tool_call>\n"
    r"<function=([A-Za-z0-9_\-]+)>\n"
    r"((?:<parameter=[A-Za-z0-9_\-]+>\n(?:.*?)\n</parameter>\n)*)"
    r"</function>\n"
    r"</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_\-]+)>\n(.*?)\n</parameter>", re.DOTALL
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk and return its contents",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "absolute path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return stdout",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "the command"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return the top results",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, replacing it",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "absolute path"},
                    "content": {"type": "string", "description": "text to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "directory path"}},
                "required": ["path"],
            },
        },
    },
]

_PATHS = [
    "/etc/hosts", "/var/log/system.log", "/tmp/report.txt", "/Users/dev/notes.md",
    "/opt/app/config.yaml", "/home/data/results.csv", "/srv/www/index.html",
]
_DIRS = ["/tmp", "/var/log", "/Users/dev/projects", "/opt/app", "/srv/www"]
_CMDS = [
    "uptime", "df -h", "ps aux | head -5", "uname -a", "whoami",
    "netstat -an | head -10", "du -sh /tmp",
]
_QUERIES = [
    "current weather in Zurich", "MLX framework documentation",
    "swiss train timetable Bern to Geneva", "python 3.13 release notes",
    "M3 Ultra memory bandwidth", "GRPO reinforcement learning paper",
]
_CONTENTS = ["hello world", "TODO: refactor the parser", "backup complete", "42"]

# (tool, request template) — several phrasings per tool, some indirect.
_SCENARIOS = [
    ("read_file", "Show me the contents of {path}."),
    ("read_file", "What's in {path}?"),
    ("read_file", "Can you open {path} and tell me what it says?"),
    ("bash", "Run `{command}` for me."),
    ("bash", "Execute this command: {command}"),
    ("bash", "I need the output of `{command}`."),
    ("web_search", "Search the web for {query}."),
    ("web_search", "Look up {query} online."),
    ("write_file", "Write '{content}' to {path}."),
    ("write_file", "Save the text '{content}' into {path}."),
    ("list_dir", "What files are in {path}?"),
    ("list_dir", "List the directory {path}."),
]

_FOLLOWUPS = [
    ("read_file", "Thanks. Now show me {path} as well."),
    ("bash", "Good. Now run `{command}`."),
    ("list_dir", "OK — and what's in {path}?"),
    ("web_search", "Now search the web for {query}."),
]


def _fill(rng: random.Random, tool: str, template: str) -> tuple[str, dict]:
    args = {}
    if "{path}" in template:
        args["path"] = rng.choice(_DIRS if tool == "list_dir" else _PATHS)
    if "{command}" in template:
        args["command"] = rng.choice(_CMDS)
    if "{query}" in template:
        args["query"] = rng.choice(_QUERIES)
    if "{content}" in template:
        args["content"] = rng.choice(_CONTENTS)
    return template.format(**args), args


def render_call(name: str, args: dict) -> str:
    """The one true form, per the model's own chat template."""
    params = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


@register
class ToolFormatTask:
    name = "toolformat"

    def __init__(self, post_tool_fraction: float = 0.5):
        self.post_tool_fraction = post_tool_fraction
        self._tools_by_name = {t["function"]["name"]: t["function"] for t in TOOLS}

    # Prompts need the tools wired into the chat template.
    chat_template_kwargs = {"tools": TOOLS}

    def sample(self, rng: random.Random) -> Example:
        tool, template = rng.choice(_SCENARIOS)
        request, args = _fill(rng, tool, template)
        messages = [{"role": "user", "content": request}]
        scenario = "cold"

        if rng.random() < self.post_tool_fraction:
            # Drift lived in post-tool-result turns: seed history with one
            # completed canonical call + its result, then ask for another.
            scenario = "post_tool"
            first_tool, first_template = rng.choice(_SCENARIOS)
            first_request, first_args = _fill(rng, first_tool, first_template)
            tool, followup = rng.choice(_FOLLOWUPS)
            request, args = _fill(rng, tool, followup)
            messages = [
                {"role": "user", "content": first_request},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": first_tool, "arguments": first_args},
                        }
                    ],
                },
                {"role": "tool", "content": "ok — completed successfully"},
                {"role": "user", "content": request},
            ]

        return Example(
            messages=messages,
            meta={"tool": tool, "args": args, "scenario": scenario},
        )

    def reward(self, example: Example, completion: str) -> RewardResult:
        parts = {
            "canonical": 0.0,
            "name": 0.0,
            "args": 0.0,
            # diagnostic only: drift that the tolerant production parser
            # might still accept (function tag present, form wrong)
            "has_function_tag": float("<function=" in completion),
        }
        m = _CALL_RE.search(completion)
        if m is None:
            return RewardResult(0.0, parts)
        # "NO suffix": nothing but whitespace after the call, and no second
        # call or stray tags before it either beyond free-text reasoning.
        if completion[m.end() :].strip():
            return RewardResult(0.0, parts)
        parts["canonical"] = 1.0

        name = m.group(1)
        got_args = {k: v for k, v in _PARAM_RE.findall(m.group(2))}
        schema = self._tools_by_name.get(name)
        if name != example.meta["tool"] or schema is None:
            return RewardResult(0.4, parts)
        parts["name"] = 1.0

        required = set(schema["parameters"].get("required", []))
        known = set(schema["parameters"]["properties"])
        expected = example.meta["args"]
        ok = (
            required <= set(got_args)
            and set(got_args) <= known
            and all(got_args.get(k, "").strip() == str(v) for k, v in expected.items())
        )
        if not ok:
            return RewardResult(0.7, parts)
        parts["args"] = 1.0
        return RewardResult(1.0, parts)
