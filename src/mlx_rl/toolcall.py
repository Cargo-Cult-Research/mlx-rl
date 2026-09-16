"""The model's native tool-call emission format (qwen3 chat template): an
inner <function=NAME> block nested in <tool_call> tags."""
from __future__ import annotations

import re

_FUNC_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>", re.S)


def parse_tool_call(text: str):
    """-> (name, {param: value}) for the first well-formed call, else None."""
    m = _FUNC_RE.search(text)
    if not m:
        return None
    return m.group(1), {k: v for k, v in _PARAM_RE.findall(m.group(2))}


def format_tool_call(name: str, **params) -> str:
    """The one true form, per the model's own chat template."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"
