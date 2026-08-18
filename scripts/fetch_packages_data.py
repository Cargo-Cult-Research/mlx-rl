"""Fetch the package-hallucination data (Spracklen et al. 2024, MIT):
- LLM_AT.json  : ~4,900 Python coding prompts derived from popular packages
- pypi_package_names.csv : PyPI master list (Jan 2024) = the grader's truth
Derives data/packages_prompts.jsonl (capability phrases, ~600) and stores the
master list under data/pypi_master.txt (gitignored: 7.5 MB).
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/Spracks/PackageHallucination/main/Data/Python/"
DATA = Path(__file__).parent.parent / "data"


def get(name: str) -> str:
    with urllib.request.urlopen(RAW + name, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def main() -> None:
    prompts = []
    for line in get("LLM_AT.json").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except Exception:
            continue
        if not isinstance(p, str) or "\n" in p or len(p) > 260:
            continue
        m = re.match(r"(?i)^(generate|write|create)\s+(a\s+)?python\s+(code|script|program|function|class)?\s*(that|to|which)?\s*", p)
        cap = p[m.end():] if m else p
        cap = cap.strip().rstrip(".")
        if len(cap) < 25:
            continue
        prompts.append(cap[0].lower() + cap[1:])
    rng = random.Random(1)
    rng.shuffle(prompts)
    prompts = prompts[:600]
    with (DATA / "packages_prompts.jsonl").open("w") as f:
        for i, c in enumerate(prompts):
            f.write(json.dumps({"id": f"pkg_{i:04d}", "capability": c}) + "\n")
    master = get("pypi_package_names.csv")
    names = set()
    for line in master.splitlines()[1:]:
        n = line.split(",")[0].strip().strip('"').lower()
        if n:
            names.add(n)
    (DATA / "pypi_master.txt").write_text("\n".join(sorted(names)) + "\n")
    print(f"{len(prompts)} prompts -> data/packages_prompts.jsonl; {len(names)} PyPI names -> data/pypi_master.txt")


if __name__ == "__main__":
    main()
