"""Serve captured real search results: real noise, deterministic replay.

A frozen corpus of SERPs captured once through the Brave API (see
scripts/capture_serps.py), served through fuzzy query matching so the
paraphrases a policy actually types resolve to the capture made for that item.

This is the arm the snapshot index cannot be. The snapshot returns a bare
"No results found." for anything fictional, which is a free tell -- the policy
learns "empty means decline" without ever facing the case that matters. A real
engine asked for a paper that does not exist returns five REAL papers on
adjacent topics, confidently ranked. Measured 2026-08-26 over 12 fictional
titles: mean relevance 0.36, and the top hit is nearly always a genuine paper
with overlapping words. Noticing that the result is a DIFFERENT paper than the
one asked about is the skill; a bare empty never asks for it.

Date filtering is possible here in a way it is not against a live engine.
Because we own the captured rows, a result whose arXiv id places it after the
stated `today` can be dropped, which keeps the future/fictional falsification
test that only the snapshot could run before. Non-arXiv results carry no
reliable date and are passed through -- documented rather than guessed at.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .webtools import _content_words, relevance, render_results

_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}")


def _clean(s: str) -> str:
    """Tag-stripped text -> what a person would read. Entities decoded
    (`&quot;` costs six characters to say `"`), whitespace collapsed."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())


def _arxiv_month(href: str) -> str | None:
    """-> 'YYYY-MM' for an arXiv URL, else None. 2104.09864 -> 2021-04."""
    m = _ARXIV_ID.search(href.lower())
    if not m:
        return None
    yy, mm = m.group(1), m.group(2)
    if not ("01" <= mm <= "12"):
        return None
    return f"20{yy}-{mm}"


class SerpIndex:
    """Captured SERPs, looked up by fuzzy query match.

    Exact normalized query first, then content-word overlap. `min_cov` is
    deliberately loose: the policy types "RoFormer authors" for a row captured
    under the full title, and refusing that would report "not found" for a
    paper the corpus plainly holds -- a lie the reward would then train on.
    """

    def __init__(self, path: str | Path, min_cov: float = 0.55,
                 max_hits: int = 5, max_chars: int = 2500,
                 body_chars: int = 300, dates: dict[str, str] | None = None):
        self.rows = []
        for line in Path(path).open():
            if line.strip():
                r = json.loads(line)
                if r.get("ok") and r.get("results"):
                    for h in r["results"]:
                        h["body"] = _clean(h.get("body", ""))
                        h["title"] = _clean(h.get("title", ""))
                    self.rows.append(r)
        self._norm = [" ".join(re.findall(r"[a-z0-9]+", r["q"].lower())) for r in self.rows]
        self._words = [_content_words(r["q"]) for r in self.rows]
        self.min_cov, self.max_hits, self.max_chars = min_cov, max_hits, max_chars
        self.body_chars = body_chars
        # id -> publication date of the ITEM the row was captured for. Per-hit
        # arXiv-id filtering alone leaves the future regime half-enforced: a
        # capture for a 2019 paper still carries undated NeurIPS and ACM hits
        # that sail past a 2018 `today`. The item's own date is the honest
        # gate -- if the paper did not exist yet, none of its coverage did.
        self.dates = dates or {}

    def _match(self, query: str):
        """Best matching captured row, or None."""
        qn = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        qw = _content_words(query)
        if not qn:
            return None
        best, best_score = None, 0.0
        for r, tn, tw in zip(self.rows, self._norm, self._words):
            if qn and (qn in tn or tn in qn):
                score = 2.0 + len(qw & tw) / max(1, len(tw))
            elif qw and tw:
                cov = len(qw & tw) / len(qw)
                if cov < self.min_cov:
                    continue
                score = cov
            else:
                continue
            if score > best_score:
                best, best_score = r, score
        return best

    def lookup(self, query: str) -> list[dict] | None:
        r = self._match(query)
        return r["results"] if r else None

    def search(self, query: str, today: str | None = None) -> list[dict]:
        row = self._match(query)
        if row is None:
            return []
        if today:
            pub = self.dates.get(row.get("id", ""))
            if pub and pub > today:
                return []      # the item does not exist yet: nothing about it does
        hits = list(row["results"])
        if today:
            # Drop what could not have been indexed yet. Only arXiv results
            # carry a date we can trust; everything else passes through, so
            # the future regime is enforced for arXiv rows and best-effort
            # elsewhere. Stated plainly because a silent partial filter would
            # look like a working falsification test that is not one.
            hits = [h for h in hits
                    if (m := _arxiv_month(h.get("href", ""))) is None or m <= today[:7]]
        return hits[: self.max_hits]

    def render(self, hits: list[dict], query: str = "") -> str:
        """Search-engine shaped. What an empty MEANS is the policy's call."""
        if not hits:
            return f'No results found for "{query.strip()[:120]}".'
        return render_results(hits, self.max_chars, self.body_chars)

    def coverage(self) -> dict:
        """What the corpus can and cannot answer -- for the lab book."""
        rel = [r.get("relevance", 0.0) for r in self.rows]
        return {"rows": len(self.rows),
                "mean_relevance": round(sum(rel) / max(1, len(rel)), 3),
                "pass_gate": sum(x >= 0.5 for x in rel)}


__all__ = ["SerpIndex", "relevance"]
