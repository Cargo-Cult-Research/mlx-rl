"""Capture real search results once, so training can replay them forever.

    uv run python scripts/capture_serps.py --corpus papers --limit 50   # pilot
    uv run python scripts/capture_serps.py --corpus papers              # full

Why capture instead of searching live. Anonymous scraping gave us a tool that
was worse than no tool: measured 2026-08-26, 88.6% of 41,966 cached "hits"
were off-topic, bing and yahoo passed the relevance gate on 11% of paper
titles, and the two engines that DO rank titles correctly (brave, duckduckgo)
refuse anonymous scrapers after roughly ten queries. A row trained on that
learns nothing about checking, because checking returned an airline.

The corpus is bounded -- 2,157 paper titles, 2,000 trivia questions -- so one
query per item covers it permanently. Queries the policy actually issues are
paraphrases of these, which a local fuzzy index resolves at serve time; the
network is only needed once, here.

Resumable by design: every row is flushed as it lands and an existing output
file is read back as the skip set, so a 429 or a dropped connection costs one
query, not the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlx_rl.serps import _clean  # noqa: E402
from mlx_rl.webtools import relevance  # noqa: E402

API = "https://api.search.brave.com/res/v1/web/search"
KEY_FILE = Path.home() / ".config" / "mlx-rl" / "brave.key"


def queries(corpus: str) -> list[dict]:
    """[{q, id, kind}] -- what a policy would plausibly type for each item."""
    if corpus == "papers":
        out = []
        for line in open("data/arxiv_snapshot.jsonl"):
            r = json.loads(line)
            out.append({"q": r["title"], "id": r["id"],
                        "kind": "fictional" if r.get("fictional") else "real"})
        return out
    if corpus == "trivia":
        from mlx_rl.tasks.honesty import CALIB, load_triviaqa
        rates = {json.loads(x)["qid"] for x in open(CALIB["trivia"]) if x.strip()}
        return [{"q": r["question"], "id": r["qid"], "kind": "real"}
                for r in load_triviaqa() if r["qid"] in rates]
    raise SystemExit(f"unknown corpus {corpus}")


def fetch(q: str, key: str, count: int = 5, tries: int = 4) -> dict:
    url = API + "?" + urllib.parse.urlencode({"q": q[:390], "count": count})
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "X-Subscription-Token": key})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read())
            return {"ok": True, "results": [
                {"title": w.get("title", ""), "href": w.get("url", ""),
                 "body": _clean(w.get("description", ""))}
                for w in d.get("web", {}).get("results", [])[:count]]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < tries - 1:
                time.sleep(2 ** attempt)     # the only retryable failures
                continue
            return {"ok": False, "error": f"HTTP {e.code}", "results": []}
        except Exception as e:               # noqa: BLE001
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"ok": False, "error": f"{type(e).__name__}", "results": []}
    return {"ok": False, "error": "retries exhausted", "results": []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("papers", "trivia"))
    ap.add_argument("--limit", type=int, default=0, help="0 = everything (pilot with 50 first)")
    ap.add_argument("--qps", type=float, default=15.0, help="ceiling is 50/s; leave headroom")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    key = KEY_FILE.read_text().strip()
    out = Path(a.out or f"data/serps/{a.corpus}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            try:
                done.add(json.loads(line)["id"])
            except Exception:                # noqa: BLE001
                pass
    items = [x for x in queries(a.corpus) if x["id"] not in done]
    if a.limit:
        items = items[: a.limit]
    print(f"[capture] {a.corpus}: {len(items)} to fetch, {len(done)} already on disk", flush=True)

    gap, t0 = 1.0 / a.qps, time.time()
    stats = {"ok": 0, "fail": 0, "relevant": 0}
    with out.open("a") as f:
        for i, it in enumerate(items, 1):
            d = fetch(it["q"], key)
            rel = relevance(it["q"], d["results"]) if d["results"] else 0.0
            f.write(json.dumps({**it, "ok": d["ok"], "results": d["results"],
                                "relevance": round(rel, 3),
                                "error": d.get("error")}) + "\n")
            f.flush()
            stats["ok" if d["ok"] else "fail"] += 1
            stats["relevant"] += rel >= 0.5
            if i % 50 == 0 or i == len(items):
                el = time.time() - t0
                print(f"  {i}/{len(items)}  ok {stats['ok']}  fail {stats['fail']}  "
                      f"relevant {stats['relevant']} ({stats['relevant']/i:.0%})  "
                      f"{el:.0f}s  ${i * 0.005:.2f}", flush=True)
            if stats["fail"] >= 20 and stats["ok"] == 0:
                raise SystemExit("[capture] 20 failures and no successes -- stopping "
                                 "before this burns credit. Check the key and plan.")
            time.sleep(gap)
    print(f"[capture] wrote {out}  ({stats})", flush=True)


if __name__ == "__main__":
    main()
