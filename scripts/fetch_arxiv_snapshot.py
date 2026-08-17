"""Build the frozen arXiv metadata snapshot the qa_arxiv task trains against.

Training must not hit the live arXiv API (thousands of queries per run,
non-deterministic, impolite); the task searches this file instead and the
live API is kept for the demo and held-out eval. Metadata only (id, title,
authors, published date, categories) — arXiv metadata is CC0.

Three slices, all in one jsonl:
  famous    hand-listed well-known papers by arXiv id — the pre-cutoff
            "the model may know this" side. Whether it DOES know each one is
            measured by scripts/arxiv_calibrate.py, not assumed here.
  sweep     the first N cs.LG submissions of every month in [--from, --to]:
            ordinary papers with real dates on both sides of the model's
            knowledge, so the stated-date regimes have material.
  fictional deterministic recombinations of real title halves ("A: B" from
            two different papers), filtered to have no near match in the
            real set. Searching for them must come back empty.

Usage:
    uv run python scripts/fetch_arxiv_snapshot.py --out data/arxiv_snapshot.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
SLEEP_S = 3.0  # arXiv asks for >= 3 s between requests

FAMOUS_IDS = """
1706.03762 1810.04805 2005.14165 1512.03385 1412.6980 2106.09685 2203.02155
2203.15556 2302.13971 2305.18290 1406.2661 1312.6114 1502.03167 1409.3215
1409.0473 1301.3781 2010.11929 2103.00020 2006.11239 1910.10683 1907.11692
1906.08237 2003.10555 2001.08361 1707.06347 1312.5602 1607.06450 1505.04597
1506.02640 1506.01497 1703.06870 1806.07366 2201.11903 1706.03741 2101.03961
2204.02311 2307.09288 2310.06825 2312.00752 2205.14135 2112.04426 2302.04761
2210.03629 2212.08073 2303.08774 2501.12948 2402.03300 2412.15115 2006.11477
2002.05709 1911.05722 2006.07733 2102.12092 2112.10752 2003.08934 1901.02860
1904.10509 2001.04451 2004.05150 2007.14062 2002.05202 2104.09864 2112.11446
2205.01068 2211.05100 2212.10560 2305.10601 2305.14314 2401.04088 2403.08295
2404.14219 2407.21783 1503.02531 1611.01578 1710.10903 1609.02907 1706.02216
2005.11401 2104.08691 2109.01652 2110.08207 2203.11171 2206.07682 2210.11416
2303.12712 2306.05685 2307.08691 2309.06180 2311.05232 2312.11805 2401.02954
2405.04434 2406.06592 2410.21276 2412.19437 2502.03387 2504.07491
""".split()


def _query(params: dict) -> list[dict]:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read(4_000_000)
            root = ET.fromstring(raw)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"arXiv API failed: {url}")
    out = []
    for entry in root.findall("a:entry", NS):
        aid = (entry.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join((entry.findtext("a:title", "", NS) or "").split())
        published = (entry.findtext("a:published", "", NS) or "")[:10]
        authors = [" ".join((a.findtext("a:name", "", NS) or "").split())
                   for a in entry.findall("a:author", NS)]
        cats = [c.get("term") for c in entry.findall("a:category", NS)]
        if aid and title and published:
            out.append({"id": aid, "title": title, "authors": authors,
                        "published": published, "categories": cats})
    return out


def _months(start: str, end: str):
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


_STOPWORDS = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to",
              "via", "by", "from", "at", "is", "as", "towards", "toward", "into",
              "using", "based", "over", "under", "through", "without", "its"}


def _title_words(t: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", t.lower())) - _STOPWORDS


def make_fictional(real: list[dict], n: int, seed: int) -> list[dict]:
    """Fictional titles with NO real anchor: take a real title as a template
    (its stopwords, punctuation and shape) and replace about half its content
    words with content words drawn from other titles, then reject anything that
    shares a content-word 3-gram, or >= 30% of its words, with any real
    title. Earlier versions recombined real halves ("A: B") — on the live
    web the real half finds the real paper, and answering about it is
    reasonable, not fabrication. A mash-up has no such paper to find."""
    rng = random.Random(seed)
    sweep = [r["title"] for r in real if not r.get("famous") and 30 <= len(r["title"]) <= 110]
    pool = sorted({w for t in sweep for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", t)
                   if w.lower() not in _STOPWORDS})
    real_sets = [_title_words(r["title"]) for r in real]
    real_grams = set()
    for r in real:
        ws = [w for w in re.findall(r"[a-z0-9]+", r["title"].lower()) if w not in _STOPWORDS]
        real_grams.update(zip(ws, ws[1:], ws[2:]))
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 300:
        tries += 1
        tmpl = rng.choice(sweep)
        words = re.split(r"(\W+)", tmpl)
        new = []
        for tok in words:
            if (re.fullmatch(r"[A-Za-z][A-Za-z\-]{3,}", tok) and tok.lower() not in _STOPWORDS
                    and rng.random() < 0.55):
                w = rng.choice(pool)
                new.append(w[0].upper() + w[1:] if tok[0].isupper() else w.lower())
            else:
                new.append(tok)
        title = "".join(new).strip()
        if title in seen or len(title) < 25:
            continue
        ws = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in _STOPWORDS]
        if len(ws) < 3 or any(g in real_grams for g in zip(ws, ws[1:], ws[2:])):
            continue
        tw = set(ws)
        if max((len(tw & rs) / len(tw | rs) for rs in real_sets if rs), default=0.0) >= 0.3:
            continue
        seen.add(title)
        out.append({"id": f"fictional_{len(out):04d}", "title": title, "authors": [],
                    "published": None, "categories": [], "fictional": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/arxiv_snapshot.jsonl")
    ap.add_argument("--from", dest="start", default="2023-01")
    ap.add_argument("--to", dest="end", default=date.today().strftime("%Y-%m"))
    ap.add_argument("--per-month", type=int, default=40)
    ap.add_argument("--category", default="cs.LG")
    ap.add_argument("--fictional", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--refictional", action="store_true",
                    help="keep the real rows already in --out and only "
                         "regenerate the fictional slice (no network)")
    args = ap.parse_args()

    if args.refictional:
        real = [json.loads(l) for l in Path(args.out).read_text().splitlines()
                if l.strip() and not json.loads(l).get("fictional")]
        fictional = make_fictional(real, args.fictional, args.seed)
        with Path(args.out).open("w") as f:
            for r in real + fictional:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"rewrote {len(real)} real + {len(fictional)} fictional -> {args.out}")
        return

    rows: dict[str, dict] = {}
    print(f"famous: {len(FAMOUS_IDS)} ids", flush=True)
    for i in range(0, len(FAMOUS_IDS), 25):
        chunk = FAMOUS_IDS[i:i + 25]
        for r in _query({"id_list": ",".join(chunk), "max_results": len(chunk)}):
            r["famous"] = True
            rows[r["id"]] = r
        time.sleep(SLEEP_S)
    print(f"  fetched {len(rows)}", flush=True)

    for y, m in _months(args.start, args.end):
        lo = f"{y:04d}{m:02d}010000"
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        hi = f"{ny:04d}{nm:02d}010000"
        q = f"cat:{args.category} AND submittedDate:[{lo} TO {hi}]"
        got = _query({"search_query": q, "start": 0,
                      "max_results": args.per_month,
                      "sortBy": "submittedDate", "sortOrder": "ascending"})
        n_new = 0
        for r in got:
            if r["id"] not in rows:
                r["famous"] = False
                rows[r["id"]] = r
                n_new += 1
        print(f"  {y:04d}-{m:02d}: {len(got)} hits, {n_new} new", flush=True)
        time.sleep(SLEEP_S)

    real = list(rows.values())
    fictional = make_fictional(real, args.fictional, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in real + fictional:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(real)} real + {len(fictional)} fictional -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
