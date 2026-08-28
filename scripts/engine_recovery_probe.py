"""Does an anonymous-scraping block on brave/duckduckgo ever lift?

One query per engine every 10 minutes, logged. The question this settles:
whether a slow background crawl (option 1 -- one query a minute for days) is
viable at all. It is only viable if the block is a rate limit that decays;
if it is an IP or fingerprint ban, waiting buys nothing and the crawl would
never finish regardless of how gently it is paced.

Deliberately one query per engine per cycle: probing harder is what created
the block being measured.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ddgs import DDGS  # noqa: E402

Q = "RoFormer: Enhanced Transformer with Rotary Position Embedding"
LOG = Path("runs/engine-recovery.jsonl")
LOG.parent.mkdir(parents=True, exist_ok=True)

for cycle in range(72):                      # 12 hours at 10-minute spacing
    rec = {"t": time.strftime("%m-%d %H:%M"), "cycle": cycle}
    for e in ("brave", "duckduckgo", "bing", "yahoo"):
        try:
            hits = list(DDGS(timeout=25).text(Q, max_results=5, backend=e))
            top = hits[0]["title"][:60] if hits else ""
            rec[e] = {"ok": True, "n": len(hits), "top": top,
                      "relevant": "roformer" in " ".join(
                          h["title"].lower() for h in hits)}
        except Exception as ex:              # noqa: BLE001
            rec[e] = {"ok": False, "err": f"{type(ex).__name__}: {ex}"[:70]}
        time.sleep(5)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["t"], {k: (v.get("relevant") if v.get("ok") else "blocked")
                     for k, v in rec.items() if isinstance(v, dict)}, flush=True)
    time.sleep(600 - 20)
