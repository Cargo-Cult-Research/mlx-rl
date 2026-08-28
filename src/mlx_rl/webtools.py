"""Real web tools for tool-using tasks: `web_search` (DuckDuckGo) and
`fetch_url` (HTTP GET, browser User-Agent, HTML reduced to text).

Train like you serve — these are the same tools the deployed assistant gets
(the shape and backend Moss used, `moss/scripts/tools.py`), and the web is
noisy on purpose: near-misses, SEO junk, paywalls, timeouts. Making sense of
that is what the policy is being trained to do; nothing here cleans it up
beyond what a real harness would (HTML -> visible text, size caps).

Two things a training loop needs on top of the served tools:

* **A cache.** Group members re-issue the same query (the paper title) and
  training re-visits prompts across steps, so results are cached on disk by
  exact query / URL (`runs/webcache/`). First sight goes live; afterwards
  the run sees the frozen answer — reproducible after first sight, like the
  judge cache, and it keeps live volume to a fraction of the ~80 calls a
  step would otherwise make. Errors and EMPTY result pages are cached only
  briefly (an empty page for a famous paper is usually throttling, not
  truth), so a rate-limited engine is not hammered but a bad first answer
  is not frozen either.
* **A fetch blocklist.** The policy chooses the URLs. Anything that resolves
  to loopback / RFC1918 / link-local / the tailnet CGNAT range is refused —
  this box serves private pages on those addresses. http(s) only, size and
  time capped, no redirects to blocked ranges.

Live calls are paced (DuckDuckGo rate-limits bursts) and each one runs
under a HARD wall-clock timeout in a helper thread: the HTTP client has been
seen to hang past its own timeout, and a hung call must never hold anything
another row needs. Nothing here holds a lock across network I/O.
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns a numbered list of results "
                       "with title, URL and a short snippet.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}
FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch a web page and return its visible text "
                       "(truncated). Use for a URL from search results or "
                       "one the user gave.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL, with scheme."}},
            "required": ["url"],
        },
    },
}
TOOLS = [WEB_SEARCH_TOOL, FETCH_URL_TOOL]

_BLOCKED_NETS = [ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "100.64.0.0/10", "0.0.0.0/8", "::1/128", "fc00::/7",
    "fe80::/10")]


def _blocked_host(host: str) -> str | None:
    """Reason string if the host resolves into a blocked range, else None."""
    if not host or host.lower() in ("localhost",) or host.endswith(".local"):
        return "local host"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return "unresolvable host"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if any(ip in n for n in _BLOCKED_NETS):
            return "private address"
    return None


_TAG_RE = re.compile(r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>", re.S | re.I)
_BR_RE = re.compile(r"</(p|div|br|li|tr|h[1-6]|section|article|blockquote|pre)>|<br\s*/?>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")


_STOP = frozenset(
    "a an the of for and or to in on with by using via is are was were we our "
    "us this that these those from at as its it new how what when where which "
    "who why do does can could would should".split())


def _content_words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - _STOP


def _render(results: list[dict], limit: int) -> str:
    lines = [f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}"
             for i, r in enumerate(results, 1)]
    return "\n".join(lines)[:limit]


def relevance(query: str, results: list[dict]) -> float:
    """How much of the QUERY the best result echoes back, in [0, 1].

    Deliberately query-only. Scoring against the gold ANSWER would make the
    tool an oracle -- "the search agreed with the grader" and "the policy
    checked" would be the same event, which is the defect that sank
    pypi_lookup. Relevance to the query is information the caller already
    has, so using it leaks nothing: it is the same judgement a person makes
    glancing at a result page before reading it.

    Content words only, title + snippet. A search for "Delta Score:
    Improving the Binding Assessment of Structure-Based Drug Design Methods"
    that comes back with "Delta Air Lines" scores ~0.02; the real paper
    scores ~1.0. Nothing in between needs to be adjudicated precisely -- the
    gate is there to catch the wholesale misses, not to rank.
    """
    qw = _content_words(query)
    if len(qw) < 3:          # "python", "IPv6" -- any page is arguably on topic
        return 1.0
    best = 0.0
    for r in results:
        hw = _content_words(f"{r.get('title', '')} {r.get('body', '')}")
        if hw:
            best = max(best, len(qw & hw) / len(qw))
    return best


def html_to_text(raw: str) -> str:
    t = _TAG_RE.sub(" ", raw)
    t = _BR_RE.sub("\n", t)
    t = _ANY_TAG.sub(" ", t)
    t = html.unescape(t)
    lines = [" ".join(l.split()) for l in t.splitlines()]
    return "\n".join(l for l in lines if l)


class WebTools:
    """web_search / fetch_url with an on-disk cache and pacing."""

    # Engines tried IN ORDER per query, first non-empty result wins. ddgs'
    # backend="auto" fans out to every engine and surfaces the slowest
    # failure (google/yandex refuse the connection outright from here);
    # measured 2026-08-16 evening: bing/yahoo answer in <2 s, duckduckgo
    # throttles after a few hundred anonymous calls. Anonymous scraping is
    # weather, not infrastructure — a search API key is the stable answer.
    ENGINES = ("bing", "yahoo", "duckduckgo", "brave")

    def __init__(self, cache_dir: str | Path = "runs/webcache", max_results: int = 5,
                 fetch_chars: int = 4000, search_chars: int = 2500,
                 min_interval_s: float = 1.0, timeout_s: float = 10.0,
                 error_ttl_s: float = 600.0, engines: tuple[str, ...] | None = None,
                 min_relevance: float = 0.5):
        self.engines = tuple(engines) if engines else self.ENGINES
        self.min_relevance = min_relevance
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_results = max_results
        self.fetch_chars, self.search_chars = fetch_chars, search_chars
        self.min_interval, self.timeout, self.error_ttl = min_interval_s, timeout_s, error_ttl_s
        self._lock = threading.Lock()   # cache + pacing bookkeeping only
        self._last = 0.0
        self.hard_timeout = timeout_s + 5.0
        self.search_budget_s = 2.5 * timeout_s   # total across the engine rotation
        # In-flight de-duplication: eight group members asking the same
        # title at once should cost ONE live call, not eight cache misses.
        self._inflight: dict[str, threading.Event] = {}
        self.stats = {"search_live": 0, "search_hit": 0, "fetch_live": 0,
                      "fetch_hit": 0, "errors": 0, "search_irrelevant": 0}

    def _claim(self, kind: str, key: str):
        """-> (cached_result_or_None, event_to_set_or_None). If another
        thread is already fetching this key, wait for it and return its
        cached result."""
        k = f"{kind}:{key}"
        while True:
            with self._lock:
                hit = self._get(kind, key)
                if hit is not None:
                    return hit, None
                ev = self._inflight.get(k)
                if ev is None:
                    ev = self._inflight[k] = threading.Event()
                    return None, ev
            ev.wait(self.hard_timeout + 5)

    def _release(self, kind: str, key: str, ev: threading.Event, d: dict) -> None:
        with self._lock:
            self._put(kind, key, d)
            self._inflight.pop(f"{kind}:{key}", None)
        ev.set()

    # -- cache -------------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.dir / kind / h[:2] / f"{h}.json"

    def _get(self, kind: str, key: str):
        p = self._path(kind, key)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except Exception:
            return None
        # Three things are provisional, not answers, and must not be frozen:
        # a hard error, an empty page (usually throttling, not truth), and a
        # result set no engine could make relevant to the query. All three
        # expire on error_ttl so a later run re-asks. A genuine hit is frozen,
        # which is what makes a run reproducible after first sight.
        empty = d.get("ok") and kind == "search" and not d.get("results")
        # Entries written before the gate existed (2026-08-26) carry no verdict,
        # so score them on read. 88.6% of the 41,966 cached successes at that
        # point were off-topic; migrating the files would work too, but scoring
        # here also covers any cache restored from an older backup.
        if kind == "search" and "relevant" not in d and d.get("results"):
            d["relevant"] = relevance(d.get("key", ""), d["results"]) >= self.min_relevance
        offtopic = kind == "search" and d.get("relevant") is False
        stale = time.time() - d.get("t", 0) > self.error_ttl
        if (not d.get("ok") or empty or offtopic) and stale:
            return None
        return d

    def _put(self, kind: str, key: str, d: dict) -> None:
        p = self._path(kind, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = dict(d, t=time.time(), key=key)
        p.write_text(json.dumps(d, ensure_ascii=False))

    def _pace(self) -> None:
        """Reserve the next start slot: slots are min_interval apart, and a
        caller sleeps until its slot. (Earlier version advanced the slot by
        the WAIT instead of the interval — a runaway backlog under load,
        every call waited longer than the last; caught arm 1 v4 step 8.)"""
        with self._lock:
            now = time.time()
            slot = max(self._last + self.min_interval, now)
            self._last = slot
        if slot > now:
            time.sleep(slot - now)

    def _run_hard(self, fn, *args):
        """Run fn in a helper thread with a wall-clock cap. On timeout the
        helper is abandoned (daemon) and TimeoutError is raised — the caller
        caches an error and moves on. A hung HTTP call must never block a
        training batch."""
        box: dict = {}

        def _go():
            try:
                box["r"] = fn(*args)
            except BaseException as e:  # noqa: BLE001
                box["e"] = e
        th = threading.Thread(target=_go, daemon=True)
        th.start()
        th.join(self.hard_timeout)
        if th.is_alive():
            raise TimeoutError(f"hung > {self.hard_timeout:.0f}s")
        if "e" in box:
            raise box["e"]
        return box["r"]

    # -- tools -------------------------------------------------------------
    def web_search(self, query: str) -> dict:
        """-> {"ok", "text", "results": [{"title","href","body"}], "cached"}"""
        q = " ".join(query.split())[:300]
        if not q:
            return {"ok": False, "text": "Error: 'query' is required.", "results": []}
        hit, ev = self._claim("search", q)
        if hit is not None:
            with self._lock:
                self.stats["search_hit"] += 1
            return dict(hit, cached=True)
        try:
            self._pace()
            with self._lock:
                self.stats["search_live"] += 1
            d = self._search_live(q)
        except BaseException:  # noqa: BLE001 — never leave waiters hanging
            d = {"ok": False, "results": [], "error": "internal", "text": "Error: search failed. Try again later."}
            self._release("search", q, ev, d)
            raise
        self._release("search", q, ev, d)
        return dict(d, cached=False)

    def _search_live(self, q: str) -> dict:
        from ddgs import DDGS
        errors = []
        best: dict | None = None      # best off-topic set seen, if nothing passes
        t_start = time.time()
        for engine in self.engines:
            if time.time() - t_start > self.search_budget_s:
                errors.append(f"{engine}: skipped (search budget {self.search_budget_s:.0f}s spent)")
                continue
            try:
                hits = list(self._run_hard(
                    lambda: DDGS(timeout=self.timeout).text(
                        q, max_results=self.max_results, backend=engine)))
            except Exception as e:  # noqa: BLE001 — throttled / refused / hung
                msg = str(e).strip().replace("\n", " ")[:60]
                errors.append(f"{engine}: {type(e).__name__}: {msg}")
                continue  # "No results found" from ddgs is an exception too: next engine
            results = [{"title": h.get("title", ""), "href": h.get("href", ""),
                        "body": h.get("body", "")} for h in hits]
            if results:
                rel = relevance(q, results)
                d = {"ok": True, "results": results, "engine": engine,
                     "relevance": round(rel, 3),
                     "text": _render(results, self.search_chars)}
                if rel >= self.min_relevance:
                    return d
                # Off-topic. NOT a hit: the engine answered a different
                # question (a query for a paper title coming back with an
                # airline). Keep the best one seen so we still hand back
                # something real, and let the next engine try -- the ones
                # that rank titles correctly sit behind the ones that merely
                # answer fastest.
                errors.append(f"{engine}: off-topic (relevance {rel:.2f})")
                if best is None or rel > best.get("relevance", 0.0):
                    best = d
        # Nothing passed the gate. Hand back the best off-topic set anyway --
        # real search returns junk and reading past it is part of the job, so
        # the policy should see what a real caller would. What changes is the
        # CACHE: `relevant: False` gives it the short error TTL instead of
        # being frozen forever, so the next run re-asks rather than inheriting
        # today's weather as permanent truth.
        if best is not None:
            with self._lock:
                self.stats["search_irrelevant"] += 1
            return dict(best, relevant=False)
        # Every engine came back empty or failed. Only "no results" from
        # every engine is an honest empty; any hard failure is an error.
        if errors and all("No results" in e or "no results" in e.lower() for e in errors):
            return {"ok": True, "results": [], "text": f'No results found for "{q}".'}
        with self._lock:
            self.stats["errors"] += 1
        return {"ok": False, "results": [], "error": " | ".join(errors)[:300],
                "text": "Error: search failed. Try again later."}

    def fetch_url(self, url: str) -> dict:
        """-> {"ok", "text", "status", "cached"}"""
        u = url.strip()
        parts = urlsplit(u)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return {"ok": False, "text": "Error: a full http(s) URL is required."}
        why = _blocked_host(parts.hostname)
        if why:
            return {"ok": False, "text": f"Error: cannot fetch that URL ({why})."}
        hit, ev = self._claim("fetch", u)
        if hit is not None:
            with self._lock:
                self.stats["fetch_hit"] += 1
            return dict(hit, cached=True)
        try:
            self._pace()
            with self._lock:
                self.stats["fetch_live"] += 1
            d = self._fetch_live(u)
        except BaseException:  # noqa: BLE001
            d = {"ok": False, "error": "internal", "text": "Error: fetch failed."}
            self._release("fetch", u, ev, d)
            raise
        self._release("fetch", u, ev, d)
        return dict(d, cached=False)

    def _fetch_live(self, u: str) -> dict:
        def _get_page():
            import requests
            r = requests.get(u, headers=_BROWSER_HEADERS, timeout=self.timeout,
                             allow_redirects=True, stream=True)
            final = urlsplit(r.url).hostname or ""   # redirects may land on a blocked host
            if _blocked_host(final):
                r.close()
                return {"ok": False, "text": "Error: cannot fetch that URL (private address)."}
            raw = r.raw.read(1_500_000, decode_content=True)
            r.close()
            body = raw.decode(r.encoding or "utf-8", errors="replace")
            ctype = r.headers.get("Content-Type", "")
            text = html_to_text(body) if "html" in ctype or body.lstrip().startswith("<") else body
            if len(text) > self.fetch_chars:
                text = text[: self.fetch_chars] + "\n...[truncated]"
            return {"ok": r.status_code < 400, "status": r.status_code,
                    "text": text if r.status_code < 400 else f"Error: HTTP {r.status_code} for {u}"}

        try:
            d = self._run_hard(_get_page)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.stats["errors"] += 1
            d = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}",
                 "text": f"Error: fetch failed ({type(e).__name__})."}
        return d
