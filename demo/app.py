#!/usr/bin/env python3
# lifecycle: core
"""Public side-by-side demo: base Qwen3.6-35B vs the C-200 adapter + system prompt.

Serves one page (demo/page.html) plus a "duel" endpoint that runs the SAME
prompt twice against the :8084 lens host — arm A the pristine base weights,
arm B the calibrated-honesty LoRA with the system prompt it is trained to
respond to — and streams both back over one SSE response. The lens host
swaps LoRA per request, so both arms share one resident model.

Two optional modes, both visitor-toggled (defaults OFF, so the public
resting behaviour is exactly the single-turn no-tools duel it always was):

  multi-turn   the browser replays prior turns, so each arm keeps its OWN
               transcript (they diverge from turn one). Answers whether the
               trained hedging survives a conversation.
  tools        offers the model a web_search function in its NATIVE tool
               format and CLOSES the loop: the call is parsed, executed
               against the arXiv API, and fed back as a tool message. Shows
               "checks when it can" next to "hedges when it can't".

Public boundary (exposed at rl.strawrunway.com via the strawrunway tunnel),
so the guardrails live here:
  * prompt length cap, fixed max_tokens/temperature (no client knobs)
  * per-IP token bucket + one duel in flight globally
  * adapter list is hardcoded — no pass-through
  * visitor feedback appends to demo/flags.jsonl (gitignored), size-capped
  * transcript replay is CAPPED and RESHAPED server-side (roles are rebuilt
    from {user,base,rl} triples), so a client cannot post arbitrary role
    sequences into the template
  * the tool is an allowlist of ONE: fixed host+path (export.arxiv.org
    /api/query), the visitor's text reaches it only as a urlencoded query
    value, short timeout, capped rounds and result size. This is the only
    outbound network the demo makes, and only when the box is ticked.

If the resting backend ever injects the honesty prompt at the serving proxy
(lens+c200 default), point --upstream at the INNER server port instead of
:8084 — otherwise the proxy adds the prompt to BOTH arms and the base arm
stops being a baseline.

Run:  python3 demo/app.py [--port 8092] [--upstream http://127.0.0.1:8084]
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "page.html"
FLAGS = ROOT / "flags.jsonl"
# The prompt the adapter is trained to respond to; ships in the adapter dir.
SYSTEM_PROMPT_PATH = (
    Path.home() / "models/adapters/qa-gloveC-200-20260731/GLOVE.txt")
ADAPTER = "c200"

MAX_PROMPT_CHARS = 400
MAX_TOKENS = 192          # the training horizon
TEMPERATURE = 0.7
RATE_PER_MIN = 4
RATE_BURST = 2
FLAGS_MAX_BYTES = 5 * 2**20

MAX_TURNS = 6             # replayed exchanges per arm
MAX_REPLY_CHARS = 1200    # per replayed assistant turn
MAX_TOOL_ROUNDS = 2       # call -> result -> answer, then stop
ARXIV_HOST = "export.arxiv.org"
ARXIV_PATH = "/api/query"
ARXIV_TIMEOUT = 12
ARXIV_MAX_RESULTS = 4
TOOL_RESULT_CHARS = 1500

# Appended to the honesty prompt ONLY when tools are offered, so the no-tools
# duel stays byte-identical to the shipped adapter + prompt pair.
#
# Why it exists: the trained prompt says nothing about tools, because training
# offered none (single-turn, 192 tokens, no tools). Measured on 6 real
# post-cutoff arXiv papers x 3 phrasings x 3 samples, how often each arm
# reached for search_arxiv (1.00 = always):
#
#     phrasing                  base  prompt  prompt+this clause
#     "the arXiv paper 'X'"     1.00    1.00        1.00
#     "the 2026 paper 'X'"      0.83    0.06        0.78
#     same, invented title      0.67    0.08        0.42
#
# The prompt does NOT suppress tools in general — it collapses only when the
# question asserts a year the model reads as future, which hands it grounds
# to conclude non-existence and stop. This clause restores the search and
# takes asserted-nonexistence to 0.00 in both year conditions.
TOOL_FIRST = (
    "When tools are available, use them before declining. If a tool could "
    "resolve the question, call it rather than telling the user to look it "
    "up themselves. Decline only when no tool can help, or after a tool has "
    "come back empty."
)

# Kept in sync with mlx_rl/tasks/qa_arxiv.py (the trainer's copy is the
# source of truth); duplicated here because the demo is stdlib-only.
DATE_LINE = "Today's date is {today}."

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns a numbered list of results "
                       "with title, URL, date and a short snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}
# The demo's web_search is backed by the arXiv API only (fixed host, see
# arxiv_search) — the tool SHAPE is the general one the adapter is trained
# with; the backend is the one this demo can offer without keys.

# The model's native emission format (see the qwen3 chat template): an
# inner <function=NAME> block nested in <tool_call> tags.
_FUNC_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>",
                      re.S)
_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*"
                       r"</parameter>", re.S)


def parse_tool_call(text: str):
    """-> (name, {param: value}) for the first call, else None."""
    m = _FUNC_RE.search(text)
    if not m:
        return None
    return m.group(1), {k: v for k, v in _PARAM_RE.findall(m.group(2))}


def _arxiv_query(search_query: str):
    """One call to the fixed endpoint. -> list of formatted hits, or None on
    a transport/parse failure (distinct from an honest zero-hit result)."""
    url = "https://{}{}?{}".format(
        ARXIV_HOST, ARXIV_PATH,
        urllib.parse.urlencode({"search_query": search_query, "start": 0,
                                "max_results": ARXIV_MAX_RESULTS}))
    try:
        with urllib.request.urlopen(url, timeout=ARXIV_TIMEOUT) as r:
            raw = r.read(400_000)
        root = ET.fromstring(raw)
    except Exception:  # noqa: BLE001 — transport or XML, same handling
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    hits = []
    for entry in root.findall("a:entry", ns):
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        published = (entry.findtext("a:published", "", ns) or "")[:10]
        aid = (entry.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        authors = [" ".join((a.findtext("a:name", "", ns) or "").split())
                   for a in entry.findall("a:author", ns)]
        snippet = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        hits.append("{}\n   https://arxiv.org/abs/{} · {} · {}".format(
            title, aid, published, snippet or "unlisted"))
    return hits


def arxiv_search(query: str) -> str:
    """The one tool. Fixed host and path — the visitor's text can only ever
    become a urlencoded query VALUE, never a host, path or scheme.

    Title-scoped first: these questions ARE paper titles, and a bare `all:`
    search buries the exact paper under keyword matches (it put "Attention
    Is All You Need" 4th, behind three papers that merely echo the meme).
    `all:` is the fallback, so a title that genuinely isn't on arXiv still
    gets a real search behind it before we report nothing."""
    q = query.strip()[:300]
    if not q:
        return "No query supplied."
    quoted = q.replace('"', " ").strip()
    hits = _arxiv_query('ti:"{}"'.format(quoted)) if quoted else None
    if not hits:
        fallback = _arxiv_query('all:"{}"'.format(quoted) if quoted
                                else "all:" + q)
        if fallback is None and hits is None:
            return "Error: search service unavailable (timed out)."
        hits = fallback or []
    if not hits:
        return 'No results found for "{}".'.format(q[:120])
    return "\n".join("{}. {}".format(i, h) for i, h in enumerate(hits, 1))[:TOOL_RESULT_CHARS]

_duel_lock = threading.Semaphore(1)
_buckets: dict[str, list[float]] = {}
_buckets_lock = threading.Lock()


def _rate_ok(ip: str) -> bool:
    now = time.monotonic()
    with _buckets_lock:
        tokens, last = _buckets.get(ip, [float(RATE_BURST), now])
        tokens = min(RATE_BURST, tokens + (now - last) * RATE_PER_MIN / 60.0)
        if tokens < 1.0:
            _buckets[ip] = [tokens, now]
            return False
        _buckets[ip] = [tokens - 1.0, now]
        return True


def make_handler(upstream: str):
    up = urlparse(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _client_ip(self) -> str:
            xff = self.headers.get("X-Forwarded-For")
            return xff.split(",")[0].strip() if xff \
                else self.client_address[0]

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ------------------------------------------------------------ GET
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/status":
                self._json(200, self._status())
            else:
                self._json(404, {"error": "not found"})

        def _status(self) -> dict:
            """ready iff the lens host answers AND serves the c200 adapter
            AND the system prompt is readable. Anything else -> the page shows
            its offline banner (the model slot is busy with research)."""
            if not SYSTEM_PROMPT_PATH.exists():
                return {"ready": False, "reason": "system prompt file missing"}
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port,
                                                  timeout=3)
                conn.request("GET", "/health")
                r = conn.getresponse()
                h = json.loads(r.read()) if r.status == 200 else {}
                conn.close()
            except Exception:
                return {"ready": False, "reason": "model slot offline"}
            if ADAPTER not in (h.get("available_adapters") or []):
                return {"ready": False,
                        "reason": "a different model holds the slot"}
            return {"ready": True, "model": h.get("model")}

        # ------------------------------------------------------------ POST
        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/flag":
                return self._do_flag()
            if path != "/api/duel":
                return self._json(404, {"error": "not found"})
            ip = self._client_ip()
            if not _rate_ok(ip):
                return self._json(429, {"error":
                    "rate limited — wait a few seconds"})
            try:
                req = json.loads(self.rfile.read(
                    min(int(self.headers.get("Content-Length", 0)), 65536)))
            except Exception:
                return self._json(400, {"error": "bad json"})
            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                return self._json(400, {"error": "empty prompt"})
            if len(prompt) > MAX_PROMPT_CHARS:
                return self._json(400, {"error":
                    f"prompt too long (max {MAX_PROMPT_CHARS} chars)"})
            turns = self._clean_turns(req.get("turns"))
            use_tools = bool(req.get("tools"))
            if not self._status().get("ready"):
                return self._json(503, {"error": "demo offline"})
            if not _duel_lock.acquire(blocking=False):
                return self._json(503, {"error":
                    "a duel is already running — try again in ~30s"})
            try:
                self._duel(prompt, turns, use_tools)
            finally:
                _duel_lock.release()

        @staticmethod
        def _clean_turns(raw) -> list:
            """Rebuild prior exchanges from {user, base, rl} triples.

            The client never gets to name roles: we read three strings per
            turn and construct the message list ourselves, so no visitor can
            post a 'system' turn or an arbitrary role sequence into the
            template (which raises on unknown roles)."""
            if not isinstance(raw, list):
                return []
            out = []
            for t in raw[-MAX_TURNS:]:
                if not isinstance(t, dict):
                    continue
                user = str(t.get("user", ""))[:MAX_PROMPT_CHARS].strip()
                if not user:
                    continue
                out.append({
                    "user": user,
                    "base": str(t.get("base", ""))[:MAX_REPLY_CHARS],
                    "rl": str(t.get("rl", ""))[:MAX_REPLY_CHARS],
                })
            return out

        def _duel(self, prompt: str, turns: list, use_tools: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
            if use_tools:
                system_prompt = system_prompt + " " + TOOL_FIRST
            # The date is rendered at request time, never hardcoded: a stale
            # date fails exactly like the bug the date is there to fix.
            system_prompt = system_prompt + " " + DATE_LINE.format(today=time.strftime("%Y-%m-%d"))
            arms = [("base", None, None), ("rl", system_prompt, ADAPTER)]
            try:
                for arm, system, adapter in arms:
                    self._emit({"arm": arm, "start": True})
                    msgs = ([{"role": "system", "content": system}]
                            if system else [])
                    for t in turns:
                        msgs.append({"role": "user", "content": t["user"]})
                        msgs.append({"role": "assistant",
                                     "content": t[arm] or "(no reply)"})
                    msgs.append({"role": "user", "content": prompt})
                    ok = self._run_arm(arm, msgs, adapter, use_tools)
                    self._emit({"arm": arm, "done": True, "ok": ok})
                self._emit({"all_done": True})
            except BrokenPipeError:
                pass  # visitor left; upstream sees the close

        def _run_arm(self, arm: str, msgs: list, adapter, use_tools: bool):
            """Stream the arm, then close the tool loop if it called one.

            Rounds are capped, so a model that loops on tool calls costs a
            bounded number of generations."""
            tools = [SEARCH_TOOL] if use_tools else None
            for _ in range(MAX_TOOL_ROUNDS if use_tools else 1):
                ok, text = self._stream_once(arm, msgs, adapter, tools)
                if not ok:
                    return False
                call = parse_tool_call(text) if use_tools else None
                if not call:
                    return True
                name, params = call
                if name != SEARCH_TOOL["function"]["name"]:
                    result = "Error: unknown tool '{}'.".format(name)
                else:
                    query = params.get("query", "")
                    self._emit({"arm": arm, "tool_call": name,
                                "query": query[:300]})
                    result = arxiv_search(query)
                self._emit({"arm": arm, "tool_result": result})
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "tool", "content": result}]
            # Out of rounds with a call still pending: say so rather than
            # letting the transcript end on an unanswered tool call.
            self._emit({"arm": arm, "note": "tool-round cap reached"})
            return True

        def _emit(self, obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def _stream_once(self, arm: str, msgs: list, adapter,
                         tools) -> tuple:
            """Stream one completion. -> (ok, full_text) so the caller can
            look for a tool call in what was just streamed."""
            payload = {
                "model": "qwen36-lens", "messages": msgs,
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "stream": True, "enable_thinking": False, "adapter": adapter,
            }
            if tools:
                payload["tools"] = tools
            acc, fin_len = [], False
            try:
                conn = http.client.HTTPConnection(up.hostname, up.port,
                                                  timeout=600)
                conn.request("POST", "/v1/chat/completions",
                             body=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    r.read()
                    conn.close()
                    self._emit({"arm": arm, "error": f"backend {r.status}"})
                    return False, ""
                while True:
                    line = r.readline()
                    if not line:
                        break
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        choice = json.loads(data)["choices"][0]
                    except Exception:
                        continue
                    if choice.get("finish_reason") == "length":
                        fin_len = True
                    text = choice.get("delta", {}).get("content")
                    if text:
                        acc.append(text)
                        self._emit({"arm": arm, "delta": text})
                    # The template tells it to emit a call with NO suffix;
                    # both arms ignore that and ramble on (the base arm
                    # denied a real paper AFTER calling search on it, and
                    # the RL arm emitted the same call twice). Stop at the
                    # close tag: everything after it is generated blind,
                    # before any tool result exists.
                    if tools and "</tool_call>" in "".join(acc):
                        break
                conn.close()
                # Truncation is never silent: a reply that hit the cap is
                # labelled in the UI rather than read as the model stopping.
                if fin_len:
                    self._emit({"arm": arm, "truncated": MAX_TOKENS})
                full = "".join(acc)
                if tools and "</tool_call>" in full:
                    full = full.split("</tool_call>")[0] + "</tool_call>"
                return True, full
            except BrokenPipeError:
                raise
            except Exception:
                self._emit({"arm": arm, "error": "backend unreachable"})
                return False, ""

        def _do_flag(self) -> None:
            """Visitor feedback — the edge-case harvest. Appends one line;
            contents are untrusted visitor data, size-capped, never executed."""
            ip = self._client_ip()
            if not _rate_ok(ip):
                return self._json(429, {"error": "rate limited"})
            try:
                req = json.loads(self.rfile.read(
                    min(int(self.headers.get("Content-Length", 0)), 16384)))
            except Exception:
                return self._json(400, {"error": "bad json"})
            verdict = str(req.get("verdict", ""))[:24]
            if verdict not in ("rl_wrong", "base_better", "both_wrong",
                               "interesting"):
                return self._json(400, {"error": "bad verdict"})
            if FLAGS.exists() and FLAGS.stat().st_size > FLAGS_MAX_BYTES:
                return self._json(503, {"error": "flag store full"})
            with FLAGS.open("a") as fh:
                fh.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "verdict": verdict,
                    "prompt": str(req.get("prompt", ""))[:MAX_PROMPT_CHARS],
                    "base": str(req.get("base", ""))[:2000],
                    "rl": str(req.get("rl", ""))[:2000],
                }) + "\n")
            return self._json(200, {"ok": True})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--upstream", default="http://127.0.0.1:8084")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(args.upstream))
    print(f"mlx-rl demo: 127.0.0.1:{args.port} -> {args.upstream}",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
