"""Exa.ai adapter — the evidence fetch behind three of the lanes.

Two calls:
  * answer(query)  → a grounded natural-language answer + citations
  * search(query)  → semantic web search returning result rows (title/url/text)

Docs: https://docs.exa.ai  ·  key = EXA_API_KEY in .env.
Cost (observed 2026-07): /answer ~$0.005/call, /search ~$0.007/call (neural + text).
"""
import json
import threading
import time

import config

from .base import RawItem, session
from .ratelimit import RateLimiter
from .retry import retry

_limiter = RateLimiter(5)          # 5 requests/second, shared
ANSWER_ENDPOINT = "https://api.exa.ai/answer"
SEARCH_ENDPOINT = "https://api.exa.ai/search"

# ── query log: every Exa query we send, appended to a JSONL file (gitignored) ──
LOG_PATH = config.ROOT / "data" / "exa_queries.jsonl"
_log_lock = threading.Lock()


def log_query(query, *, label=None, cost=None, request_id=None, n_citations=None):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "label": label, "query": query,
               "cost": cost, "request_id": request_id, "citations": n_citations}
        with _log_lock:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _headers():
    return {"x-api-key": config.require("EXA_API_KEY"), "Content-Type": "application/json"}


_OUT_OF_CREDIT = False
# Per-process answer cache: the SAME query is asked once and reused across lanes,
# so this cuts Exa calls. One process = one server, entries are tiny.
_ANS_CACHE = {}
_cache_lock = threading.Lock()


@retry(attempts=4, base_delay=1.0)
def answer(query: str, *, text: bool = True, label: str = None) -> dict:
    """Grounded answer. Returns {answer, citations:[{title,url,text}], cost}.
    Cached by query. Degrades gracefully if Exa is out of credit (402)."""
    global _OUT_OF_CREDIT
    if _OUT_OF_CREDIT:
        return {"answer": "", "citations": [], "cost": 0, "_exa_skipped": True}
    with _cache_lock:
        hit = _ANS_CACHE.get(query)
    if hit is not None:
        return {**hit, "cost": 0, "_cached": True}     # reused → no extra cost
    _limiter.acquire()
    try:
        r = session().post(ANSWER_ENDPOINT, headers=_headers(),
                           json={"query": query, "text": text}, timeout=60)
    except Exception:
        return {"answer": "", "citations": [], "cost": 0, "_exa_skipped": True}
    if r.status_code == 402:                       # out of credit → stop calling Exa this run
        _OUT_OF_CREDIT = True
        log_query(query, label=label, cost=0, request_id="OUT_OF_CREDIT")
        return {"answer": "", "citations": [], "cost": 0, "_exa_skipped": True}
    if r.status_code >= 500 or r.status_code == 429:   # transient Exa error → degrade, don't crash
        log_query(query, label=label, cost=0, request_id=f"HTTP_{r.status_code}")
        return {"answer": "", "citations": [], "cost": 0, "_exa_skipped": True}
    r.raise_for_status()
    d = r.json()
    out = {
        "answer": d.get("answer") or "",
        "citations": [{"title": c.get("title", ""), "url": c.get("url", ""),
                       "text": c.get("text", "")} for c in (d.get("citations") or [])],
        "cost": (d.get("costDollars") or {}).get("total"),
    }
    log_query(query, label=label, cost=out["cost"], request_id=d.get("requestId"),
              n_citations=len(out["citations"]))
    with _cache_lock:
        _ANS_CACHE[query] = {"answer": out["answer"], "citations": out["citations"]}
    return out


@retry(attempts=4, base_delay=1.0)
def search(query: str, *, n: int = 5, max_chars: int = 800,
           start_published: str = None, end_published: str = None,
           highlights: bool = False, search_type: str = "auto",
           user_location: str = None) -> list:
    """Semantic search. Returns list[RawItem].
    highlights=True: Exa extracts query-relevant sentences instead of raw text.
    search_type: 'auto' | 'neural' | 'keyword'.
    user_location: ISO country code (e.g. 'HR') to bias results."""
    _limiter.acquire()
    contents = {"highlights": True} if highlights else {"text": {"maxCharacters": max_chars}}
    body = {"query": query, "numResults": n, "type": search_type, "contents": contents}
    if start_published:
        body["startPublishedDate"] = start_published
    if end_published:
        body["endPublishedDate"] = end_published
    if user_location:
        body["userLocation"] = user_location
    r = session().post(SEARCH_ENDPOINT, headers=_headers(), json=body, timeout=60)
    r.raise_for_status()
    out = []
    for x in (r.json().get("results") or []):
        if highlights:
            content = "\n".join(x.get("highlights") or [])
        else:
            content = x.get("text", "")
        out.append(RawItem(url=x.get("url", ""), title=x.get("title", ""),
                           content=content, postdate=x.get("publishedDate", ""),
                           author=x.get("author", "") or ""))
    return out
