"""Confidence Check — live backend.

One claim in → structure it (Gemini) → fan out to independent evidence lanes IN PARALLEL
(Exa answer, Exa search, SerpAPI, Parallel.ai deep research) → each lane is judged GROUNDED
against what it actually fetched → aggregate into a verdict + calibrated-ish confidence.

Streams every step over SSE so the UI can show lanes landing live.

Run:  python3 server.py        → http://localhost:8899
Keys: copy .env.example to .env and fill in what you have. A lane whose key is
missing reports an error and the check continues on the remaining lanes —
GEMINI_API_KEY + EXA_API_KEY is the working minimum.
"""
import json
import os
import queue
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config                                    # noqa: E402  (loads .env from the repo root)
import gemini                                    # noqa: E402  (self-contained Gemini helper)
from adapters import exa                         # noqa: E402
from adapters import parallel as parallel_ai     # noqa: E402

config.load()          # populate credentials into os.environ before anything reads them
PORT = int(os.environ.get("PORT", 8899))

# ── lane weights: how much a lane's judgement counts in the aggregate ──────────
# Deep research (own citations + per-field basis) > grounded answer > raw snippets.
# Lane ids/labels are deliberately vendor-neutral — they surface in the UI, the SSE
# stream and the MCP payload, and we don't expose our providers to clients.
WEIGHTS = {"deep_research": 1.35, "web_grounded": 1.15, "web_semantic": 0.9,
           "web_serp": 0.85, "market": 1.25, "research": 1.4}


# ══════════════════════════════════════════════════════════════════════════════
# 1. STRUCTURE — turn loose natural language into something checkable
# ══════════════════════════════════════════════════════════════════════════════
def today():
    """Every prompt gets the real date. Without it the models guess, and they guess wrong —
    e.g. treating a finished year as "not yet complete" and penalising good evidence."""
    return time.strftime("%Y-%m-%d")


STRUCTURE_PROMPT = """You are the intake step of a fact-verification system.

TODAY'S DATE IS {TODAY}. Use it. Resolve every relative time reference ("recently", "last year",
"the most recent full year", "currently") into an EXPLICIT year or date range in the normalized
claim — never leave a moving reference point, because a later step cannot resolve it.

NEVER CORRECT THE CLAIM. Your job is to sharpen it, not to fix it. If the claim is false, absurd,
or a well-known myth, restate it faithfully AS THE USER ASSERTED IT and let the later steps refute
it. You must preserve the direction of the assertion: if the user says X is true, the normalized
sentence must assert X is true — never its negation, never "X is a myth", never the debunking.
The sub-claims must be what would have to hold for the USER'S claim to be true, not for the
correct answer to be true. Silently flipping a false claim into a true one makes the whole system
report the opposite of what was asked.

Take the user's claim and make it CHECKABLE. Return JSON only:
{
 "normalized": "<the claim restated as one precise, falsifiable sentence>",
 "claim_type": "<factual|statistical|causal|predictive|opinion|definitional>",
 "checkable": <true|false>,
 "checkable_note": "<if not checkable (opinion/unfalsifiable/about the future), say why in one line; else empty>",
 "entities": ["<the specific named things this hinges on>"],
 "subclaims": ["<2-4 atomic assertions that must ALL hold for the claim to be true>"],
 "queries": ["<3 DIFFERENT search queries that would surface evidence — vary the angle,
              include one phrased to find CONTRADICTING evidence>"]
}

Claim: {CLAIM}"""


def structure(claim):
    d = gemini.gen_json(STRUCTURE_PROMPT.replace("{CLAIM}", claim)
                        .replace("{TODAY}", today()), max_tokens=800)
    d.setdefault("normalized", claim)
    d.setdefault("claim_type", "factual")
    d.setdefault("checkable", True)
    d.setdefault("entities", [])
    d.setdefault("subclaims", [])
    qs = [q for q in (d.get("queries") or []) if isinstance(q, str) and q.strip()]
    d["queries"] = qs[:3] or [claim]

    # Guard: the model sometimes "helpfully" normalizes a false claim into the TRUE one —
    # "humans use only 10% of their brain" came back as "no areas of the brain are dormant",
    # every lane then supported it, and the user's false claim scored 100/100. A flipped
    # normalization is worse than no normalization, so verify polarity and fall back if it moved.
    chk = gemini.gen_json(
        "Does sentence B assert the SAME thing as sentence A, in the same direction? Answer false "
        "if B negates A, debunks A, or states the opposite conclusion — even if B is the more "
        "accurate statement.\n\nReturn JSON only: {\"same\": <true|false>}\n\n"
        f"A: {claim}\nB: {d['normalized']}", max_tokens=100)
    if chk.get("same") is False:
        d["normalized"] = claim          # keep the user's wording; let the lanes judge it
        d["subclaims"] = []              # the sub-claims decomposed the flipped version too
    return d


# ══════════════════════════════════════════════════════════════════════════════
# 2. JUDGE — grounded: rate the claim ONLY against the evidence this lane fetched
# ══════════════════════════════════════════════════════════════════════════════
JUDGE_PROMPT = """You judge one claim against ONE source bundle.

TODAY'S DATE IS {TODAY}. Any year before this one is a completed, settled period — do not treat
a finished year's figures as provisional. Do check that the evidence covers the period the claim
is actually about. Judge ONLY on the evidence given —
do NOT use your own background knowledge, and do NOT be agreeable. If the evidence is thin,
off-topic, or merely adjacent, say insufficient rather than guessing.

Watch for SUBTLE MISMATCH: evidence that is about a different population, a different time period,
a weaker effect than claimed, or correlation where the claim asserts causation. That is NOT support.

Return JSON only:
{
 "stance": "<support|refute|mixed|insufficient>",
 "p_true": <0.0-1.0 probability the claim is TRUE given this evidence alone>,
 "strength": <0.0-1.0 how much this evidence bundle can settle anything: 0.1 = vague/tangential,
              1.0 = direct, specific, authoritative>,
 "summary": "<one sentence: what this evidence actually says about the claim>",
 "quote": "<the single most decisive verbatim snippet from the evidence, or empty>",
 "used": ["<exact [bracketed titles] of the sources you actually relied on — omit any
           source that was off-topic or that you ignored>"]
}

CLAIM: {CLAIM}

EVIDENCE:
{EVIDENCE}"""


def judge(claim, evidence):
    if not (evidence or "").strip():
        return {"stance": "insufficient", "p_true": 0.5, "strength": 0.0,
                "summary": "No evidence returned by this lane.", "quote": "", "used": []}
    d = gemini.gen_json(
        JUDGE_PROMPT.replace("{CLAIM}", claim).replace("{EVIDENCE}", evidence[:14000])
                    .replace("{TODAY}", today()),
        max_tokens=600)
    try:
        p = float(d.get("p_true", 0.5))
        s = float(d.get("strength", 0.0))
    except (TypeError, ValueError):
        p, s = 0.5, 0.0
    return {"stance": d.get("stance", "insufficient"),
            "p_true": min(max(p, 0.0), 1.0), "strength": min(max(s, 0.0), 1.0),
            "summary": d.get("summary", ""), "quote": (d.get("quote") or "")[:400],
            "used": [u for u in (d.get("used") or []) if isinstance(u, str)]}


CHALLENGE_PROMPT = """You are the opposing counsel. A lane of a verification system reached a
conclusion. Your job is to attack it using ONLY the evidence it was given.

TODAY'S DATE IS {TODAY}.

Try hard to find the flaw: does the evidence actually cover the claim's time period, population and
scope? Does it confuse correlation with causation? Does it rest on one original source restated
several times? Is the wording of the claim stronger than the evidence supports?

If after genuinely trying you cannot break it, say so — do not manufacture doubt.

Return JSON only:
{"breaks": <true|false — true only if you found a real, specific flaw>,
 "reason": "<the flaw in one sentence, or empty>"}

CLAIM: {CLAIM}
LANE CONCLUDED: {STANCE} (p_true {P}, strength {S}) — {SUMMARY}

EVIDENCE IT SAW:
{EVIDENCE}"""


def challenge(claim, j, evidence):
    """Adversarial second opinion per lane. A lane that survives keeps its weight; one that
    breaks gets its strength cut, so shaky lanes stop driving the verdict."""
    if j["stance"] == "insufficient" or j["strength"] <= 0.05:
        return j
    d = gemini.gen_json(CHALLENGE_PROMPT
                        .replace("{CLAIM}", claim).replace("{TODAY}", today())
                        .replace("{STANCE}", j["stance"]).replace("{P}", str(j["p_true"]))
                        .replace("{S}", str(j["strength"])).replace("{SUMMARY}", j["summary"])
                        .replace("{EVIDENCE}", (evidence or "")[:10000]), max_tokens=350)
    if d.get("breaks") and (d.get("reason") or "").strip():
        return {**j, "strength": round(j["strength"] * 0.55, 2),
                "challenge": (d.get("reason") or "")[:250]}
    return j


SUBCLAIM_PROMPT = """The claim below was broken into atomic assertions that must ALL hold for it to
be true. Rate each one against the pooled evidence. Judge only on the evidence.

TODAY'S DATE IS {TODAY}.

Return JSON only:
{"results": [{"subclaim": "<verbatim>", "verdict": "<supported|refuted|unsupported>",
              "note": "<one short sentence>"}]}

CLAIM: {CLAIM}

SUBCLAIMS:
{SUBS}

POOLED EVIDENCE:
{EVIDENCE}"""


def verify_subclaims(claim, subs, evidence):
    """The decomposition is worthless if nothing checks it. A claim whose parts don't all hold
    cannot be fully supported, however confident the lanes sound about the headline."""
    if not subs:
        return []
    d = gemini.gen_json(SUBCLAIM_PROMPT
                        .replace("{CLAIM}", claim).replace("{TODAY}", today())
                        .replace("{SUBS}", "\n".join(f"- {s}" for s in subs))
                        .replace("{EVIDENCE}", (evidence or "")[:16000]), max_tokens=800)
    out = []
    for r in (d.get("results") or []):
        if isinstance(r, dict) and r.get("subclaim"):
            out.append({"subclaim": str(r["subclaim"])[:300],
                        "verdict": str(r.get("verdict", "unsupported")).lower(),
                        "note": str(r.get("note", ""))[:200]})
    return out


# User-generated / aggregator domains: they restate other reporting rather than being
# evidence themselves, so they never count as a load-bearing source.
UGC = {"facebook.com", "x.com", "twitter.com", "reddit.com", "youtube.com", "instagram.com",
       "tiktok.com", "quora.com", "medium.com", "pinterest.com", "linkedin.com"}


def _src(title, url, excerpt=""):
    dom = (url or "").split("//")[-1].split("/")[0].replace("www.", "")
    return {"title": (title or dom or "source")[:120], "url": url or "",
            "dom": dom, "excerpt": (excerpt or "")[:220],
            "ugc": any(dom == u or dom.endswith("." + u) for u in UGC)}


def mark_cited(sources, used_titles):
    """Flag the sources the judge actually leaned on, so we can stop advertising the rest."""
    norm = [t.strip().strip("[]").lower() for t in (used_titles or []) if t.strip()]
    for s in sources:
        t = s["title"].lower()
        s["cited"] = any(n and (n in t or t in n) for n in norm) and not s["ugc"]
    return sources


# ══════════════════════════════════════════════════════════════════════════════
# 3. LANES — each returns (evidence_text, sources, cost). All run concurrently.
# ══════════════════════════════════════════════════════════════════════════════
def _grounded(queries, label):
    """Ask each query separately and keep the QUESTION attached to its answer.

    One of our queries is deliberately adversarial ("did X actually NOT happen?"). Fusing the
    answers into one anonymous blob makes an answer to that question read as evidence about the
    claim itself, which can flip a judge. Labelling each answer removes the ambiguity while
    keeping the contradiction-hunting value.
    """
    parts, cits, seen, cost = [], [], set(), 0.0
    for i, q in enumerate([q for q in queries if q][:3]):
        a = exa.answer(q, label=f"{label}#{i}")
        if a.get("answer"):
            parts.append(f"QUESTION ASKED: {q}\nANSWER: {a['answer']}")
        for c in a["citations"]:
            if c.get("url") and c["url"] not in seen:
                seen.add(c["url"])
                cits.append(c)
        cost += a.get("cost") or 0
    return "\n\n".join(parts), cits, round(cost, 4)


def lane_exa_answer(st):
    answers, cits, cost = _grounded(st["queries"], "claimcheck")
    ev = answers + "\n\nCITED PASSAGES:\n" + "\n\n".join(
        f"[{c.get('title', '')}] {(c.get('text') or '')[:900]}" for c in cits[:6])
    return ev, [_src(c.get("title"), c.get("url"), c.get("text")) for c in cits[:6]], cost


def lane_exa_search(st):
    items, seen = [], set()
    for q in st["queries"][:2]:
        for it in exa.search(q, n=4, max_chars=900):
            if it.url and it.url not in seen:
                seen.add(it.url)
                items.append(it)
    ev = "\n\n".join(f"[{i.title}] ({i.postdate or 'n/a'}) {i.content[:900]}" for i in items[:8])
    return ev, [_src(i.title, i.url, i.content) for i in items[:8]], 0.007 * len(st["queries"][:2])


def lane_serpapi(st):
    key = config.require("SERPAPI_API_KEY")
    rows, seen = [], set()
    for q in st["queries"][:2]:
        u = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
            {"q": q, "engine": "google", "num": 8, "api_key": key})
        try:
            d = json.load(urllib.request.urlopen(u, timeout=30))
        except Exception:
            continue
        for r in (d.get("organic_results") or [])[:8]:
            link = r.get("link") or ""
            if link and link not in seen:
                seen.add(link)
                rows.append((r.get("title") or "", link,
                             r.get("snippet") or "", r.get("date") or ""))
        # an answer box is Google's own extracted answer — strong signal, keep it first
        ab = d.get("answer_box") or {}
        if ab.get("answer") or ab.get("snippet"):
            rows.insert(0, ("Google answer box", ab.get("link") or "",
                            ab.get("answer") or ab.get("snippet") or "", ""))
    ev = "\n\n".join(f"[{t}] ({dt or 'n/a'}) {s}" for t, _l, s, dt in rows[:10])
    return ev, [_src(t, l, s) for t, l, s, _dt in rows[:10]], 0.01 * len(st["queries"][:2])


PARALLEL_SCHema = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "description": "Is the claim true? One of: true, mostly true, mixed, "
                                   "mostly false, false, cannot determine"},
        "evidence": {"type": "string",
                     "description": "The specific findings that decide it — figures, dates, "
                                    "who reported what. Include contradicting findings if any."},
    },
    "required": ["verdict", "evidence"],
}


def lane_deep_exa(st):
    """Deep lane, fast path: instead of one broad research run, we ask a grounded question per
    SUB-CLAIM. That probes the claim's load-bearing parts individually and returns in seconds
    rather than the ~70s the heavyweight research processor needs."""
    qs = (st.get("subclaims") or [])[:3] or [st["normalized"]]
    answers, cits, cost = _grounded([f"Is this accurate: {q}" for q in qs], "claimcheck-deep")
    ev = "PER-SUB-CLAIM FINDINGS:\n" + answers + "\n\nCITED PASSAGES:\n" + "\n\n".join(
        f"[{c.get('title', '')}] {(c.get('text') or '')[:900]}" for c in cits[:8])
    return ev, [_src(c.get("title"), c.get("url"), c.get("text")) for c in cits[:8]], cost


def lane_parallel(st):
    r = parallel_ai.task(
        f"Verify this claim against the live web. Report what the best available sources "
        f"actually say, including anything that contradicts it.\n\nCLAIM: {st['normalized']}",
        PARALLEL_SCHema, processor="lite", poll=30, interval=2)
    c, basis = r.get("content") or {}, r.get("basis") or {}
    cits = []
    for b in basis.values():
        cits.extend(b.get("citations") or [])
    ev = f"DEEP RESEARCH VERDICT: {c.get('verdict', '')}\n\nFINDINGS:\n{c.get('evidence', '')}"
    return ev, parallel_ai.dedup_citations(cits, k=6), 0.02


def _openalex_abstract(work):
    """OpenAlex ships abstracts as an inverted index {word: [positions]} — rebuild the text."""
    inv = work.get("abstract_inverted_index")
    if not isinstance(inv, dict):
        return ""
    words = {}
    for w, ps in inv.items():
        for pos in ps or []:
            words[pos] = w
    return " ".join(words[i] for i in sorted(words))[:900]


def lane_research(st):
    """Peer-reviewed literature lane. Causal and statistical claims are settled by studies, not
    by news write-ups of studies — and this is exactly the class where a language model's recall
    is least trustworthy. Two free, keyless APIs; citation counts let the judge weigh authority.
    """
    # Academic search wants keywords, not prose. The normalized claim is a long, date-stamped
    # sentence and returns nothing; the extracted entities are exactly the right query.
    q = " ".join((st.get("entities") or [])[:4]) or st["normalized"][:120]
    items, srcs = [], []

    try:
        u = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"search": q, "per-page": 5, "sort": "relevance_score:desc"})
        d = json.load(urllib.request.urlopen(
            urllib.request.Request(u, headers={"User-Agent": "rcmnd-claimcheck/1.0 (mailto:denis@recommend.co)"}),
            timeout=25))
        for w in (d.get("results") or [])[:5]:
            ab = _openalex_abstract(w)
            if not ab:
                continue
            venue = (((w.get("primary_location") or {}).get("source") or {}).get("display_name")) or ""
            title = w.get("title") or "study"
            items.append(f"[{title}] ({w.get('publication_year')}, {venue}; "
                         f"cited {w.get('cited_by_count', 0)}×)\n{ab}")
            srcs.append(_src(title, w.get("doi") or (w.get("id") or ""), ab))
    except Exception:
        pass

    try:
        u = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": q, "format": "json", "pageSize": 4, "resultType": "core"})
        d = json.load(urllib.request.urlopen(u, timeout=25))
        for r in ((d.get("resultList") or {}).get("result") or [])[:4]:
            ab = (r.get("abstractText") or "")[:900]
            if not ab:
                continue
            title = r.get("title") or "study"
            items.append(f"[{title}] ({r.get('pubYear')}, {r.get('journalTitle', '')}; "
                         f"cited {r.get('citedByCount', 0)}×)\n{ab}")
            srcs.append(_src(title, f"https://europepmc.org/article/MED/{r.get('pmid')}"
                             if r.get("pmid") else (r.get("doi") or ""), ab))
    except Exception:
        pass

    if not items:
        return "", [], 0
    ev = ("PEER-REVIEWED LITERATURE (citation counts indicate how much the field relies on each; "
          "a single small study is weak evidence, a well-cited review is strong):\n\n"
          + "\n\n".join(items))
    return ev, srcs, 0


def lane_market(st):
    """Prediction-market lane. A market price IS a probability — money-backed, continuously
    updated crowd forecasting, and historically well calibrated. It is the one credible signal
    for exactly the claims evidence cannot settle: the ones about the future. Public read API,
    no key, so it costs nothing.
    """
    # Search Polymarket properly. (Listing the top markets by volume and hoping for an overlap
    # only ever matched whatever happened to be trending — almost every claim missed.)
    # public-search returns EVENTS with all of their markets and current prices, so we then pick
    # the market inside the event that actually matches this claim.
    terms = " ".join((st.get("entities") or [])[:4]) or st["normalized"][:90]
    events = []
    for q_try in [terms, st["normalized"][:90]]:
        u = "https://gamma-api.polymarket.com/public-search?" + urllib.parse.urlencode(
            {"q": q_try, "limit_per_type": 6})
        try:
            d = json.load(urllib.request.urlopen(
                urllib.request.Request(u, headers={"User-Agent": "rcmnd-claimcheck/1.0"}),
                timeout=25))
            events = d.get("events") or []
        except Exception:
            events = []
        if events:
            break
    if not events:
        return "", [], 0

    want = {w.strip(".,?'s").lower() for w in
            (terms + " " + st["normalized"]).split() if len(w) > 3}
    hits = []
    for e in events:
        for m in (e.get("markets") or []):
            q = m.get("question") or ""
            if not q:
                continue
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
            except Exception:
                continue
            if not prices or not outs or prices[0] in (None, "None"):
                continue
            overlap = len(want & {w.strip(".,?'s").lower() for w in q.split()})
            if overlap:
                hits.append((overlap, q, dict(zip(outs, prices)),
                             m.get("volume24hr") or e.get("volume24hr") or 0,
                             e.get("slug") or m.get("slug") or ""))
    if not hits:
        return "", [], 0
    hits.sort(key=lambda x: (-x[0], -float(x[3] or 0)))

    # Keep only the strongest matches and drop duplicates. Without this a claim about
    # "Bitcoin reaching $200,000" happily pulled in the $65,000 market sitting at 100% —
    # a near-miss the judge could read as confirmation.
    best, seen, kept = hits[0][0], set(), []
    for h in hits:
        if h[0] < best or h[1] in seen:
            continue
        seen.add(h[1])
        kept.append(h)
    hits = kept

    lines, srcs = [], []
    for _o, q, px, vol, slug in hits[:3]:
        yes = px.get("Yes")
        lines.append(f'MARKET: "{q}"\n  implied probability: '
                     + ", ".join(f"{k} {float(v)*100:.1f}%" for k, v in px.items())
                     + f"  (24h volume ${float(vol or 0):,.0f})")
        srcs.append(_src(f"Prediction market: {q}",
                         f"https://polymarket.com/event/{slug}" if slug else "",
                         f"Yes {float(yes)*100:.1f}% implied" if yes else ""))
    ev = ("LIVE PREDICTION MARKET PRICING (real money at stake; the price of a 'Yes' share is the "
          "market's implied probability of the event).\n\n"
          "STATUS: these markets are still OPEN — nothing here is a settled outcome. A live price "
          "is what traders currently BELIEVE, not a record of what happened. Treat it the way you "
          "would treat a well-informed forum consensus: it is a signal about likelihood, never "
          "confirmation that the claim is true. Only a RESOLVED market is a record of fact.\n\n"
          "IMPORTANT: these markets were found by text search and may be NEAR MISSES. Before using "
          "one, check its question matches the claim exactly — the same threshold or number, the "
          "same date range, the same person or party. A market on a DIFFERENT threshold or deadline "
          "says nothing about this claim; treat it as insufficient rather than as confirmation.\n\n"
          + "\n\n".join(lines))
    return ev, srcs, 0


LANES = [
    ("web_grounded", "Grounded Web", lane_exa_answer, "fast"),
    ("web_semantic", "Semantic Web", lane_exa_search, "fast"),
    ("web_serp", "Live Index", lane_serpapi, "fast"),
    # DEEP_ENGINE=parallel switches back to the heavyweight research processor: better
    # citations, but ~70s vs ~8s. Fast path is the default.
    ("deep_research", "Deep Research",
     lane_parallel if os.environ.get("DEEP_ENGINE") == "parallel" else lane_deep_exa, "deep"),
    # only meaningful for claims about the future — no market exists for settled history
    ("market", "Prediction Market", lane_market, "forecast"),
    # studies settle cause-and-effect and population statistics; news coverage of them does not
    ("research", "Research Literature", lane_research, "science"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE — strength-weighted, and honest about disagreement
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(results, st=None, n_lanes=None, subs=None):
    """results = finished lanes. n_lanes = how many were STARTED — a lane that came back
    with nothing must cost us confidence, so it has to stay in the denominator."""
    st = st or {}
    n_lanes = n_lanes or max(len(results), 1)
    usable = [r for r in results if r["strength"] > 0.05 and r["stance"] != "insufficient"]
    abstained = n_lanes - len(usable)

    # A claim about the future, an opinion, or anything unfalsifiable cannot be settled by
    # evidence at all. Absence of reporting that Trump plans to attack Croatia is NOT proof
    # he won't — so these never get a factual verdict, however one-sided the lanes look.
    ctype = (st.get("claim_type") or "").lower()
    unverifiable = (not st.get("checkable", True)) or ctype in ("predictive", "opinion")

    if not usable:
        return {"label": "NOT VERIFIABLE" if unverifiable else "INSUFFICIENT EVIDENCE",
                "tone": "warn", "p_true": 0.5, "confidence": 0.1, "unverifiable": unverifiable,
                "status": "open" if unverifiable else "unknown",
                "status_note": ("Open question — no evidence settles this yet." if unverifiable
                                else "No lane found evidence solid enough to classify this."),
                "rationale": (st.get("checkable_note") or
                              "No lane returned evidence strong enough to decide this."),
                "lanes_used": 0, "lanes_total": n_lanes}

    wts = [WEIGHTS.get(r["lane"], 1.0) * r["strength"] for r in usable]
    p = sum(r["p_true"] * w for r, w in zip(usable, wts)) / sum(wts)
    # if belief is the ONLY thing we have, we do not have a finding
    market_only = all(r["lane"] == "market" for r in usable)

    # Disagreement between independent lanes is the honest confidence killer: two lanes that
    # both looked at real evidence and landed on opposite sides means we do NOT know.
    spread = max(r["p_true"] for r in usable) - min(r["p_true"] for r in usable)
    # divide by lanes STARTED, not by a constant — silence from a lane now lowers the score
    evidence_mass = min(sum(wts) / max(n_lanes, 1), 1.0)
    decisiveness = abs(p - 0.5) * 2
    conf = evidence_mass * (1 - 0.55 * spread) * (0.45 + 0.55 * decisiveness)
    conf = min(max(conf, 0.05), 0.96)
    if market_only:
        conf = min(conf, 0.3)
        p = 0.5 + (p - 0.5) * 0.6      # pull the lean back toward "unknown"

    # The claim can only be as true as its weakest necessary part. A headline the lanes love,
    # resting on a sub-claim the evidence refutes, must not come out as SUPPORTED.
    broken = [s for s in (subs or []) if s["verdict"] == "refuted"]
    unbacked = [s for s in (subs or []) if s["verdict"] == "unsupported"]
    if broken and p > 0.5:
        p = min(p, 0.45)
        conf = min(conf, 0.6)
    elif unbacked and p > 0.8:
        p = min(p, 0.75)                              # a part nobody could confirm caps certainty

    if p >= 0.80:
        label, tone = "SUPPORTED", "good"
    elif p >= 0.60:
        label, tone = "LIKELY TRUE", "good"
    elif p > 0.40:
        label, tone = "DISPUTED", "warn"
    elif p > 0.20:
        label, tone = "LIKELY FALSE", "bad"
    else:
        label, tone = "REFUTED", "bad"
    if spread >= 0.5 and 0.25 < p < 0.75:
        label, tone = "DISPUTED", "warn"

    # What KIND of question is this? Without saying so, a user has to infer it from which lanes
    # ran — e.g. no market lane on "France wins the 2026 World Cup" because that tournament has
    # already finished, making it a settled question rather than an open one.
    if ctype == "opinion" or (not st.get("checkable", True) and ctype != "predictive"):
        status, status_note = "opinion", "Not falsifiable — a matter of judgement, not of fact."
    elif unverifiable:
        status, status_note = "open", ("Open question — the event hasn't happened yet, so this is "
                                       "a lean from current evidence, not a verdict.")
    elif (spread >= 0.35 or 0.35 < p < 0.65 or conf < 0.45
          or any(r["stance"] == "mixed" for r in usable)):
        # Low confidence and "settled" cannot both be true — if we're unsure, say the question
        # is open among the sources rather than claiming the evidence decided it.
        status, status_note = "contested", ("Contested — credible evidence exists on both sides "
                                            "and the question is genuinely unsettled.")
    else:
        status, status_note = "settled", "Settled question — evidence can decide this, and does."

    note = ""
    if unverifiable:
        # We still give a lean — "nobody credible is reporting this" is real information —
        # but evidence cannot CLOSE a claim about the future, so we say lean, not verdict.
        label = "LEANS FALSE" if p <= 0.35 else ("LEANS TRUE" if p >= 0.65 else "UNSETTLED")
        tone = "warn"
        conf = min(conf, 0.45)
        note = (st.get("checkable_note")
                or f"This is a {ctype or 'non-factual'} claim — evidence can inform it "
                   f"but cannot settle it. Confidence is capped accordingly.")

    agree = sum(1 for r in usable if (r["p_true"] > 0.5) == (p > 0.5))
    bits = [f"{agree} of {n_lanes} evidence lanes agree"]
    if abstained:
        bits.append(f"{abstained} returned nothing usable, which lowers confidence")
    bits.append("lanes disagree, so confidence is held down" if spread >= 0.4
                else f"lane spread {spread:.2f}, consistent")
    return {"label": label, "tone": tone, "p_true": round(p, 3),
            "confidence": round(conf, 2), "unverifiable": unverifiable, "note": note,
            "status": status, "status_note": status_note,
            "rationale": ". ".join(bits) + ".",
            "lanes_used": len(usable), "lanes_total": n_lanes}


REVIEW_PROMPT = """You are the calibration check on a verification system.

TODAY'S DATE IS {TODAY}. Never flag a completed past year as "not yet complete" — check the date
before raising any temporal concern. Your only job is to catch
OVERCONFIDENCE — the system asserting more than its evidence can carry.

IMPORTANT — what "confidence" means here: it is how much to trust THE VERDICT, not how likely the
claim is to be true. A verdict of "LEANS FALSE" backed by strong, on-topic evidence deserves HIGH
confidence. Do not lower confidence merely because the claim turned out to be false or unsupported —
that is the verdict doing its job. Lower it only when the VERDICT ITSELF is shaky.

Ask yourself: could this verdict be wrong? Is the evidence about exactly this claim, or only
adjacent to it? Did the lanes actually converge, or did they just repeat one original source?

You may ONLY lower confidence or leave it. You may never raise it.

Return JSON only:
{"confidence": <0.0-1.0, your calibrated value — <= the proposed one>,
 "concern": "<one sentence naming the specific weakness, or empty if the verdict is well supported>"}

CLAIM: {CLAIM}
CLAIM TYPE: {TYPE}
PROPOSED VERDICT: {LABEL} at {CONF} confidence
LANE FINDINGS:
{LANES}"""


def calibration_review(claim, st, verdict, results):
    """Final adversarial pass. It can only take confidence away — a one-way ratchet means a
    bug here can make us too cautious, never too sure."""
    lanes = "\n".join(f"- {r['name']}: {r['stance']} (p_true {r['p_true']}, "
                      f"strength {r['strength']}) — {r['summary']}" for r in results) or "none"
    d = gemini.gen_json(REVIEW_PROMPT
                        .replace("{CLAIM}", st.get("normalized", claim))
                        .replace("{TYPE}", st.get("claim_type", "factual"))
                        .replace("{LABEL}", verdict["label"])
                        .replace("{CONF}", str(verdict["confidence"]))
                        .replace("{LANES}", lanes)
                        .replace("{TODAY}", today()), max_tokens=400)
    try:
        c = float(d.get("confidence", verdict["confidence"]))
    except (TypeError, ValueError):
        return verdict
    capped = min(max(c, 0.05), verdict["confidence"])        # one-way: never upward
    out = {**verdict, "confidence": round(capped, 2)}
    concern = (d.get("concern") or "").strip()
    if concern and capped < verdict["confidence"] - 0.01:
        out["concern"] = concern[:300]
    return out


SUMMARY_PROMPT = """Write the plain-language readout a person sees after a claim was checked.
Address what the EVIDENCE showed and WHY the verdict came out this way.

Rules:
- 2-3 sentences, no bullet points, no preamble, no restating the score as a number.
- Lead with which way the evidence leans and WHY — the substance, not the process.
- If confidence is low or lanes disagreed, say plainly what is unresolved.
- If the question cannot be settled by evidence (a prediction, an opinion), say so.
- Never invent a fact that is not in the lane findings below.
- Neutral and specific. No hedging filler, no "it is important to note".

CLAIM: {CLAIM}
VERDICT: {LABEL} (truth score {SCORE}/100, confidence {CONF})
QUESTION TYPE: {STATUS}

LANE FINDINGS:
{LANES}

SUB-CLAIM RESULTS:
{SUBS}"""


def narrate(claim, verdict, results, subs):
    """The human-readable 'so what'. Everything else on screen is a number or a source list."""
    lanes = "\n".join(f"- {r['name']}: {r['stance']} — {r['summary']}"
                      for r in results if r.get("summary")) or "none"
    sub_txt = "\n".join(f"- [{s['verdict']}] {s['subclaim']} {s.get('note', '')}"
                        for s in (subs or [])) or "none"
    txt = gemini.gen(SUMMARY_PROMPT
                     .replace("{CLAIM}", claim).replace("{LABEL}", verdict.get("label", ""))
                     .replace("{SCORE}", str(verdict.get("score", "")))
                     .replace("{CONF}", str(verdict.get("confidence", "")))
                     .replace("{STATUS}", verdict.get("status_note", ""))
                     .replace("{LANES}", lanes).replace("{SUBS}", sub_txt),
                     temperature=0.3, max_tokens=350)
    return " ".join((txt or "").split())[:900]


# ══════════════════════════════════════════════════════════════════════════════
# 5. ORCHESTRATION — emits SSE events as things land
# ══════════════════════════════════════════════════════════════════════════════
def run_check(claim, emit, use_deep=True):
    t0 = time.time()
    emit("status", {"msg": "Structuring the claim…"})
    st = structure(claim)
    emit("structured", st)

    # route by claim type: a market only exists for the future, and studies are what settle
    # cause-and-effect — firing every lane on every claim is slower AND noisier
    forward = (not st.get("checkable", True)) or st.get("claim_type") == "predictive"
    scientific = st.get("claim_type") in ("causal", "statistical")
    lanes = [l for l in LANES
             if (l[3] == "fast" or (l[3] == "deep" and use_deep)
                 or (l[3] == "forecast" and forward)
                 or (l[3] == "science" and scientific))]
    emit("lanes", {"lanes": [{"id": i, "name": n, "kind": k} for i, n, _f, k in lanes]})

    results, lock = [], threading.Lock()

    def run_lane(lane_id, name, fn):
        s = time.time()
        try:
            ev, sources, cost = fn(st)
            j = challenge(st["normalized"], judge(st["normalized"], ev), ev)
        except Exception as e:                       # one lane dying must not kill the check
            emit("lane_done", {"id": lane_id, "error": str(e)[:200],
                               "ms": int((time.time() - s) * 1000)})
            return
        # An open prediction market is crowd belief, not a record of fact, so it is capped
        # to opinion-grade weight — it can inform a lean, never carry a verdict on its own.
        if lane_id == "market" and j["strength"] > 0.35:
            j = {**j, "strength": 0.35}
        sources = mark_cited(sources, j.get("used"))
        # a lane that found nothing usable doesn't get to pad the source list
        if j["stance"] == "insufficient" or j["strength"] <= 0.05:
            sources = []
        r = {"lane": lane_id, "name": name, "sources": sources, "cost": cost,
             "evidence": ev, "ms": int((time.time() - s) * 1000), **j}
        with lock:
            results.append(r)
        emit("lane_done", {k: v for k, v in r.items() if k != "evidence"})

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futs = [pool.submit(run_lane, i, n, f) for i, n, f, _k in lanes]
        # preliminary verdict the moment the FAST lanes are in — the deep lane refines it
        fast_ids = {l[0] for l in lanes if l[3] == "fast"}
        while any(not f.done() for f in futs):
            with lock:
                done_fast = {r["lane"] for r in results} >= fast_ids
            if done_fast:
                with lock:
                    snap = list(results)
                emit("verdict", {**aggregate(snap, st, len(fast_ids)), "preliminary": True,
                                 "elapsed": round(time.time() - t0, 1)})
                break
            time.sleep(0.15)
        for f in futs:
            f.result()

    emit("status", {"msg": "Verifying sub-claims…"})
    pooled = "\n\n".join(f"[{r['name']}]\n{r.get('evidence','')}" for r in results)
    subs = verify_subclaims(st["normalized"], st.get("subclaims"), pooled)
    emit("subclaims", {"results": subs})

    emit("status", {"msg": "Calibrating…"})
    final = calibration_review(claim, st, aggregate(results, st, len(lanes), subs), results)
    final["subclaims"] = subs
    cited = sum(sum(1 for s in r["sources"] if s.get("cited")) for r in results)
    final = {**final, "preliminary": False,
             "elapsed": round(time.time() - t0, 1),
             "cost": round(sum(r.get("cost") or 0 for r in results), 4),
             "n_sources": sum(len(r["sources"]) for r in results), "n_cited": cited,
             # NB: `or 0.5` would be wrong here — p_true of exactly 0.0 is falsy but meaningful
             "score": int(round((final.get("p_true") if final.get("p_true") is not None
                                 else 0.5) * 100))}
    emit("verdict", final)
    emit("status", {"msg": "Writing the readout…"})
    final["summary_text"] = narrate(st.get("normalized", claim), final, results, subs)
    emit("summary", {"text": final["summary_text"]})
    log_check(claim, st, results, final)
    emit("done", {})


# ── every check is appended to disk: this is the corpus we tune the judges on later ──
LOG = os.path.join(HERE, "data", "checks.jsonl")


def log_check(claim, st, results, verdict):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        rec = {"ts": time.time(), "claim": claim, "structured": st,
               "verdict": {k: verdict.get(k) for k in
                           ("label", "score", "p_true", "confidence", "unverifiable",
                            "concern", "rationale", "elapsed", "cost")},
               "lanes": [{k: r.get(k) for k in
                          ("lane", "stance", "p_true", "strength", "summary", "ms")}
                         for r in results]}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass                                          # logging must never break a check


# ══════════════════════════════════════════════════════════════════════════════
# 5b. TRACKING — a claim isn't always settled once. Anyone can pin a claim to their own
#     session and we re-check it on a schedule, so the score's daily movement is visible.
#     Scoped by session id, so two people never see each other's watchlists.
# ══════════════════════════════════════════════════════════════════════════════
TRACK_FILE = os.path.join(HERE, "data", "tracked.json")
TRACK_EVERY_H = float(os.environ.get("TRACK_EVERY_H", 24))
_track_lock = threading.Lock()


def _track_load():
    try:
        return json.load(open(TRACK_FILE))
    except Exception:
        return {}


def _track_save(d):
    os.makedirs(os.path.dirname(TRACK_FILE), exist_ok=True)
    tmp = TRACK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, TRACK_FILE)                       # atomic: never leave a half-written file


def track_add(sid, claim, verdict):
    key = claim.strip().lower()[:200]
    with _track_lock:
        d = _track_load()
        sess = d.setdefault(sid, {})
        item = sess.setdefault(key, {"claim": claim.strip(), "history": []})
        item["history"].append({"ts": time.time(), "score": verdict.get("score"),
                                "confidence": verdict.get("confidence"),
                                "label": verdict.get("label")})
        item["history"] = item["history"][-60:]       # ~2 months of daily points is plenty
        item["last"] = time.time()
        _track_save(d)
    return track_list(sid)


def track_remove(sid, claim):
    with _track_lock:
        d = _track_load()
        d.get(sid, {}).pop(claim.strip().lower()[:200], None)
        _track_save(d)
    return track_list(sid)


def track_list(sid):
    d = _track_load().get(sid, {})
    out = []
    for key, it in d.items():
        h = it["history"]
        cur, prev = (h[-1] if h else {}), (h[-2] if len(h) > 1 else None)
        out.append({"key": key, "claim": it["claim"],
                    "score": cur.get("score"), "label": cur.get("label"),
                    "confidence": cur.get("confidence"),
                    "delta": (cur.get("score", 0) - prev["score"]) if prev and
                             prev.get("score") is not None and cur.get("score") is not None else None,
                    "checks": len(h), "last": it.get("last"),
                    "history": [{"ts": p["ts"], "score": p["score"]} for p in h]})
    return sorted(out, key=lambda x: -(x["last"] or 0))


def track_recheck(sid, claim):
    """Re-run a tracked claim now and append the new point."""
    events = []
    run_check(claim, lambda ev, d: events.append((ev, d)), use_deep=False)
    v = [d for ev, d in events if ev == "verdict" and not d.get("preliminary")]
    if v:
        track_add(sid, claim, v[-1])
    return track_list(sid)


def _track_daemon():
    """Background re-check loop: anything older than TRACK_EVERY_H gets a fresh point."""
    while True:
        try:
            due = []
            for sid, items in _track_load().items():
                for it in items.values():
                    if time.time() - (it.get("last") or 0) > TRACK_EVERY_H * 3600:
                        due.append((sid, it["claim"]))
            for sid, claim in due[:5]:               # a few per wake-up, so we never stampede
                try:
                    track_recheck(sid, claim)
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(900)                               # wake every 15 min


# ══════════════════════════════════════════════════════════════════════════════
# 6. BLIND LLMs — the control group: same claim, no evidence, no retrieval.
#    This is the comparison that sells the product: a raw model answers from memory
#    and reports its OWN confidence, which is famously ~95% on everything including
#    its mistakes. We put that next to our grounded, source-backed verdict.
# ══════════════════════════════════════════════════════════════════════════════
BLIND_PROMPT = """Answer from your own knowledge ONLY. You have no internet access and no sources.

Return JSON only:
{
 "verdict": "<true|mostly true|mixed|mostly false|false>",
 "confidence": <0-100, how confident YOU are in your verdict>,
 "answer": "<2-3 sentences: your reasoning and any specifics you recall>"
}

CLAIM: {CLAIM}"""

def _norm_blind(d):
    try:
        c = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        c = 0.0
    return {"verdict": (d.get("verdict") or "unknown").lower(),
            "confidence": round(min(max(c, 0.0), 100.0) / 100.0, 2),
            "answer": (d.get("answer") or "")[:700]}


def blind_gemini(claim):
    return _norm_blind(gemini.gen_json(BLIND_PROMPT.replace("{CLAIM}", claim), max_tokens=500))


BLIND_MODELS = [("gemini", f"Gemini · {gemini.model_name()}", blind_gemini)]

# how a model's word-verdict maps onto our 0-1 truth axis, so we can score it
VERDICT_P = {"true": 0.95, "mostly true": 0.75, "mixed": 0.5,
             "mostly false": 0.25, "false": 0.05, "unknown": 0.5}


def compare(claim, grounded, emit):
    """Ask each raw model blind, in parallel, then contrast with the grounded verdict."""
    results, lock = [], threading.Lock()

    def ask(mid, name, fn):
        s = time.time()
        try:
            r = {"id": mid, "name": name, "ms": int((time.time() - s) * 1000), **fn(claim)}
        except Exception as e:
            r = {"id": mid, "name": name, "error": str(e)[:250],
                 "ms": int((time.time() - s) * 1000)}
        with lock:
            results.append(r)
        emit("model_done", r)

    with ThreadPoolExecutor(max_workers=len(BLIND_MODELS)) as pool:
        for f in [pool.submit(ask, *m) for m in BLIND_MODELS]:
            f.result()

    gp = grounded.get("p_true", 0.5)
    rows = []
    for r in results:
        if r.get("error"):
            continue
        mp = VERDICT_P.get(r["verdict"], 0.5)
        # does the blind model land on the same side of the fence as the grounded check?
        agrees = (mp > 0.5) == (gp > 0.5) if abs(gp - 0.5) > 0.1 else abs(mp - gp) < 0.25
        rows.append({"id": r["id"], "name": r["name"], "verdict": r["verdict"],
                     "claimed": r["confidence"], "agrees": agrees,
                     # overconfidence = how sure it SOUNDED minus how right it was
                     "overconfidence": round(r["confidence"] - (1 - abs(mp - gp)), 2)})

    ok = [r for r in rows if r["agrees"]]
    avg_claimed = round(sum(r["claimed"] for r in rows) / len(rows), 2) if rows else 0
    who = "Both models" if len(rows) > 1 else "The blind model"
    if not rows:
        insight = "No blind model returned an answer to compare against."
    elif len(ok) == len(rows):
        insight = (f"{who} happened to land on the right side here — but asserted it at "
                   f"{int(avg_claimed*100)}% confidence with zero sources. On a claim they get wrong, "
                   f"they sound exactly this certain. Our {int(grounded.get('confidence',0)*100)}% is "
                   f"backed by {grounded.get('n_sources', 0)} readable sources.")
    elif not ok:
        insight = (f"{who} contradict{'' if len(rows) > 1 else 's'} the evidence — while reporting "
                   f"{int(avg_claimed*100)}% confidence. This is the failure mode: fluent, certain, wrong.")
    else:
        insight = (f"The blind models disagree with each other ({len(ok)} of {len(rows)} match the "
                   f"evidence), yet each reports ~{int(avg_claimed*100)}% confidence. Self-reported "
                   f"confidence cannot tell you which one to trust — grounding can.")

    emit("delta", {"rows": rows, "grounded": grounded, "insight": insight,
                   "avg_claimed": avg_claimed})
    emit("done", {})


# ══════════════════════════════════════════════════════════════════════════════
# 7. MCP — expose the whole thing as a tool any LLM agent can call.
#    Streamable-HTTP transport (JSON-RPC over POST /mcp), gated on a bearer key
#    so it only works for someone we hand a key to.
# ══════════════════════════════════════════════════════════════════════════════
KEYS_FILE = os.path.join(HERE, "keys.json")

# ── spend guards: this is public, and every check costs real API credit ──────────
PER_IP_DAY = int(os.environ.get("PER_IP_DAY", 25))
GLOBAL_DAY = int(os.environ.get("GLOBAL_DAY", 400))
_rate = {"day": None, "ip": {}, "total": 0}
_rate_lock = threading.Lock()


def rate_ok(ip):
    today = time.strftime("%Y-%m-%d")
    with _rate_lock:
        if _rate["day"] != today:                      # new day → wipe the counters
            _rate.update(day=today, ip={}, total=0)
        if _rate["total"] >= GLOBAL_DAY:
            return False, "Daily demo limit reached for this instance."
        if _rate["ip"].get(ip, 0) >= PER_IP_DAY:
            return False, f"You've used the {PER_IP_DAY} checks available per day."
        _rate["ip"][ip] = _rate["ip"].get(ip, 0) + 1
        _rate["total"] += 1
        return True, ""


def load_keys():
    """{key: label}. Auto-creates one demo key on first run so the PoC is usable immediately."""
    if os.path.exists(KEYS_FILE):
        try:
            return json.load(open(KEYS_FILE))
        except Exception:
            pass
    import secrets
    keys = {"rcmd_" + secrets.token_urlsafe(24): "demo"}
    json.dump(keys, open(KEYS_FILE, "w"), indent=1)
    return keys


KEYS = {}

MCP_TOOLS = [{
    "name": "verify_claim",
    "description": (
        "Verify a factual claim against live web evidence. Structures the claim, queries four "
        "independent evidence lanes in parallel (grounded web answer, semantic web search, live "
        "search index, deep research), judges each strictly against the sources it fetched, "
        "and returns a verdict with a calibrated confidence that drops when lanes disagree. "
        "Use this instead of answering factual questions from memory."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "claim": {"type": "string", "description": "The claim to verify, as a statement."},
            "deep": {"type": "boolean", "default": False,
                     "description": "Include the slower deep-research lane (~35s vs ~12s)."},
        },
        "required": ["claim"],
    },
}]


def mcp_verify(claim, deep=False):
    """Run a check and collapse the event stream into one agent-friendly result."""
    events = []
    run_check(claim, lambda ev, d: events.append((ev, d)), use_deep=deep)
    verdicts = [d for ev, d in events if ev == "verdict" and not d.get("preliminary")]
    lanes = [d for ev, d in events if ev == "lane_done" and not d.get("error")]
    st = next((d for ev, d in events if ev == "structured"), {})
    v = verdicts[-1] if verdicts else {}
    srcs, seen = [], set()
    for l in lanes:
        for s in l["sources"]:
            if s["url"] and s["url"] not in seen:
                seen.add(s["url"])
                srcs.append({"title": s["title"], "url": s["url"]})
    return {
        "claim": claim, "normalized": st.get("normalized", claim),
        "verdict": v.get("label"), "confidence": v.get("confidence"),
        "p_true": v.get("p_true"), "rationale": v.get("rationale"),
        # the VERIFIED sub-claims with their verdicts — an agent calling this needs to know
        # which parts actually held, not just how the claim was decomposed
        "subclaims": v.get("subclaims") or [{"subclaim": x, "verdict": "unchecked"}
                                           for x in st.get("subclaims", [])],
        "score": v.get("score"), "unverifiable": v.get("unverifiable"),
        "question_status": v.get("status"), "status_note": v.get("status_note"),
        "readout": v.get("summary_text"),
        "calibration_concern": v.get("concern"),
        "lanes": [{"lane": l["lane"], "stance": l["stance"], "p_true": l["p_true"],
                   "strength": l["strength"], "summary": l["summary"],
                   "challenged": l.get("challenge")} for l in lanes],
        "sources": srcs[:20], "elapsed_s": v.get("elapsed"),
    }


def mcp_handle(msg):
    """Minimal MCP: initialize / tools/list / tools/call. Returns a JSON-RPC reply (or None)."""
    mid, method = msg.get("id"), msg.get("method")

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}},
                   "serverInfo": {"name": "recommend-claimcheck", "version": "0.1.0"}})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None                                   # notifications get no reply
    if method == "tools/list":
        return ok({"tools": MCP_TOOLS})
    if method == "tools/call":
        p = msg.get("params") or {}
        if p.get("name") != "verify_claim":
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"Unknown tool: {p.get('name')}"}}
        a = p.get("arguments") or {}
        claim = (a.get("claim") or "").strip()
        if not claim:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32602, "message": "Missing required argument: claim"}}
        try:
            out = mcp_verify(claim, deep=bool(a.get("deep")))
        except Exception as e:
            return ok({"content": [{"type": "text", "text": f"Verification failed: {e}"}],
                       "isError": True})
        return ok({"content": [{"type": "text",
                                "text": json.dumps(out, ensure_ascii=False, indent=1)}]})
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"Unknown method: {method}"}}


# ══════════════════════════════════════════════════════════════════════════════
# HTTP
# ══════════════════════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _authed(self):
        """Bearer token (or ?key=) must match a key we issued."""
        h = self.headers.get("Authorization") or ""
        key = h[7:].strip() if h.lower().startswith("bearer ") else ""
        if not key:
            key = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                   .get("key") or [""])[0]
        return key in KEYS

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/mcp":
            self.send_error(404)
            return
        # Always drain the body FIRST, even when we're about to reject the call. Leaving it
        # unread desynchronises a keep-alive connection, and because the proxy pools upstream
        # connections the leftover bytes get parsed as the NEXT request's request-line — so one
        # unauthorised call would break the next legitimate one with a bogus 501.
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""

        if not self._authed():
            self._json(401, {"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32001, "message": "Unauthorized: valid key required"}})
            return
        try:
            msg = json.loads(raw or b"{}")
        except Exception as e:
            self._json(400, {"jsonrpc": "2.0", "id": None,
                             "error": {"code": -32700, "message": f"Parse error: {e}"}})
            return
        reply = mcp_handle(msg)
        if reply is None:                              # notification → 202, no body
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._json(200, reply)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            body = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")   # we're iterating on this file
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path.startswith("/api/track"):
            q = urllib.parse.parse_qs(u.query)
            sid = (q.get("sid") or [""])[0].strip()
            claim = (q.get("claim") or [""])[0].strip()
            if not sid:
                self._json(400, {"error": "missing sid"})
                return
            try:
                if u.path == "/api/tracked":
                    out = track_list(sid)
                elif u.path == "/api/track/remove":
                    out = track_remove(sid, claim)
                elif u.path == "/api/track/recheck":
                    out = track_recheck(sid, claim)
                else:                                  # /api/track/add — needs the verdict fields
                    def _f(k, dv=0.0):
                        try:
                            return float((q.get(k) or [dv])[0])
                        except (TypeError, ValueError):
                            return dv
                    out = track_add(sid, claim, {"score": int(_f("score", 50)),
                                                 "confidence": _f("conf", 0.0),
                                                 "label": (q.get("label") or [""])[0]})
            except Exception as e:
                self._json(500, {"error": str(e)[:200]})
                return
            self._json(200, {"tracked": out})
            return
        if u.path == "/api/mcpinfo":
            # PoC convenience: surfaces the demo key so the page can show copy-paste
            # connect instructions. Do NOT do this once it's on a public host.
            # On a public deploy the key is NOT handed out to every visitor — they get
            # placeholder instructions. Locally (or with a key already in hand) we show it.
            # Behind a reverse proxy every request arrives from 127.0.0.1, so loopback alone
            # proves nothing — a forwarded-for header means a real external visitor.
            local = (self.client_address[0] in ("127.0.0.1", "::1")
                     and not self.headers.get("X-Forwarded-For"))
            k = next(iter(KEYS), "") if (local or self._authed()) else ""
            self._json(200, {"key": k or "<ask recommend for a key>", "issued": bool(k),
                             "url": f"http://localhost:{PORT}/mcp",
                             "tool": MCP_TOOLS[0]["name"], "n_keys": len(KEYS)})
            return
        if u.path in ("/api/check", "/api/compare"):
            qs = urllib.parse.parse_qs(u.query)
            claim = (qs.get("claim") or [""])[0].strip()
            deep = (qs.get("deep") or ["1"])[0] != "0"

            def _f(k, d=0.0):
                try:
                    return float((qs.get(k) or [d])[0])
                except (TypeError, ValueError):
                    return d

            grounded = {"p_true": _f("p_true", 0.5), "confidence": _f("conf", 0.0),
                        "label": (qs.get("label") or [""])[0],
                        "n_sources": int(_f("n_sources", 0))}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            q = queue.Queue()

            def emit(ev, data):
                q.put(f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

            def work():
                ip = (self.headers.get("X-Forwarded-For") or
                      self.client_address[0]).split(",")[0].strip()
                allowed, why = rate_ok(ip)
                try:
                    if not claim:
                        emit("error", {"msg": "Empty claim."})
                    elif not allowed:
                        emit("error", {"msg": why})
                    elif u.path == "/api/compare":
                        compare(claim, grounded, emit)
                    else:
                        run_check(claim, emit, use_deep=deep)
                except Exception as e:
                    emit("error", {"msg": f"{type(e).__name__}: {e}"[:300]})
                q.put(None)

            threading.Thread(target=work, daemon=True).start()
            while True:
                # A lane can take 20s+, during which nothing would be written. An idle
                # connection gets dropped by the proxy (and looks dead to the browser), which
                # is why this failed intermittently and got worse under load. A comment line
                # every few seconds keeps it alive; SSE clients ignore lines starting with ":".
                try:
                    chunk = q.get(timeout=5)
                except queue.Empty:
                    chunk = ": keepalive\n\n"
                if chunk is None:
                    break
                try:
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break                            # user navigated away mid-check
            return
        self.send_error(404)


if __name__ == "__main__":
    config.load()
    KEYS = load_keys()
    print(f"  GEMINI          : {'ok' if gemini.configured() else 'MISSING (required — see .env.example)'}")
    for k in ("EXA_API_KEY", "SERPAPI_API_KEY", "PARALLEL_API_KEY"):
        print(f"  {k:16}: {'ok' if config.get(k) else 'missing (lane will sit out)'}")
    threading.Thread(target=_track_daemon, daemon=True).start()
    print(f"\n  Confidence Check → http://localhost:{PORT}")
    print(f"  tracking         → re-check every {TRACK_EVERY_H}h (TRACK_EVERY_H to override)")
    print(f"  MCP endpoint     → http://localhost:{PORT}/mcp")
    for key, label in KEYS.items():
        print(f"  key ({label})     → {key}")
    print()
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), PORT), Handler).serve_forever()
