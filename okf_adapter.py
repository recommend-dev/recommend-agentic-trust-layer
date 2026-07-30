"""OKF adapter — citation-integrity mode (Ivan's spec): judge a bundle's claims against
the sources the bundle itself declares.

okf_verify.py answers "is this claim true in the world?" (evidence = live web).
This adapter answers the OKF-native question: "does the source this concept CITES
actually state what the claim says?" — evidence = the bundle's own `sources` entries,
resolved to file text (bundle-relative) or fetched full text (external URL). The
pipeline core is untouched: the same grounded judge and adversarial challenge from
server.py do the judging; this file is ingest + emit only.

Per-claim verdicts (Ivan's taxonomy):
  supports      — the cited source states it (survived the challenge)
  contradicts   — the cited source says otherwise
  related-only  — on-topic but does not actually state the claim
  no-evidence   — nothing in the cited source addresses it
  unverifiable  — the cited source could not be read (unreachable URL)

Reason-over-verdict backstop (mechanical): a `supports` whose own written summary
denies support ("does not address", "never states", …) is downgraded — the reason
outranks the label, the same rule the EuroVilla evidence gate runs.

The stamp gate: a concept gets `verified: {by: recommend-trust-layer/…}` only when
every CITED claim survives as `supports` and nothing contradicts. Anything else —
including "the source is merely related" — leaves the concept unstamped, with the
reasons in the report. Deprecated concepts and Attested Computations are out of
scope by design (noted, never judged: SQL-equality is different machinery).

Usage:  python3 okf_adapter.py path/to/bundle [--report-dir DIR] [--dry] [--limit N]
Keys:   GEMINI_API_KEY (or Vertex) only — no search APIs needed for this mode.
"""
import argparse
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import okf_verify                                 # noqa: E402  (frontmatter + stamp machinery)
import server                                     # noqa: E402  (judge + challenge — the core)

ACTOR = okf_verify.ACTOR
# a `supports` whose reason contains one of these is lying about itself → downgrade
_DENIAL = re.compile(r"does not|doesn't|not address|no mention|never (?:states|mentions)|"
                     r"fails to|not stated|not specified|silent on", re.I)

ATTRIBUTION_PROMPT = """You check CITATION INTEGRITY. Below is a claim from a document, and the
full text of the source that the claim CITES. The question is NOT whether the claim is true —
only whether THIS source actually states it.

Answer strictly from the source text. Paraphrase is fine — the claim's informational content
must be present in the source. If the source states only part of it, or merely mentions the
same terms without asserting the claim's content, answer "partial". If it is about something
else entirely, answer "no".

Return JSON only: {"stated": "<yes|partial|no>", "reason": "<one sentence>"}

CLAIM: {CLAIM}

CITED SOURCE ({TITLE}):
{EVIDENCE}"""


def attribution_check(claim, evidence, src_title):
    """The wrong-citation catch: a claim can be perfectly true and still cite a source
    that never says it. World-truth judging misses that case by design (true is true),
    so citation mode asks the narrower question explicitly."""
    d = okf_verify.gemini.gen_json(
        ATTRIBUTION_PROMPT.replace("{CLAIM}", claim)
                          .replace("{TITLE}", src_title)
                          .replace("{EVIDENCE}", (evidence or "")[:12000]),
        max_tokens=250)
    return (d.get("stated") or "").lower(), (d.get("reason") or "")[:250]


# ── ingest: sources ───────────────────────────────────────────────────────────
def parse_sources(fm):
    """Parse the frontmatter `sources:` block into [{id, resource, title}].
    Line-based on purpose: concepts are agent-rewritten constantly, and a strict YAML
    parse dying on one odd entry would silently skip a whole concept's evidence."""
    out, cur, in_block = [], None, False
    for line in fm.splitlines():
        if re.match(r"^sources:\s*$", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^[A-Za-z_]", line):     # next top-level key → block over
                break                              # (list items may sit at indent 0: "- resource: …")
            m = re.match(r"^\s*-\s+(\w+):\s*(.+?)\s*$", line)
            if m:                                  # new entry, first key inline
                cur = {m.group(1): m.group(2)}
                out.append(cur)
                continue
            m = re.match(r"^\s+(\w+):\s*(.+?)\s*$", line)
            if m and cur is not None:
                cur[m.group(1)] = m.group(2)
    return out


def resolve_source(entry, bundle, cache):
    """Source id → readable evidence text, or None when the resource can't be read.
    Bundle-relative paths are read directly (frontmatter stripped — the evidence is
    what the source SAYS, not its own metadata); external URLs are fetched full-text."""
    res = (entry.get("resource") or "").strip()
    if res in cache:
        return cache[res]
    text = None
    if res and not res.startswith(("http://", "https://")):
        p = os.path.join(bundle, res)
        if os.path.exists(p):
            raw = open(p, encoding="utf-8").read()
            _fm, body = okf_verify.split_frontmatter(raw)
            text = body.strip()
    elif res:
        try:
            req = urllib.request.Request(res, headers={"User-Agent": "recommend-trust-layer/0.1"})
            raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "replace")
            text = re.sub(r"<[^>]+>", " ", raw)   # crude de-tag; full text beats snippets
            text = re.sub(r"\s+", " ", text)[:14000]
        except Exception:
            text = None
    cache[res] = text
    return text


# ── ingest: claims ────────────────────────────────────────────────────────────
_FOOT_DEF = re.compile(r"^\[\^[\w\-]+\]:", re.M)
_FOOT_REF = re.compile(r"\[\^([\w\-]+)\]")


def _clean(line):
    """Markdown table row / bullet → plain claim text the judge can read."""
    t = _FOOT_REF.sub("", line)
    t = re.sub(r"^\s*[-*]\s+", "", t)
    if "|" in t:
        cells = [c.strip() for c in t.strip().strip("|").split("|")]
        t = " — ".join(c for c in cells if c)
    return re.sub(r"\s+", " ", t).replace("`", "").strip(" -—")


def extract_citation_claims(body):
    """Return (cited, uncited): cited = [(claim_text, [source_ids])] from lines carrying
    [^id] markers; uncited = assertion-looking lines (table rows, bullets) with none.
    Deterministic extraction, no LLM — the demo's credibility rests on the claim list
    being reproducible and auditable, not on a model's mood."""
    cited, uncited = [], []
    for line in body.splitlines():
        if _FOOT_DEF.match(line):                  # footnote definitions, not claims
            continue
        ids = _FOOT_REF.findall(line)
        text = _clean(line)
        if len(text) < 25 or text.lower().startswith(("column — type", "| column")):
            continue
        if ids:
            cited.append((text, ids))
        elif (line.strip().startswith(("|", "-", "*")) and len(text) >= 40
              and not text.startswith("[")):
            uncited.append(text)
    return cited, uncited


# ── judge one claim against one evidence text ────────────────────────────────
def judge_against(claim, evidence, src_title):
    if evidence is None:
        return {"verdict": "unverifiable", "reason": f"source '{src_title}' could not be read"}
    j = server.judge(claim, f"[{src_title}]\n{evidence}")
    j = server.challenge(claim, j, evidence)
    stance, strength = j["stance"], j["strength"]
    if stance == "support" and strength > 0.3:
        verdict = "supports"
    elif stance == "refute":
        verdict = "contradicts"
    elif stance == "mixed":
        verdict = "related-only"
    else:
        verdict = "no-evidence" if strength <= 0.1 else "related-only"
    reason = j.get("summary") or ""
    if verdict == "supports" and _DENIAL.search(reason):
        verdict = "related-only"
        reason = f"[downgraded: reason denies support] {reason}"
    # a supported claim must ALSO pass the narrower question: does the cited source
    # actually state it? True-but-miscited is an attribution failure, not a support.
    if verdict == "supports":
        stated, why = attribution_check(claim, evidence, src_title)
        if stated == "partial":
            verdict = "related-only"
            reason = f"[source only partially states this] {why}"
        elif stated == "no":
            verdict = "no-evidence"
            reason = f"[cited source does not state this] {why}"
    if j.get("challenge"):
        reason += f" | challenged: {j['challenge']}"
    return {"verdict": verdict, "reason": reason[:300], "quote": j.get("quote", "")}


# ── per-concept flow ─────────────────────────────────────────────────────────
def process_concept(path, bundle, cache, on_row=print):
    rel = os.path.relpath(path, bundle)
    text = open(path, encoding="utf-8").read()
    fm, body = okf_verify.split_frontmatter(text)
    if fm is None:
        return {"rel": rel, "outcome": "skipped", "note": "no frontmatter", "rows": []}
    title = okf_verify.fm_get(fm, "title") or rel
    ctype = (okf_verify.fm_get(fm, "type") or "").lower()
    status = (okf_verify.fm_get(fm, "status") or "stable").lower()
    if status == "deprecated":
        return {"rel": rel, "title": title, "outcome": "skipped",
                "note": "status: deprecated — kept for history, not judged", "rows": []}
    if "attested computation" in ctype:
        return {"rel": rel, "title": title, "outcome": "skipped",
                "note": "Attested Computation — deterministic SQL checking, not our lane", "rows": []}

    sources = parse_sources(fm)
    by_id = {s.get("id"): s for s in sources if s.get("id")}
    cited, uncited = extract_citation_claims(body)
    if not cited and not uncited:
        return {"rel": rel, "title": title, "outcome": "skipped",
                "note": "no claims found in body", "rows": []}
    if cited and not sources:
        return {"rel": rel, "title": title, "outcome": "failed",
                "note": "body carries citations but frontmatter declares no sources", "rows": []}

    rows = []
    for claim, ids in cited:
        for sid in ids:
            entry = by_id.get(sid)
            if entry is None:
                # numbered [^1]-style footnotes not keyed to sources[].id — a bundle
                # conformance gap (SPEC §5.1 wants keyed labels), not a false claim
                rows.append({"claim": claim, "src": sid, "verdict": "unverifiable",
                             "reason": f"citation [^{sid}] has no matching sources entry "
                                       f"(SPEC §5.1 expects footnote labels keyed to sources[].id)"})
                continue
            ev = resolve_source(entry, bundle, cache)
            r = judge_against(claim, ev, entry.get("title") or sid)
            rows.append({"claim": claim, "src": sid, **r})
            on_row(rel, rows[-1])
    for claim in uncited:
        ev_parts = []
        for s in sources:
            ev = resolve_source(s, bundle, cache)
            if ev:
                ev_parts.append(f"[{s.get('title') or s.get('id')}]\n{ev[:6000]}")
        if not ev_parts:
            rows.append({"claim": claim, "src": "(uncited)", "verdict": "no-evidence",
                         "reason": "uncited claim and no readable source to check it against"})
            continue
        r = judge_against(claim, "\n\n".join(ev_parts), "all declared sources")
        rows.append({"claim": claim, "src": "(uncited)", **r,
                     "reason": "[uncited claim] " + r["reason"]})
        on_row(rel, rows[-1])

    cited_rows = [r for r in rows if r["src"] != "(uncited)"]
    contradicts = [r for r in rows if r["verdict"] == "contradicts"]
    weak_cited = [r for r in cited_rows if r["verdict"] != "supports"]
    if contradicts:
        outcome = "failed"
    elif cited_rows and not weak_cited:
        outcome = "verified"
    elif not cited_rows:
        outcome = "inconclusive"       # only uncited material — flagged, never stamped
    else:
        outcome = "inconclusive"
    return {"rel": rel, "title": title, "outcome": outcome, "rows": rows,
            "fm": fm, "body": body, "note": ""}


# ── emit ─────────────────────────────────────────────────────────────────────
def stamp(fm, at):
    return okf_verify.append_verified(fm, at)


def concept_report(c, at):
    lines = [f"# Verification report — {c['title']}", "",
             f"*{ACTOR} · {at} · mode: citation-integrity (bundle sources as evidence)*", "",
             f"**Outcome: {c['outcome'].upper()}**", ""]
    if c.get("note"):
        lines += [c["note"], ""]
    for r in c["rows"]:
        lines += [f"- **{r['verdict']}** · `[^{r['src']}]` — {r['claim']}",
                  f"  - {r['reason']}"]
        if r.get("quote"):
            lines.append(f"  - evidence quote: “{r['quote'][:200]}”")
    return "\n".join(lines) + "\n"


def diff_table(concepts):
    lines = ["| Concept | Claims checked | Survived | Caught | Outcome |",
             "|---|---|---|---|---|"]
    for c in concepts:
        n = len(c["rows"])
        ok = sum(1 for r in c["rows"] if r["verdict"] == "supports")
        bad = sum(1 for r in c["rows"] if r["verdict"] in ("contradicts", "related-only",
                                                           "no-evidence", "unverifiable"))
        note = f" — {c['note']}" if c.get("note") else ""
        lines.append(f"| `{c['rel']}` | {n} | {ok} | {bad} | **{c['outcome']}**{note} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Judge an OKF bundle's claims against its own declared sources.")
    ap.add_argument("bundle")
    ap.add_argument("--report-dir", default=None,
                    help="where reports go (default: <bundle>-reports, sibling dir)")
    ap.add_argument("--dry", action="store_true", help="judge and report, don't touch the bundle")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    report_dir = os.path.abspath(args.report_dir or bundle.rstrip("/") + "-reports")
    concepts_paths = okf_verify.find_concepts(bundle)
    if args.limit:
        concepts_paths = concepts_paths[:args.limit]
    cache, results = {}, []
    at = okf_verify.now_iso()
    print(f"bundle: {bundle}\nconcepts: {len(concepts_paths)}  ·  mode: citation-integrity"
          f"{'  ·  DRY RUN' if args.dry else ''}\n")

    def on_row(rel, r):
        print(f"   [{r['verdict']:12}] [^{r['src']}] {r['claim'][:70]}")

    for path in concepts_paths:
        c = process_concept(path, bundle, cache, on_row=lambda rel, r: on_row(rel, r))
        results.append(c)
        print(f"● {c['rel']} → {c['outcome'].upper()}"
              + (f"  ({c['note']})" if c.get("note") else ""))
        if not args.dry and c["outcome"] == "verified":
            okf_verify.write_concept(path, stamp(c["fm"], at), c["body"])

    os.makedirs(report_dir, exist_ok=True)
    for c in results:
        if c["rows"] or c.get("note"):
            name = c["rel"].replace(os.sep, "__").replace(".md", "") + ".report.md"
            open(os.path.join(report_dir, name), "w", encoding="utf-8").write(concept_report(c, at))
    table = diff_table(results)
    open(os.path.join(report_dir, "REPORT.md"), "w", encoding="utf-8").write(
        f"# OKF citation-integrity run — {os.path.basename(bundle)}\n\n"
        f"*{ACTOR} · {at}*\n\n{table}\n")
    print(f"\n{table}\n\nreports → {report_dir}")


if __name__ == "__main__":
    main()
