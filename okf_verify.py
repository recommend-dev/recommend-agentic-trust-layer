"""OKF v0.2 verifier — fill `verified` with an evidence-backed verdict, not a signature.

Google's Open Knowledge Format (v0.2) records who *generated* a concept and who
*verified* it — but "verified" is a name typed into a field. This tool is the machine
behind the field: it walks an OKF bundle, extracts each concept's externally checkable
claims, runs every claim through the trust pipeline (independent evidence lanes →
grounded judge → adversarial challenge → calibrated confidence), and then:

  * every checked claim SUPPORTED  → appends `verified: {by: recommend-trust-layer/<v>}`
    (a machine-confirmed trust tier per SPEC §5.3) plus an `x_verification` extension
    block with per-claim scores and sources,
  * any claim REFUTED              → does NOT stamp verified. A verifier that stamps
    everything is a rubber stamp; the failure lands in `x_verification` and log.md
    so a human can fix or deprecate the concept,
  * nothing externally checkable   → leaves the concept untouched. Internal definitions
    ("how WE compute margin") have no web ground truth; claiming to verify them would
    be the exact self-reported-signature problem this tool exists to end.

Every run appends a dated report to the bundle's log.md (SPEC §9: chronological history).

Usage:  python3 okf_verify.py path/to/bundle [--deep] [--dry] [--limit N]
        --dry    judge and report, write nothing
        --deep   include the deep-research lane (slower, better citations)
        --limit  stop after N concepts (spend guard while trying it out)
Keys:   same .env as server.py — GEMINI_API_KEY + EXA_API_KEY is the working minimum.
"""
import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gemini                                     # noqa: E402
import server                                     # noqa: E402  (pipeline: run_check)

ACTOR = "recommend-trust-layer/0.1"
SUPPORT_MIN = 60      # score at/above which a claim counts as supported
REFUTE_MAX = 40       # score at/below which a claim counts as refuted
CONF_MIN = 0.35       # below this confidence a verdict is treated as inconclusive
SKIP_FILES = {"log.md", "index.md", "README.md"}   # structure, not knowledge


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── bundle walking ────────────────────────────────────────────────────────────
def find_concepts(bundle):
    out = []
    for root, dirs, files in os.walk(bundle):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "references"]
        for f in sorted(files):
            if f.endswith(".md") and f not in SKIP_FILES:
                out.append(os.path.join(root, f))
    return out


def split_frontmatter(text):
    """Return (frontmatter_text, body) or (None, text) when there is no frontmatter."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    return (m.group(1), m.group(2)) if m else (None, text)


def fm_get(fm, key):
    """Flat scalar lookup ('title: X') — enough for the fields this tool reads."""
    m = re.search(rf"^{key}:\s*(.+?)\s*$", fm, re.M)
    return m.group(1).strip().strip('"').strip("'") if m else None


# ── claim extraction ──────────────────────────────────────────────────────────
EXTRACT_PROMPT = """You are the intake step of a knowledge-base verifier.

Below is one concept file from an organization's knowledge base (OKF). Extract the
claims in it that are EXTERNALLY CHECKABLE — statements about the public world whose
truth does not depend on this organization's internal choices.

DO extract: facts about markets, countries, laws, products, companies, science,
public statistics, third-party tools and their behaviour.
Do NOT extract: internal definitions ("we define X as"), internal policies, schema
descriptions, thresholds the organization chose, or anything only their own systems
could confirm.

Return JSON only: {{"claims": ["...", "..."]}}  (0 to 3 claims, each a single
self-contained declarative sentence; resolve pronouns and internal shorthand).
If nothing is externally checkable, return {{"claims": []}}.

CONCEPT ({title}):
{text}"""


def extract_claims(title, fm, body):
    text = (fm or "") + "\n\n" + body
    d = gemini.gen_json(EXTRACT_PROMPT.format(title=title, text=text[:6000]),
                        max_tokens=500)
    return [c.strip() for c in (d.get("claims") or []) if isinstance(c, str) and c.strip()][:3]


# ── verification ──────────────────────────────────────────────────────────────
def check_claim(claim, deep=False, on_event=None):
    """Run one claim through the full pipeline; return the verdict event plus the
    cited sources collected across lanes (the evidence behind the stamp).
    on_event(ev, data), when given, sees every pipeline event — the OKF UI uses it
    to show lanes landing live."""
    verdict, sources, seen = {}, [], set()

    def emit(ev, data):
        if ev == "verdict":
            verdict.update(data)
        elif ev == "lane_done":
            for s in data.get("sources") or []:
                u = s.get("url")
                if u and u not in seen:
                    seen.add(u)
                    sources.append(u)
        if on_event:
            on_event(ev, data)

    server.run_check(claim, emit, use_deep=deep)
    verdict["_sources"] = sources[:3]
    return verdict


def classify(v):
    score, conf = v.get("score"), v.get("confidence") or 0
    if v.get("unverifiable") or score is None or conf < CONF_MIN:
        return "inconclusive"
    if score >= SUPPORT_MIN:
        return "supported"
    if score <= REFUTE_MAX:
        return "refuted"
    return "inconclusive"


# ── frontmatter writing ───────────────────────────────────────────────────────
def append_verified(fm, at):
    """Append a verification event per SPEC §5.2, preserving any existing entries."""
    entry = f"  - {{ by: {ACTOR}, at: {at} }}"
    m = re.search(r"^verified:\s*(\{.*\})\s*$", fm, re.M)
    if m:  # bare mapping → one-element list (SPEC treats them the same), then append
        return fm[:m.start()] + f"verified:\n  - {m.group(1)}\n{entry}" + fm[m.end():]
    m = re.search(r"^verified:\s*\n((?:[ \t]+-.*\n?)+)", fm, re.M)
    if m:
        block = m.group(0).rstrip("\n")
        return fm[:m.start()] + block + "\n" + entry + fm[m.start() + len(block):]
    return fm.rstrip("\n") + f"\nverified:\n{entry}\n"


def append_x_verification(fm, at, results):
    """Extension block: the evidence behind the stamp. Namespaced x_ so OKF consumers
    that don't know it can ignore it; ones that do get scores and sources, not a name."""
    lines = ["x_verification:", f"  by: {ACTOR}", f"  at: {at}", "  claims:"]
    for claim, v, cls in results:
        lines += [f"    - claim: {yaml_str(claim)}",
                  f"      verdict: {cls}",
                  f"      label: {v.get('label', 'N/A')}",
                  f"      truth_score: {v.get('score')}",
                  f"      confidence: {v.get('confidence')}"]
        if v.get("_sources"):
            lines.append("      sources:")
            lines += [f"        - {u}" for u in v["_sources"]]
    return fm.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def yaml_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_concept(path, fm, body):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"---\n{fm.rstrip()}\n---\n{body}")


def append_log(bundle, report_lines):
    log = os.path.join(bundle, "log.md")
    stamp = time.strftime("%Y-%m-%d")
    block = [f"\n## {stamp} — verification run ({ACTOR})\n"] + report_lines + [""]
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n".join(block))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Verify an OKF bundle's externally checkable claims.")
    ap.add_argument("bundle")
    ap.add_argument("--deep", action="store_true", help="include the deep-research lane")
    ap.add_argument("--dry", action="store_true", help="judge and report, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N concepts")
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    concepts = find_concepts(bundle)
    if args.limit:
        concepts = concepts[:args.limit]
    print(f"bundle: {bundle}\nconcepts: {len(concepts)}"
          f"{' (dry run — nothing will be written)' if args.dry else ''}\n")

    log_lines = []
    for path in concepts:
        rel = os.path.relpath(path, bundle)
        text = open(path, encoding="utf-8").read()
        fm, body = split_frontmatter(text)
        if fm is None:
            print(f"— {rel}: no frontmatter, skipped")
            continue
        title = fm_get(fm, "title") or rel

        claims = extract_claims(title, fm, body)
        if not claims:
            print(f"— {rel}: no externally checkable claims — left untouched")
            log_lines.append(f"- `{rel}`: nothing externally checkable; not stamped.")
            continue

        print(f"● {rel} — {len(claims)} claim(s)")
        results = []
        for c in claims:
            v = check_claim(c, deep=args.deep)
            cls = classify(v)
            results.append((c, v, cls))
            print(f"   [{cls.upper():12}] {v.get('label', '?'):22} score {v.get('score')} "
                  f"conf {v.get('confidence')}  ← {c[:80]}")

        classes = [cls for _c, _v, cls in results]
        at = now_iso()
        if all(c == "supported" for c in classes):
            outcome = "VERIFIED"
            if not args.dry:
                fm2 = append_verified(fm, at)
                fm2 = append_x_verification(fm2, at, results)
                write_concept(path, fm2, body)
        elif "refuted" in classes:
            outcome = "FAILED — refuted claim, verified NOT stamped"
            if not args.dry:
                fm2 = append_x_verification(fm, at, results)
                write_concept(path, fm2, body)
        else:
            outcome = "INCONCLUSIVE — verified not stamped"
            if not args.dry:
                fm2 = append_x_verification(fm, at, results)
                write_concept(path, fm2, body)
        print(f"   → {outcome}\n")

        log_lines.append(f"- `{rel}`: **{outcome}**")
        for c, v, cls in results:
            log_lines.append(f"  - {cls} ({v.get('score')}/100, conf {v.get('confidence')}): {c}")

    if not args.dry and log_lines:
        append_log(bundle, log_lines)
        print(f"report appended to {os.path.join(bundle, 'log.md')}")


if __name__ == "__main__":
    main()
