"""Parallel.ai Task API adapter — structured web-research read.

One structured Task per claim: create run (processor tier) → poll /result → parse
output.content (schema fields) + output.basis[] (per-field confidence/reasoning/citations).
Key = PARALLEL_API_KEY in .env. Docs: https://docs.parallel.ai
"""
import json
import time

import requests

import config

BASE = "https://api.parallel.ai"


def _headers():
    key = config.require("PARALLEL_API_KEY")
    return {"Authorization": f"Bearer {key}", "x-api-key": key, "Content-Type": "application/json"}


def task(input_text, schema, processor="core", poll=120, interval=5, timeout=40):
    """Run one Parallel Task; return {"content": {field: value}, "basis": {field: {...}}, "run_id"}.

    schema = a JSON Schema object placed at task_spec.output_schema.json_schema.
    Raises TimeoutError if the run doesn't complete within poll*interval seconds.
    """
    h = _headers()
    r = requests.post(f"{BASE}/v1/tasks/runs", headers=h, timeout=timeout,
                      json={"input": input_text, "processor": processor,
                            "task_spec": {"output_schema": {"type": "json", "json_schema": schema}}})
    r.raise_for_status()
    rid = r.json()["run_id"]
    out = None
    for _ in range(poll):
        try:
            g = requests.get(f"{BASE}/v1/tasks/runs/{rid}/result", headers=h, timeout=30)
        except requests.exceptions.RequestException:
            continue
        if g.status_code == 200 and g.json().get("output"):
            out = g.json()["output"]
            break
        time.sleep(interval)
    if not out:
        raise TimeoutError(f"Parallel timeout for run {rid}")
    content = out.get("content") or {}
    if isinstance(content, str):
        content = json.loads(content)
    basis = {b["field"]: b for b in out.get("basis", [])}
    return {"content": content, "basis": basis, "run_id": rid}


def dedup_citations(cits, k=4):
    """One citation per domain: [{title, url, dom, excerpt}]."""
    out, seen = [], set()
    for c in cits or []:
        url = c.get("url") or ""
        dom = url.split("//")[-1].split("/")[0].replace("www.", "")
        if not dom or dom in seen:
            continue
        seen.add(dom)
        out.append({"title": (c.get("title") or dom)[:90], "url": url, "dom": dom,
                    "excerpt": ((c.get("excerpts") or [c.get("text", "")])[0] or "")[:180]})
        if len(out) >= k:
            break
    return out
