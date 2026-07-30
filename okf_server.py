"""OKF verification UI — upload OKF concept files, watch the checker work, download stamps.

A deliberately separate server from server.py (the Claim Checker) so the public app
stays untouched: this one serves okf.html, accepts .md concept files, streams every
step over SSE (claims extracted → lanes landing per claim → verdict → stamp decision),
and hands back the stamped file when the evidence holds.

Run:  python3 okf_server.py        → http://localhost:8898
Keys: same .env as server.py.
"""
import json
import os
import queue
import secrets
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import okf_verify                                 # noqa: E402

PORT = int(os.environ.get("OKF_PORT", 8898))

# jobs live in memory: one browser, one machine, a demo — not a queue service
JOBS = {}
_jobs_lock = threading.Lock()


class Job:
    def __init__(self):
        self.q = queue.Queue()
        self.history = []          # replay buffer so a reconnecting EventSource catches up
        self.done = False

    def emit(self, ev, data):
        item = (ev, data)
        self.history.append(item)
        self.q.put(item)
        if ev == "done":
            self.done = True


def run_job(job, files, deep):
    """The okf_verify flow, narrated event by event instead of printed."""
    try:
        for f in files:
            name, text = f.get("name") or "concept.md", f.get("content") or ""
            fm, body = okf_verify.split_frontmatter(text)
            title = (okf_verify.fm_get(fm, "title") if fm else None) or name
            job.emit("file_start", {"name": name, "title": title, "has_fm": fm is not None})
            if fm is None:
                job.emit("file_done", {"name": name, "outcome": "skipped",
                                       "note": "No YAML frontmatter — not an OKF concept file."})
                continue

            claims = okf_verify.extract_claims(title, fm, body)
            job.emit("claims", {"name": name, "claims": claims})
            if not claims:
                job.emit("file_done", {"name": name, "outcome": "skipped",
                                       "note": "Nothing externally checkable — honestly left unstamped."})
                continue

            results = []
            for i, c in enumerate(claims):
                job.emit("claim_start", {"name": name, "i": i, "claim": c})

                def forward(ev, data, _i=i, _n=name):
                    if ev == "lanes":
                        job.emit("claim_lanes", {"name": _n, "i": _i, **data})
                    elif ev == "lane_done":
                        job.emit("claim_lane", {"name": _n, "i": _i,
                                                **{k: data.get(k) for k in
                                                   ("id", "lane", "stance", "strength", "error", "ms")}})

                v = okf_verify.check_claim(c, deep=deep, on_event=forward)
                cls = okf_verify.classify(v)
                results.append((c, v, cls))
                job.emit("claim_verdict", {"name": name, "i": i, "verdict": cls,
                                           "label": v.get("label"), "score": v.get("score"),
                                           "confidence": v.get("confidence"),
                                           "sources": v.get("_sources") or []})

            classes = [cls for _c, _v, cls in results]
            at = okf_verify.now_iso()
            if all(c == "supported" for c in classes):
                outcome = "verified"
                fm2 = okf_verify.append_verified(fm, at)
            elif "refuted" in classes:
                outcome = "failed"
                fm2 = fm
            else:
                outcome = "inconclusive"
                fm2 = fm
            fm2 = okf_verify.append_x_verification(fm2, at, results)
            stamped = f"---\n{fm2.rstrip()}\n---\n{body}"
            job.emit("file_done", {"name": name, "outcome": outcome, "stamped": stamped})
        job.emit("done", {})
    except Exception as e:                        # a broken file must not hang the stream
        job.emit("error", {"msg": f"{type(e).__name__}: {e}"[:300]})
        job.emit("done", {})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            page = open(os.path.join(HERE, "okf.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if u.path == "/api/stream":
            job_id = (urllib.parse.parse_qs(u.query).get("job") or [""])[0]
            with _jobs_lock:
                job = JOBS.get(job_id)
            if not job:
                self._json(404, {"error": "unknown job"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            sent = 0
            try:
                while True:
                    while sent < len(job.history):
                        ev, data = job.history[sent]
                        sent += 1
                        self.wfile.write(f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                        if ev == "done":
                            return
                    try:
                        job.q.get(timeout=5)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/verify":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                d = json.loads(self.rfile.read(n))
                files = d.get("files") or []
                assert files, "no files"
            except Exception as e:
                self._json(400, {"error": str(e)[:200]})
                return
            job = Job()
            job_id = secrets.token_urlsafe(8)
            with _jobs_lock:
                JOBS[job_id] = job
            threading.Thread(target=run_job, args=(job, files, bool(d.get("deep"))),
                             daemon=True).start()
            self._json(200, {"job": job_id})
            return
        self.send_error(404)


if __name__ == "__main__":
    okf_verify.server.config.load()
    print(f"  OKF verifier UI → http://localhost:{PORT}")
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), PORT), Handler).serve_forever()
