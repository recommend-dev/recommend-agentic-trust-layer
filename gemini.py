"""Self-contained Gemini helper. Two auth paths, checked in this order:

  1. GEMINI_API_KEY                → Gemini API (get a free key at https://aistudio.google.com/apikey)
  2. GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT → Vertex AI (service-account JSON)

Model/location are env-overridable (GEMINI_MODEL, GEMINI_LOCATION). Everything is
resolved at call time, not import time, so a .env loaded after import still counts.
"""
import datetime, json, os, re, time, urllib.error, urllib.request

_tok = None
_tok_exp = 0.0          # Vertex access tokens last ~1h; a long-running server MUST refresh


def model_name():
    return os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")


def configured():
    """True if either auth path has what it needs."""
    return bool(os.environ.get("GEMINI_API_KEY") or
                (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                 and os.environ.get("GOOGLE_CLOUD_PROJECT")))


def _token(force=False):
    """Vertex path only. Cached, but with an expiry: caching it forever is fine in a batch
    script and fatal in a server — after an hour every call comes back 401 Unauthorized."""
    global _tok, _tok_exp
    if _tok and not force and time.time() < _tok_exp:
        return _tok
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "No Gemini credentials. Set GEMINI_API_KEY (free: https://aistudio.google.com/apikey) "
            "or GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT in .env.")
    c = service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    c.refresh(Request())
    _tok = c.token
    exp = getattr(c, "expiry", None)
    # google-auth returns a NAIVE datetime that is already UTC; .timestamp() would read it as
    # local time and land an hour or two in the past, so the cache would never hold.
    if exp is not None:
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=datetime.timezone.utc)
        _tok_exp = exp.timestamp() - 60          # renew a minute early
    else:
        _tok_exp = time.time() + 3540
    return _tok


def _request(body):
    """Build the generateContent request for whichever auth path is configured."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_name()}:generateContent")
        return urllib.request.Request(url, data=json.dumps(body).encode(),
                                      headers={"x-goog-api-key": api_key,
                                               "Content-Type": "application/json"})
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GEMINI_LOCATION", "global")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    url = (f"https://{host}/v1/projects/{project}/locations/{location}"
           f"/publishers/google/models/{model_name()}:generateContent")
    return urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Authorization": f"Bearer {_token()}",
                                           "Content-Type": "application/json"})


def gen(prompt, temperature=0.2, max_tokens=700):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
    for attempt in range(6):
        req = _request(body)
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)
        except urllib.error.HTTPError as e:
            if e.code == 401 and not os.environ.get("GEMINI_API_KEY") and attempt < 5:
                _token(force=True)          # Vertex token died mid-flight → mint a fresh one
                continue
            if e.code in (429, 500, 503) and attempt < 5:
                time.sleep(2 ** attempt + 1); continue
            raise
    return ""


def gen_json(prompt, **kw):
    t = re.sub(r"```json|```", "", gen(prompt, **kw)).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1]) if s >= 0 else {}
    except Exception:
        return {}
