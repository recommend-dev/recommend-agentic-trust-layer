"""Retry with exponential backoff + jitter. Retries transient failures (timeouts,
connection errors, HTTP 429 / 5xx) and gives up on permanent ones (4xx except 429)."""
import functools
import random
import time

import requests


class RetryableError(Exception):
    """Raise to force a retry (e.g. provider returned a retryable status)."""


# transient HTTP statuses worth retrying
RETRY_STATUS = {429, 500, 502, 503, 504, 408}
# exception types that are always transient
TRANSIENT_EXC = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    RetryableError,
)


def retry(attempts: int = 5, base_delay: float = 1.0, max_delay: float = 30.0,
          factor: float = 2.0, jitter: float = 0.3):
    """Decorator. Retries `attempts` times on transient errors with capped exp backoff.

    `requests.HTTPError` is retried only if its status is in RETRY_STATUS; honours a
    `Retry-After` header when present.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last = None
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last = e
                    status = getattr(e.response, "status_code", None)
                    if status not in RETRY_STATUS or i == attempts:
                        raise
                    ra = _retry_after(e.response)
                    sleep_for = ra if ra is not None else _backoff(delay, max_delay, jitter)
                except TRANSIENT_EXC as e:
                    last = e
                    if i == attempts:
                        raise
                    sleep_for = _backoff(delay, max_delay, jitter)
                time.sleep(sleep_for)
                delay = min(max_delay, delay * factor)
            raise last  # pragma: no cover
        return wrapper
    return deco


def _backoff(delay: float, max_delay: float, jitter: float) -> float:
    d = min(max_delay, delay)
    return d + random.uniform(0, d * jitter)


def _retry_after(resp) -> float | None:
    try:
        v = resp.headers.get("Retry-After")
        return float(v) if v else None
    except (TypeError, ValueError):
        return None
