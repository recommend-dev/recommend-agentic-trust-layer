"""Thread-safe token-bucket rate limiter. Shared across threads so a per-provider cap
(e.g. Diffbot 5 req/s) holds no matter how many worker threads call it concurrently."""
import threading
import time


class RateLimiter:
    """Allows up to `rate` operations per second (smoothed via a token bucket).

    Usage:
        limiter = RateLimiter(5)          # 5/s
        limiter.acquire()                 # blocks until a token is available
    """

    def __init__(self, rate: float, burst: float | None = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self.rate
            time.sleep(wait)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        return False
