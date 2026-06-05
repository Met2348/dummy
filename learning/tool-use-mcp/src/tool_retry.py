"""Exponential backoff + circuit breaker."""
from __future__ import annotations
import time
from typing import Callable


def with_retry(
    fn: Callable[[dict], dict],
    max_attempts: int = 4,
    base: float = 0.01,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[dict], dict]:
    """Decorate fn: retry transient failures with exponential backoff.

    Expects fn(args) -> dict with {ok: bool, error?: str}.
    Permanent errors (containing 'permanent' or 'bad_arg') NOT retried.
    """
    def wrapper(args: dict) -> dict:
        last: dict = {"ok": False, "error": "no attempts"}
        for attempt in range(max_attempts):
            result = fn(args)
            if result.get("ok"):
                result["attempts"] = attempt + 1
                return result
            last = result
            err = (result.get("error") or "").lower()
            if "permanent" in err or "bad_arg" in err:
                result["attempts"] = attempt + 1
                return result
            if attempt == max_attempts - 1:
                break
            sleep(base * (2 ** attempt))
        last["attempts"] = max_attempts
        return last
    return wrapper


class CircuitBreaker:
    def __init__(self, threshold: int = 3, timeout: float = 0.05, now_fn=time.monotonic):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.opened_at: float | None = None
        self._now = now_fn

    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self._now() - self.opened_at >= self.timeout:
            return "half_open"
        return "open"

    def call(self, fn: Callable[[], dict]) -> dict:
        st = self.state()
        if st == "open":
            return {"ok": False, "error": "circuit_open"}
        result = fn()
        if result.get("ok"):
            self.failures = 0
            self.opened_at = None
            return result
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self._now()
        return result


def _self_test() -> None:
    attempts = {"n": 0}
    def flaky(args):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return {"ok": False, "error": "timeout"}
        return {"ok": True, "value": "yes"}

    wrapped = with_retry(flaky, max_attempts=5, sleep=lambda x: None)
    r = wrapped({})
    assert r["ok"] and r["attempts"] == 3, r

    def permanent(args):
        return {"ok": False, "error": "permanent: bad_arg"}
    wrapped2 = with_retry(permanent, max_attempts=5, sleep=lambda x: None)
    r2 = wrapped2({})
    assert not r2["ok"] and r2["attempts"] == 1, r2

    fake_time = [0.0]
    cb = CircuitBreaker(threshold=2, timeout=0.05, now_fn=lambda: fake_time[0])
    for _ in range(2):
        cb.call(lambda: {"ok": False, "error": "x"})
    assert cb.state() == "open"
    res = cb.call(lambda: {"ok": True})
    assert res["error"] == "circuit_open", res

    fake_time[0] = 0.1
    assert cb.state() == "half_open"
    res2 = cb.call(lambda: {"ok": True})
    assert res2["ok"]
    assert cb.state() == "closed"
    print("[OK] tool_retry._self_test passed")


if __name__ == "__main__":
    _self_test()
