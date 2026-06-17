"""Error handling — classification, retry, and the loop guard.

Two ideas the harness must get right:
  * transient vs terminal — retry the first, surface the second.
  * loop guard — an open loop with no progress check burns budget forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class HarnessError(Exception):
    pass


class TransientError(HarnessError):
    """Retryable: rate limit, timeout, flaky network."""


class TerminalError(HarnessError):
    """Not retryable: bad request, auth failure, logic bug."""


def with_retry(fn, max_attempts: int = 3):
    """Retry transient failures with a deterministic (no-sleep) backoff count."""
    attempts = 0
    last = None
    while attempts < max_attempts:
        attempts += 1
        try:
            return {"ok": True, "value": fn(), "attempts": attempts}
        except TransientError as e:  # retry
            last = e
            continue
        except Exception as e:  # noqa: BLE001 — terminal: surface immediately
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "attempts": attempts}
    return {"ok": False, "error": f"gave up after {attempts}: {last}", "attempts": attempts}


@dataclass
class LoopGuard:
    """Trips when the last N actions are identical (no progress)."""

    no_progress_limit: int = 3
    history: list = field(default_factory=list)

    def record(self, action_signature: str) -> bool:
        """Record an action; return True if the guard should trip."""
        self.history.append(action_signature)
        if len(self.history) >= self.no_progress_limit:
            window = self.history[-self.no_progress_limit:]
            if len(set(window)) == 1:
                return True
        return False


def _self_test() -> None:
    # transient retries then succeeds
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise TransientError("rate limited")
        return "ok"

    r = with_retry(flaky, max_attempts=5)
    assert r["ok"] and r["value"] == "ok" and r["attempts"] == 3, r

    # terminal surfaces immediately, no retry
    def boom():
        raise TerminalError("bad request")

    r2 = with_retry(boom, max_attempts=5)
    assert (not r2["ok"]) and r2["attempts"] == 1 and "bad request" in r2["error"]

    # exhausts attempts on persistent transient
    r3 = with_retry(lambda: (_ for _ in ()).throw(TransientError("x")), max_attempts=2)
    assert not r3["ok"] and r3["attempts"] == 2

    # loop guard trips on 3 identical
    g = LoopGuard(no_progress_limit=3)
    assert not g.record("a")
    assert not g.record("a")
    assert g.record("a"), "should trip on 3rd identical"
    g2 = LoopGuard(no_progress_limit=3)
    for sig in ("a", "b", "a"):
        assert not g2.record(sig)
    print("[OK] harness.errors._self_test passed")


if __name__ == "__main__":
    _self_test()
