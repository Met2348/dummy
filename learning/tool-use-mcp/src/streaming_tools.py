"""Streaming tool with cancel hook (sync generator version)."""
from __future__ import annotations
from typing import Iterator


class StreamingTool:
    def __init__(self):
        self.cancelled = False
        self.events: list[dict] = []

    def cancel(self) -> None:
        self.cancelled = True

    def reset(self) -> None:
        self.cancelled = False
        self.events.clear()

    def stream(self, n_steps: int = 5) -> Iterator[dict]:
        self.reset()
        for i in range(n_steps):
            if self.cancelled:
                ev = {"status": "cancelled", "at_step": i}
                self.events.append(ev)
                yield ev
                return
            ev = {"step": i + 1, "progress": (i + 1) / n_steps}
            self.events.append(ev)
            yield ev
        done = {"status": "done", "total": n_steps}
        self.events.append(done)
        yield done


def _self_test() -> None:
    t = StreamingTool()
    events = list(t.stream(n_steps=3))
    assert len(events) == 4, events
    assert events[-1]["status"] == "done"
    assert events[0]["step"] == 1

    t2 = StreamingTool()
    gen = t2.stream(n_steps=10)
    first = next(gen)
    assert first["step"] == 1
    t2.cancel()
    rest = list(gen)
    assert any(ev.get("status") == "cancelled" for ev in rest), rest
    print("[OK] streaming_tools._self_test passed")


if __name__ == "__main__":
    _self_test()
