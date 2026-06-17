"""Context management — the window as a finite budget, with compaction.

Messages are plain dicts ({"role","content",...}). The window tracks token
usage and, when over budget, compacts older turns into a summary while keeping
system messages and the most recent turns verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .util import est_tokens, preview


def _default_summarize(messages) -> str:
    roles = []
    for m in messages:
        if m["role"] not in roles:
            roles.append(m["role"])
    toks = sum(est_tokens(m["content"]) for m in messages)
    return f"{len(messages)} earlier turns ({'/'.join(roles)}), ~{toks} tokens condensed"


@dataclass
class ContextWindow:
    budget: int
    messages: list = field(default_factory=list)

    def add(self, role: str, content, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def used(self) -> int:
        return sum(est_tokens(m["content"]) for m in self.messages)

    def over_budget(self) -> bool:
        return self.used() > self.budget

    def headroom(self) -> int:
        return self.budget - self.used()

    def compact(self, keep_recent: int = 3, summarizer=None) -> int:
        """Summarize old non-system turns. Returns tokens reclaimed."""
        system = [m for m in self.messages if m["role"] == "system"]
        rest = [m for m in self.messages if m["role"] != "system"]
        if len(rest) <= keep_recent:
            return 0
        old, recent = rest[:-keep_recent], rest[-keep_recent:]
        note = (summarizer or _default_summarize)(old)
        before = self.used()
        self.messages = system + [{"role": "system", "content": f"[summary] {note}"}] + recent
        return before - self.used()


def _self_test() -> None:
    w = ContextWindow(budget=40)
    w.add("system", "you are a helpful agent")
    for i in range(10):
        w.add("user", f"message number {i} with some padding content here")
    assert w.over_budget()
    before = w.used()
    last = w.messages[-1]

    reclaimed = w.compact(keep_recent=2)
    assert reclaimed > 0 and w.used() < before
    assert w.messages[-1] == last, "most recent turn must survive verbatim"
    assert w.messages[0]["role"] == "system" and "helpful agent" in w.messages[0]["content"]
    assert any("[summary]" in m["content"] for m in w.messages)

    # no-op when nothing to compact
    small = ContextWindow(budget=1000)
    small.add("user", "hi")
    assert small.compact() == 0
    print("[OK] harness.context._self_test passed")


if __name__ == "__main__":
    _self_test()
