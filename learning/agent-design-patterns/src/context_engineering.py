"""Context engineering — treating the context window as a finite budget.

The 2025 reframing: a long-running agent's bottleneck isn't "can the model
reason", it's "does the right stuff fit in the window". Core moves:

  * budget accounting  — know how many tokens each message costs
  * compaction         — summarize old turns into a compact note
  * tool-output trimming — cap verbose tool results before they enter context
  * offload            — push detail to memory/sub-agents, keep the window lean
"""
from __future__ import annotations

from dataclasses import dataclass, field

from common import est_tokens, preview


@dataclass
class ContextWindow:
    budget: int
    messages: list = field(default_factory=list)  # list[(role, text)]

    def add(self, role: str, text: str) -> None:
        self.messages.append((role, text))

    def used(self) -> int:
        return sum(est_tokens(t) for _, t in self.messages)

    def over_budget(self) -> bool:
        return self.used() > self.budget

    def headroom(self) -> int:
        return self.budget - self.used()


def trim_tool_output(text: str, max_tokens: int) -> str:
    """Cap a verbose tool result. Keep head + tail, drop the middle —
    the pattern real harnesses use for huge file reads / command output."""
    if est_tokens(text) <= max_tokens:
        return text
    keep_chars = max_tokens * 4
    head = text[: keep_chars // 2]
    tail = text[-keep_chars // 2:]
    dropped = est_tokens(text) - max_tokens
    return f"{head}\n…[trimmed ~{dropped} tokens]…\n{tail}"


def compact(window: ContextWindow, keep_recent: int = 2, summarizer=None) -> ContextWindow:
    """Replace all but the last `keep_recent` messages with one summary note.
    `summarizer` is a deterministic mock here; in a real harness it's an LLM."""
    if len(window.messages) <= keep_recent:
        return window
    old = window.messages[:-keep_recent] if keep_recent else window.messages
    recent = window.messages[-keep_recent:] if keep_recent else []
    summarizer = summarizer or _mock_summarize
    note = summarizer(old)
    new = ContextWindow(budget=window.budget)
    new.add("system", f"[summary of {len(old)} earlier msgs] {note}")
    for role, text in recent:
        new.add(role, text)
    return new


def _mock_summarize(messages) -> str:
    # Deterministic: list the distinct roles + a token count, drop the prose.
    roles = []
    for r, _ in messages:
        if r not in roles:
            roles.append(r)
    toks = sum(est_tokens(t) for _, t in messages)
    return f"covered {len(messages)} turns ({'/'.join(roles)}), ~{toks} tokens condensed"


def _self_test() -> None:
    # Budget accounting + overflow detection.
    w = ContextWindow(budget=50)
    for i in range(20):
        w.add("user", "this is a fairly long message number %d padding padding" % i)
    assert w.over_budget(), w.used()
    before = w.used()

    # Compaction must shrink usage while keeping the most recent turns verbatim.
    w2 = compact(w, keep_recent=2)
    assert w2.used() < before, (before, w2.used())
    assert w2.messages[-1] == w.messages[-1], "recent turn must survive verbatim"
    assert w2.messages[0][0] == "system" and "summary" in w2.messages[0][1]

    # Tool-output trimming caps large blobs but keeps head+tail markers.
    big = "x" * 4000
    trimmed = trim_tool_output(big, max_tokens=50)
    assert est_tokens(trimmed) < est_tokens(big)
    assert "trimmed" in trimmed
    assert trim_tool_output("short", 50) == "short"
    print("[OK] context_engineering._self_test passed")


if __name__ == "__main__":
    _self_test()
    w = ContextWindow(budget=40)
    for i in range(8):
        w.add("user", f"turn {i} with some content to spend tokens here")
    print("before:", w.used(), "tokens,", len(w.messages), "msgs")
    w2 = compact(w)
    print("after :", w2.used(), "tokens,", len(w2.messages), "msgs")
    print(preview(w2.messages[0][1], 80))
