"""Long conversation 4-layer architecture."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from context_mgmt import _mock_summary, approx_tokens


@dataclass
class LongConversation:
    recent: list[dict] = field(default_factory=list)
    summary: str = ""
    core_facts: dict = field(default_factory=dict)
    max_recent: int = 5
    summary_threshold: int = 10
    total_summary_calls: int = 0

    def add_turn(self, role: str, content: str, summary_fn: Callable | None = None) -> None:
        self.recent.append({"role": role, "content": content})
        if len(self.recent) > self.summary_threshold:
            keep = self.recent[-self.max_recent:]
            to_sum = self.recent[:-self.max_recent]
            summary_fn = summary_fn or _mock_summary
            new_sum = summary_fn(to_sum)
            self.total_summary_calls += 1
            self.summary = self._merge_summary(self.summary, new_sum)
            self.recent = keep

    def _merge_summary(self, old: str, new: str) -> str:
        if not old:
            return new
        return f"{old}\n{new}"

    def update_core(self, key: str, value: str) -> None:
        self.core_facts[key] = value

    def build_prompt(self, query: str) -> str:
        core_str = "\n".join(f"  {k}: {v}" for k, v in self.core_facts.items())
        recent_str = "\n".join(f"  [{m['role']}] {m['content']}" for m in self.recent)
        return (
            "SYSTEM: long-conversation agent\n"
            f"CORE FACTS:\n{core_str or '  (none)'}\n"
            f"CONTEXT SUMMARY:\n  {self.summary or '(none)'}\n"
            f"RECENT:\n{recent_str or '  (none)'}\n"
            f"USER: {query}"
        )

    def total_tokens(self) -> int:
        return (approx_tokens(self.summary)
                + sum(approx_tokens(m["content"]) for m in self.recent)
                + sum(approx_tokens(v) for v in self.core_facts.values()))


def _self_test() -> None:
    conv = LongConversation(max_recent=3, summary_threshold=6)
    conv.update_core("user_name", "Alice")

    for i in range(15):
        conv.add_turn("user" if i % 2 == 0 else "assistant", f"message {i}")

    assert len(conv.recent) <= 5, len(conv.recent)
    assert conv.summary != ""
    assert conv.total_summary_calls > 0

    prompt = conv.build_prompt("What did we discuss?")
    assert "Alice" in prompt
    assert "summary" in prompt.lower()
    assert "message 14" in prompt or "message 13" in prompt
    print("[OK] long_conv._self_test passed")


if __name__ == "__main__":
    _self_test()
