"""Toy MemGPT virtual context manager from the paper guide."""
from __future__ import annotations
from dataclasses import dataclass, field
from context_mgmt import approx_tokens, _mock_summary


@dataclass
class QueueEvent:
    role: str
    content: str


@dataclass
class FlushReport:
    warned: bool = False
    flushed: bool = False
    evicted_count: int = 0
    prompt_tokens_after: int = 0


@dataclass
class VirtualContext:
    capacity_tokens: int = 120
    warning_ratio: float = 0.70
    system_tokens: int = 20
    working_context: str = ""
    fifo: list[QueueEvent] = field(default_factory=list)
    recall_storage: list[QueueEvent] = field(default_factory=list)
    recursive_summary: str = ""

    def prompt_tokens(self) -> int:
        return (
            self.system_tokens
            + approx_tokens(self.working_context)
            + approx_tokens(self.recursive_summary)
            + sum(approx_tokens(e.content) for e in self.fifo)
        )

    def add_event(self, role: str, content: str) -> FlushReport:
        event = QueueEvent(role=role, content=content)
        self.fifo.append(event)
        self.recall_storage.append(event)
        return self._check_pressure()

    def core_replace(self, text: str) -> None:
        self.working_context = text

    def recall_search(self, query: str, k: int = 3) -> list[QueueEvent]:
        terms = set(query.lower().split())
        scored: list[tuple[QueueEvent, int]] = []
        for event in self.recall_storage:
            score = sum(1 for term in terms if term in event.content.lower())
            if score:
                scored.append((event, score))
        return [event for event, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:k]]

    def _check_pressure(self) -> FlushReport:
        before = self.prompt_tokens()
        warned = before >= int(self.capacity_tokens * self.warning_ratio)
        if before < self.capacity_tokens:
            return FlushReport(warned=warned, prompt_tokens_after=before)

        evict_count = max(1, len(self.fifo) // 2)
        evicted = self.fifo[:evict_count]
        self.fifo = self.fifo[evict_count:]
        new_summary = _mock_summary([{"content": e.content} for e in evicted])
        if self.recursive_summary:
            self.recursive_summary = self.recursive_summary + "\n" + new_summary
        else:
            self.recursive_summary = new_summary
        return FlushReport(
            warned=warned,
            flushed=True,
            evicted_count=evict_count,
            prompt_tokens_after=self.prompt_tokens(),
        )


def nested_kv_lookup(store: dict[str, str], start_key: str, max_hops: int = 8) -> tuple[list[str], str]:
    """Simulate function chaining for MemGPT's nested KV task."""
    path = [start_key]
    current = start_key
    for _ in range(max_hops):
        if current not in store:
            return path, current
        current = store[current]
        path.append(current)
    raise RuntimeError("max_hops exceeded")


def paginated_search(records: list[str], query: str, page: int = 0, page_size: int = 3) -> list[str]:
    matches = [record for record in records if query.lower() in record.lower()]
    start = page * page_size
    return matches[start:start + page_size]


def _self_test() -> None:
    vc = VirtualContext(capacity_tokens=45, system_tokens=5)
    vc.core_replace("human: Alice prefers Anthropic Claude")

    report = FlushReport()
    for i in range(12):
        report = vc.add_event("user", f"turn {i}: Alice discussed memory systems and RAG details")

    assert report.warned is True
    assert report.flushed is True
    assert len(vc.recall_storage) == 12
    assert vc.recursive_summary != ""

    found = vc.recall_search("RAG details", k=2)
    assert found and "RAG" in found[0].content

    path, final = nested_kv_lookup({"k1": "k2", "k2": "k3", "k3": "answer"}, "k1")
    assert path == ["k1", "k2", "k3", "answer"]
    assert final == "answer"

    records = ["nobel physics page 1", "nobel physics page 2", "other", "nobel physics page 3"]
    assert paginated_search(records, "nobel", page=1, page_size=2) == ["nobel physics page 3"]
    print("[OK] memgpt_virtual_context._self_test passed")


if __name__ == "__main__":
    _self_test()
