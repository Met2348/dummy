"""Letta (MemGPT) mock - main context + archival memory + tool calls."""
from __future__ import annotations
from dataclasses import dataclass, field
from common import hash_embed, cosine


@dataclass
class LettaMemory:
    system: str = "You are a Letta agent with managed memory."
    core: dict[str, str] = field(default_factory=lambda: {"human": "", "persona": "helpful agent"})
    recent: list[dict] = field(default_factory=list)
    archive: list[dict] = field(default_factory=list)
    max_recent: int = 6
    max_context_chars: int = 2000

    def add_message(self, role: str, content: str) -> None:
        self.recent.append({"role": role, "content": content})
        while len(self.recent) > self.max_recent:
            evicted = self.recent.pop(0)
            self.archival_insert(f"[evicted {evicted['role']}] {evicted['content']}")

    def archival_insert(self, text: str) -> str:
        record = {
            "id": f"a{len(self.archive)}",
            "text": text,
            "embedding": hash_embed(text),
        }
        self.archive.append(record)
        return record["id"]

    def archival_search(self, query: str, k: int = 3) -> list[dict]:
        q_vec = hash_embed(query)
        scored = [(rec, cosine(q_vec, rec["embedding"])) for rec in self.archive]
        return [r for r, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]

    def core_replace(self, label: str, new_value: str) -> None:
        self.core[label] = new_value

    def build_main_context(self) -> str:
        parts = [
            f"SYSTEM: {self.system}",
            "CORE:",
            f"  <human>: {self.core.get('human','')}",
            f"  <persona>: {self.core.get('persona','')}",
            "RECENT:",
        ]
        for m in self.recent:
            parts.append(f"  [{m['role']}] {m['content']}")
        return "\n".join(parts)[: self.max_context_chars]


def _self_test() -> None:
    mem = LettaMemory(max_recent=3)
    mem.core_replace("human", "User name is Alice")

    mem.add_message("user", "msg 1")
    mem.add_message("assistant", "reply 1")
    mem.add_message("user", "msg 2")
    mem.add_message("assistant", "reply 2")
    mem.add_message("user", "msg 3")

    assert len(mem.recent) == 3
    assert len(mem.archive) >= 2
    assert mem.recent[-1]["content"] == "msg 3"

    a_id = mem.archival_insert("Alice is a senior ML engineer.")
    assert a_id.startswith("a")
    found = mem.archival_search("Alice engineer", k=3)
    assert any("engineer" in f["text"] for f in found), found

    ctx = mem.build_main_context()
    assert "Alice" in ctx
    assert "SYSTEM" in ctx and "CORE" in ctx and "RECENT" in ctx
    print("[OK] letta_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
