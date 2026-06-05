"""Common types for multi-agent-orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str | None  # None = broadcast
    content: str
    topic: str = ""


@dataclass
class AgentReply:
    name: str
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    transfer_to: str | None = None  # for hand-off


@dataclass
class CostTracker:
    tokens_in: int = 0
    tokens_out: int = 0
    n_calls: int = 0

    def add(self, tin: int, tout: int) -> None:
        self.tokens_in += tin
        self.tokens_out += tout
        self.n_calls += 1

    def usd(self, in_price: float = 0.003, out_price: float = 0.015) -> float:
        return round(
            self.tokens_in / 1000 * in_price + self.tokens_out / 1000 * out_price,
            6,
        )


def make_mock_agent(name: str, rules: list[tuple[str, str]]) -> Callable[[str], AgentReply]:
    """Pattern-match mock agent. First matching rule wins."""
    import re
    compiled = [(re.compile(p, re.IGNORECASE), reply) for p, reply in rules]

    def agent(prompt: str) -> AgentReply:
        tin = max(1, len(prompt) // 4)
        for pat, reply in compiled:
            if pat.search(prompt):
                tout = max(1, len(reply) // 4)
                return AgentReply(name=name, content=reply, tokens_in=tin, tokens_out=tout)
        tout = 4
        return AgentReply(name=name, content="(no reply)", tokens_in=tin, tokens_out=tout)

    return agent


def _self_test() -> None:
    agent = make_mock_agent("test", [("hello", "hi back"), (".*", "default")])
    r = agent("hello world")
    assert r.name == "test" and r.content == "hi back"
    r = agent("anything")
    assert r.content == "default"
    assert r.tokens_in > 0 and r.tokens_out > 0

    ct = CostTracker()
    ct.add(100, 50)
    ct.add(200, 100)
    assert ct.n_calls == 2
    assert ct.usd() > 0
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
