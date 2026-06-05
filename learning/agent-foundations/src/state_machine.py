"""Minimal LangGraph-style StateGraph."""
from __future__ import annotations
from typing import Callable, Any
from dataclasses import dataclass, field


END = "__END__"
START = "__START__"


@dataclass
class StateGraph:
    nodes: dict[str, Callable[[dict], dict]] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional: dict[str, Callable[[dict], str]] = field(default_factory=dict)
    entry: str = ""

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        if name in (START, END):
            raise ValueError(f"reserved: {name}")
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        if src in self.conditional:
            raise ValueError(f"{src} already has conditional edges")
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, fn: Callable[[dict], str]) -> None:
        if src in self.edges:
            raise ValueError(f"{src} already has unconditional edge")
        self.conditional[src] = fn

    def run(self, init_state: dict, max_steps: int = 20) -> tuple[dict, list[str]]:
        if not self.entry:
            raise ValueError("entry not set")
        state = dict(init_state)
        path = []
        node = self.entry
        for _ in range(max_steps):
            if node == END:
                break
            if node not in self.nodes:
                raise ValueError(f"unknown node: {node}")
            path.append(node)
            update = self.nodes[node](state)
            state.update(update or {})

            if node in self.conditional:
                node = self.conditional[node](state)
            elif node in self.edges:
                node = self.edges[node]
            else:
                node = END
        return state, path


def _self_test() -> None:
    g = StateGraph()
    g.add_node("plan", lambda s: {"plan": [1, 2, 3]})
    g.add_node("execute", lambda s: {"step": s.get("step", 0) + 1})
    g.add_node("done", lambda s: {"final": "OK"})

    g.set_entry("plan")
    g.add_edge("plan", "execute")
    g.add_conditional_edges(
        "execute",
        lambda s: "execute" if s.get("step", 0) < len(s.get("plan", [])) else "done",
    )
    g.add_edge("done", END)

    state, path = g.run({})
    assert state["step"] == 3, state
    assert state["final"] == "OK", state
    assert path == ["plan", "execute", "execute", "execute", "done"], path
    print("[OK] state_machine._self_test passed")


if __name__ == "__main__":
    _self_test()
