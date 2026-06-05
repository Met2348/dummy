"""LangGraph-style StateGraph with reducers + conditional edges + checkpoint."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


START = "__START__"
END = "__END__"


@dataclass
class StateGraph:
    state_keys: dict[str, Callable[[Any, Any], Any]] = field(default_factory=dict)
    nodes: dict[str, Callable[[dict], dict]] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional: dict[str, tuple] = field(default_factory=dict)

    def add_reducer(self, key: str, fn: Callable[[Any, Any], Any]) -> None:
        self.state_keys[key] = fn

    def add_node(self, name: str, fn: Callable[[dict], dict]) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        if src in self.conditional:
            raise ValueError(f"{src} already conditional")
        self.edges[src] = dst

    def add_conditional_edges(
        self,
        src: str,
        decider: Callable[[dict], str],
        mapping: dict[str, str] | None = None,
    ) -> None:
        if src in self.edges:
            raise ValueError(f"{src} already has edge")
        self.conditional[src] = (decider, mapping or {})

    def _reduce(self, state: dict, update: dict) -> dict:
        new = dict(state)
        for k, v in update.items():
            if k in self.state_keys:
                new[k] = self.state_keys[k](new.get(k), v)
            else:
                new[k] = v
        return new

    def compile(self, checkpointer: dict | None = None) -> "CompiledGraph":
        return CompiledGraph(self, checkpointer=checkpointer)


class CompiledGraph:
    def __init__(self, graph: StateGraph, checkpointer: dict | None = None):
        self.graph = graph
        self.checkpointer = checkpointer if checkpointer is not None else {}

    def invoke(self, init_state: dict, config: dict | None = None, max_steps: int = 30) -> dict:
        config = config or {}
        thread_id = config.get("thread_id", "default")

        if thread_id in self.checkpointer:
            state = self.checkpointer[thread_id]["state"]
            current = self.checkpointer[thread_id]["next_node"]
        else:
            state = dict(init_state)
            current = START

        path = []
        for _ in range(max_steps):
            if current == END:
                break
            if current == START:
                current = self.graph.edges.get(START, END)
                continue
            if current not in self.graph.nodes:
                raise ValueError(f"unknown node: {current}")
            path.append(current)
            update = self.graph.nodes[current](state) or {}
            state = self.graph._reduce(state, update)

            if current in self.graph.conditional:
                decider, mapping = self.graph.conditional[current]
                key = decider(state)
                current = mapping.get(key, key)
            elif current in self.graph.edges:
                current = self.graph.edges[current]
            else:
                current = END

        self.checkpointer[thread_id] = {"state": state, "next_node": current, "path": path}
        return state


def append_reducer(old, new):
    if old is None:
        return list(new)
    return old + list(new)


def replace_reducer(old, new):
    return new


def _self_test() -> None:
    g = StateGraph()
    g.add_reducer("messages", append_reducer)
    g.add_reducer("step", lambda a, b: (a or 0) + b)

    g.add_node("plan", lambda s: {"messages": ["planned"], "step": 1})
    g.add_node("exec", lambda s: {"messages": ["executed"], "step": 1})
    g.add_node("review", lambda s: {"messages": ["reviewed"], "step": 1})

    g.add_edge(START, "plan")
    g.add_edge("plan", "exec")
    g.add_conditional_edges(
        "exec",
        lambda s: "review" if s["step"] >= 3 else "exec",
    )
    g.add_edge("review", END)

    app = g.compile()
    final = app.invoke({"messages": [], "step": 0}, config={"thread_id": "t1"})
    assert final["step"] == 4, final
    assert "planned" in final["messages"]
    assert "executed" in final["messages"]
    assert "reviewed" in final["messages"]

    cont = app.invoke({"messages": [], "step": 0}, config={"thread_id": "t1"})
    assert cont["step"] == 4
    print("[OK] langgraph_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
