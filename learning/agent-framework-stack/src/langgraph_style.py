"""LangGraph-style — Topic 4 mock 加 interrupt + time travel."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


START = "__START__"
END = "__END__"


class InterruptException(Exception):
    def __init__(self, value: Any):
        self.value = value


def interrupt(value: Any) -> Any:
    raise InterruptException(value)


@dataclass
class StateGraphV2:
    nodes: dict[str, Callable[[dict], dict]] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)
    conditional: dict[str, Callable[[dict], str]] = field(default_factory=dict)
    entry: str = ""

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, fn: Callable[[dict], str]) -> None:
        self.conditional[src] = fn

    def compile(self) -> "CompiledV2":
        return CompiledV2(self)


@dataclass
class Checkpoint:
    state: dict
    next_node: str
    interrupt_value: Any = None
    pending_resume: bool = False


class CompiledV2:
    def __init__(self, graph: StateGraphV2):
        self.graph = graph
        self.checkpoints: dict[str, list[Checkpoint]] = {}

    def invoke(
        self,
        init_state: dict | None,
        thread_id: str = "default",
        resume_value: Any = None,
        max_steps: int = 30,
    ) -> dict:
        history = self.checkpoints.setdefault(thread_id, [])
        if init_state is not None and not history:
            state = dict(init_state)
            node = self.graph.entry
        else:
            last = history[-1]
            state = dict(last.state)
            node = last.next_node
            if last.pending_resume and resume_value is not None:
                state["_resume"] = resume_value
                last.pending_resume = False

        for _ in range(max_steps):
            if node == END:
                history.append(Checkpoint(state=state, next_node=END))
                return state
            fn = self.graph.nodes.get(node)
            if fn is None:
                raise ValueError(f"unknown node: {node}")
            try:
                update = fn(state) or {}
                state.update(update)
            except InterruptException as e:
                history.append(Checkpoint(
                    state=state, next_node=node,
                    interrupt_value=e.value, pending_resume=True,
                ))
                state["_interrupted"] = True
                state["_interrupt_value"] = e.value
                return state

            if node in self.graph.conditional:
                node = self.graph.conditional[node](state)
            elif node in self.graph.edges:
                node = self.graph.edges[node]
            else:
                node = END
            history.append(Checkpoint(state=dict(state), next_node=node))
        return state

    def get_state_history(self, thread_id: str) -> list[Checkpoint]:
        return list(self.checkpoints.get(thread_id, []))


def _self_test() -> None:
    g = StateGraphV2()
    g.set_entry("plan")

    def plan(s): return {"step": 1, "amount": s.get("amount", 0)}

    def approve(s):
        if s.get("amount", 0) > 1000 and "_resume" not in s:
            interrupt({"reason": "approve big amount?"})
        return {"approved": s.get("_resume") == "yes" or s["amount"] <= 1000}

    def execute(s): return {"executed": True, "step": s["step"] + 1}

    g.add_node("plan", plan)
    g.add_node("approve", approve)
    g.add_node("execute", execute)
    g.add_edge("plan", "approve")
    g.add_edge("approve", "execute")
    g.add_edge("execute", END)

    app = g.compile()
    res1 = app.invoke({"amount": 50}, thread_id="t_small")
    assert res1["executed"] is True
    assert res1["approved"] is True

    res2 = app.invoke({"amount": 5000}, thread_id="t_big")
    assert res2.get("_interrupted"), res2
    assert res2["_interrupt_value"]["reason"] == "approve big amount?"

    res3 = app.invoke(None, thread_id="t_big", resume_value="yes")
    assert res3.get("approved") is True
    assert res3.get("executed") is True

    hist = app.get_state_history("t_big")
    assert len(hist) >= 3
    print("[OK] langgraph_style._self_test passed")


if __name__ == "__main__":
    _self_test()
