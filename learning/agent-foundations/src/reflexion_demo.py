"""Reflexion - Shinn 2023.

Actor (ReAct) + Evaluator + Self-reflect -> verbal memory -> next attempt.
"""
from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from common import Tool, Trace, make_pattern_llm
from react_loop import react_loop


@dataclass
class ReflexionState:
    task: str
    memory: list[str] = field(default_factory=list)
    attempts: list[Trace] = field(default_factory=list)
    success: bool = False


def evaluator_correct_answer(trace: Trace, expected: str) -> float:
    """Score 1.0 if final == expected, else 0.0."""
    if trace.final is None:
        return 0.0
    return 1.0 if str(trace.final).strip() == str(expected).strip() else 0.0


def make_reflect_llm(canned_reflections: list[str]) -> Callable[[str], str]:
    """Mock reflection LLM - returns canned reflection per attempt."""
    idx = [0]

    def llm(prompt: str) -> str:
        i = idx[0]
        idx[0] += 1
        return canned_reflections[i] if i < len(canned_reflections) else canned_reflections[-1]

    return llm


def reflexion_loop(
    task: str,
    actor_llm_factory: Callable[[list[str]], Callable[[str], str]],
    tools: dict[str, Tool],
    evaluator: Callable[[Trace], float],
    reflect_llm: Callable[[str], str],
    expected_answer: str = "",
    max_trials: int = 3,
) -> ReflexionState:
    state = ReflexionState(task=task)
    for trial in range(max_trials):
        actor_llm = actor_llm_factory(state.memory)
        trace = react_loop(task, actor_llm, tools, max_steps=5)
        state.attempts.append(trace)
        score = evaluator(trace)
        if score >= 1.0:
            state.success = True
            return state
        prompt = f"Task: {task}\nTrace: {trace.to_md()}\nWhy failed? Brief lesson:"
        reflection = reflect_llm(prompt)
        state.memory.append(f"Trial {trial+1} lesson: {reflection}")
    return state


def _self_test() -> None:
    from tools import ALL_TOOLS

    def actor_factory(memory: list[str]):
        n = len(memory)
        if n == 0:
            return make_pattern_llm([(r"Thought", "Thought: guess\nAction 1: calculator(2+2)\n"),
                                     (r".*", "Final Answer: 4")])
        return make_pattern_llm([(r"Thought 1:", "Thought: use right op now\nAction 1: calculator(2*3)\n"),
                                 (r"Thought 2:", "Final Answer: 6")])

    reflect_llm = make_reflect_llm(["I used + instead of *. Should multiply."])

    state = reflexion_loop(
        task="2*3?",
        actor_llm_factory=actor_factory,
        tools=ALL_TOOLS,
        evaluator=lambda t: evaluator_correct_answer(t, "6"),
        reflect_llm=reflect_llm,
        expected_answer="6",
        max_trials=3,
    )
    assert state.success, state
    assert len(state.attempts) == 2, len(state.attempts)
    assert len(state.memory) == 1, state.memory
    print("[OK] reflexion_demo._self_test passed")


if __name__ == "__main__":
    _self_test()
