"""ReAct + ToT + Self-Consistency mock agents using `frontend_lang.Stream`."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional
from frontend_lang import Stream, Gen, Select, function


def mock_tool(action: str) -> str:
    if "search" in action.lower():
        return "result: 42"
    if "calc" in action.lower():
        return "result: 3.14"
    return "tool: ok"


@function
def react_loop(s: Stream, question: str, max_steps: int = 4) -> None:
    s += f"Question: {question}\n"
    for step in range(max_steps):
        s += f"Thought {step}: "
        s += Gen(f"t{step}", max_tokens=20)
        s += f"\nAction {step}: "
        s += Gen(f"a{step}", max_tokens=10)
        obs = mock_tool(s.vars[f"a{step}"])
        s += f"\nObservation: {obs}\n"
        if step >= 1 and "Final" in s.vars[f"t{step}"]:
            break


def tree_of_thought(question: str, k: int = 3) -> str:
    """Run k thought-branches in parallel; pick the longest as the answer."""
    s = Stream()
    s += f"Q: {question}\nThought: "
    forks = s.fork(k)
    for f in forks:
        f += Gen("thought", max_tokens=30)
    s += Select("choice", choices=[f.vars["thought"] for f in forks])
    s += "\nAnswer: "
    s += Gen("answer")
    return s.prompt


def self_consistency(question: str, k: int = 5) -> str:
    """Sample k completions; return majority answer."""
    s = Stream()
    s += f"Q: {question}\nA: "
    samples = s.fork(k)
    for sample in samples:
        sample += Gen("ans", max_tokens=10)
    votes = Counter(sample.vars["ans"] for sample in samples)
    return votes.most_common(1)[0][0]


if __name__ == "__main__":
    res = react_loop("What's 2+3?", max_steps=3)
    print(res.prompt)
    print("--- ToT ---")
    print(tree_of_thought("Capital of France?", k=3))
