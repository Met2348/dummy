"""ReAct loop — Yao 2022.

Thought-Action-Observation cycle until Final Answer or max_steps.
"""
from __future__ import annotations
from typing import Callable
from common import Tool, Trace, Step, parse_action, parse_final


REACT_SYSTEM = """You are a ReAct agent.
Available tools: {tool_names}

Use this format strictly:
Thought N: <reasoning>
Action N: tool_name(arg)
Observation N: <provided by env>

When you have the answer:
Final Answer: <answer>
"""


def build_initial_prompt(question: str, tools: dict[str, Tool]) -> str:
    names = ", ".join(tools.keys())
    return REACT_SYSTEM.format(tool_names=names) + f"\n\nQuestion: {question}\n"


def react_loop(
    question: str,
    llm: Callable[[str], str],
    tools: dict[str, Tool],
    max_steps: int = 8,
) -> Trace:
    trace = Trace(question=question)
    history = build_initial_prompt(question, tools)

    for step in range(1, max_steps + 1):
        out = llm(history + f"Thought {step}:")
        trace.tokens_in += len(history) // 4
        trace.tokens_out += len(out) // 4

        final = parse_final(out)
        if final is not None:
            trace.final = final
            trace.add(Step(step_num=step, thought=out.split("Action")[0].strip(),
                           action="FINAL", observation=final))
            return trace

        name, args = parse_action(out)
        if not name:
            trace.add(Step(step_num=step, thought=out[:200], action="PARSE_FAIL",
                           observation="(no action parsed)"))
            return trace

        if name not in tools:
            obs = f"ERROR: unknown tool {name!r}"
        else:
            result = tools[name].func(args)
            obs = result.to_obs() if hasattr(result, "to_obs") else str(result)

        trace.add(Step(step_num=step, thought=out.split("Action")[0].strip(),
                       action=f"{name}({args})", args=args, observation=obs))
        history += f"\n{out}\nObservation {step}: {obs}\n"

    return trace


def _self_test() -> None:
    from common import make_pattern_llm
    from tools import ALL_TOOLS

    llm = make_pattern_llm([
        (r"Thought 1:", "Thought: Need to compute.\nAction 1: calculator(2+3)\n"),
        (r"Thought 2:", "Thought: Got 5.\nFinal Answer: 5\n"),
    ])
    trace = react_loop("What is 2+3?", llm, ALL_TOOLS)
    assert trace.final == "5", trace
    assert len(trace.steps) == 2, trace
    print("[OK] react_loop._self_test passed")


if __name__ == "__main__":
    _self_test()
