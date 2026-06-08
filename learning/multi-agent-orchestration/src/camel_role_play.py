"""Minimal CAMEL-style role-playing loop and failure checks."""
from __future__ import annotations

from dataclasses import dataclass, field

from common import CostTracker


END_TOKEN = "<CAMEL_TASK_DONE>"


@dataclass(frozen=True)
class RoleSpec:
    assistant_role: str
    user_role: str
    idea: str
    specified_task: str


@dataclass(frozen=True)
class Turn:
    instruction: str
    user_input: str
    solution: str


@dataclass
class RolePlayResult:
    spec: RoleSpec
    turns: list[Turn] = field(default_factory=list)
    termination_reason: str = ""
    failures: list[str] = field(default_factory=list)
    cost: CostTracker = field(default_factory=CostTracker)


def specify_task(idea: str, assistant_role: str, user_role: str) -> RoleSpec:
    """Task specifier: turn a vague idea into a concrete joint task."""
    task = (
        f"{assistant_role} will help {user_role} complete this task: "
        f"turn the idea '{idea}' into a concrete plan with implementation steps, "
        "risks, and a final checklist."
    )
    return RoleSpec(
        assistant_role=assistant_role,
        user_role=user_role,
        idea=idea,
        specified_task=task,
    )


def inception_prompts(spec: RoleSpec) -> dict[str, str]:
    """Return compact versions of CAMEL task, assistant, and user prompts."""
    task_prompt = (
        f"Here is a task that {spec.assistant_role} will help "
        f"{spec.user_role} to complete: {spec.idea}. Make it more specific."
    )
    assistant_prompt = (
        f"Never forget you are a {spec.assistant_role} and I am a {spec.user_role}. "
        f"Here is the task: {spec.specified_task}. Never flip roles. "
        "Always answer with 'Solution:' and end with 'Next request.'"
    )
    user_prompt = (
        f"Never forget you are a {spec.user_role} and I am a {spec.assistant_role}. "
        f"Here is the task: {spec.specified_task}. Give one instruction at a time. "
        f"When the task is complete, reply only with {END_TOKEN}."
    )
    return {
        "task_specifier": task_prompt,
        "assistant": assistant_prompt,
        "user": user_prompt,
    }


def plan_instructions(spec: RoleSpec) -> list[tuple[str, str]]:
    return [
        ("Break the task into major subtasks.", "No extra input."),
        ("Provide a concrete implementation plan.", spec.specified_task),
        ("List risks, checks, and the final acceptance criteria.", "No extra input."),
    ]


def assistant_solution(instruction: str, user_input: str) -> str:
    """Educational mock of the AI assistant response format."""
    body = instruction.rstrip(".")
    if user_input and user_input != "No extra input.":
        body += f" using context: {user_input}"
    return f"Solution: {body}. Provide specific steps and examples. Next request."


def detect_failures(turns: list[Turn]) -> list[str]:
    failures: list[str] = []
    for turn in turns:
        solution_low = turn.solution.lower()
        instruction_low = turn.instruction.lower()

        if turn.solution.strip().startswith("Instruction:"):
            failures.append("role_flipping")

        if instruction_low and instruction_low in solution_low:
            if "solution:" not in solution_low:
                failures.append("assistant_repeats_instruction")

        if solution_low.startswith("i will") or " i will " in solution_low:
            if "solution:" not in solution_low:
                failures.append("flake_reply")

    tail = " ".join(t.solution.lower() for t in turns[-3:])
    loop_words = ["thank you", "you are welcome", "goodbye"]
    if sum(1 for word in loop_words if word in tail) >= 2:
        failures.append("infinite_loop")

    return sorted(set(failures))


def run_role_play(
    idea: str,
    assistant_role: str,
    user_role: str,
    max_messages: int = 40,
) -> RolePlayResult:
    spec = specify_task(idea, assistant_role, user_role)
    prompts = inception_prompts(spec)
    result = RolePlayResult(spec=spec)
    result.cost.add(
        tin=sum(len(p) for p in prompts.values()) // 4,
        tout=len(spec.specified_task) // 4,
    )

    for instruction, user_input in plan_instructions(spec):
        if len(result.turns) * 2 >= max_messages:
            result.termination_reason = "max_messages"
            break
        solution = assistant_solution(instruction, user_input)
        result.turns.append(Turn(instruction=instruction, user_input=user_input, solution=solution))
        result.cost.add(tin=(len(instruction) + len(user_input)) // 4, tout=len(solution) // 4)
    else:
        result.termination_reason = "end_of_task_token"

    result.failures = detect_failures(result.turns)
    return result


def _self_test() -> None:
    result = run_role_play(
        idea="build a paper-reading agent",
        assistant_role="Python engineer",
        user_role="LLM researcher",
    )
    assert result.termination_reason == "end_of_task_token"
    assert len(result.turns) == 3
    assert all(t.solution.startswith("Solution:") for t in result.turns)
    assert all(t.solution.endswith("Next request.") for t in result.turns)
    assert result.failures == []
    assert result.cost.n_calls == 4

    bad = [
        Turn("Please implement the plan.", "None", "Instruction: ask me what to do."),
        Turn("Continue.", "None", "I will do that later."),
        Turn("Finish.", "None", "Thank you. You are welcome. Goodbye."),
    ]
    failures = detect_failures(bad)
    assert "role_flipping" in failures
    assert "flake_reply" in failures
    assert "infinite_loop" in failures

    short = run_role_play("x", "assistant", "user", max_messages=2)
    assert short.termination_reason == "max_messages"

    prompts = inception_prompts(result.spec)
    assert "Never flip roles" in prompts["assistant"]
    assert END_TOKEN in prompts["user"]
    print("[OK] camel_role_play._self_test passed")


if __name__ == "__main__":
    _self_test()
