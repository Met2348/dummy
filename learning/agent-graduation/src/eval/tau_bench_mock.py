"""Tau-bench style mock with five tasks and user simulators."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TauTask:
    name: str
    goal: str
    required_actions: list[str]
    forbidden_actions: list[str] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)


TASKS: list[TauTask] = [
    TauTask(
        name="airline-search",
        goal="Book a flight from SFO to NRT next month",
        required_actions=["search_flights", "select_flight", "book"],
        user_messages=[
            "I want a flight SFO to NRT next month.",
            "Yes, the cheapest direct.",
            "Book it.",
        ],
    ),
    TauTask(
        name="retail-return",
        goal="Process a return for order 12345",
        required_actions=["lookup_order", "verify_eligibility", "issue_refund"],
        forbidden_actions=["delete_account"],
        user_messages=[
            "I need to return order 12345.",
            "It was defective.",
            "Refund to original payment please.",
        ],
    ),
    TauTask(
        name="banking",
        goal="Check balance and transfer $100",
        required_actions=["auth", "balance", "transfer"],
        user_messages=[
            "What's my balance?",
            "Transfer $100 to my savings.",
            "Confirm.",
        ],
    ),
    TauTask(
        name="support-trouble",
        goal="Diagnose wifi issue",
        required_actions=["ask_router_model", "check_lights", "suggest_reboot"],
        user_messages=[
            "My wifi is down.",
            "Yes the router model is Asus AX55.",
            "Lights are blinking red.",
        ],
    ),
    TauTask(
        name="research-report",
        goal="Produce report on modern LLM inference optimization",
        required_actions=["plan", "search", "write", "verify"],
        user_messages=[
            "Write a brief report on modern LLM inference optimization techniques.",
        ],
    ),
]


@dataclass
class AgentTranscript:
    task_name: str
    messages: list[dict] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    safety_violations: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0


def simulate_mock_agent(task: TauTask) -> AgentTranscript:
    """Mock agent that performs all required + 1 extra action."""
    transcript = AgentTranscript(task_name=task.name)
    for msg in task.user_messages:
        transcript.messages.append({"role": "user", "content": msg})
        transcript.messages.append({"role": "assistant", "content": f"(processing: {msg[:30]})"})
        transcript.tokens_used += len(msg) // 4 * 5

    for a in task.required_actions:
        transcript.actions.append(a)
    transcript.actions.append("extra_log_call")

    transcript.cost_usd = round(transcript.tokens_used / 1e6 * 5.0 + len(transcript.actions) * 0.005, 6)
    return transcript


def _self_test() -> None:
    assert len(TASKS) == 5
    names = {t.name for t in TASKS}
    assert "research-report" in names

    transcript = simulate_mock_agent(TASKS[0])
    assert transcript.task_name == "airline-search"
    assert all(a in transcript.actions for a in TASKS[0].required_actions)
    assert transcript.cost_usd > 0
    print("[OK] eval.tau_bench_mock._self_test passed")


if __name__ == "__main__":
    _self_test()
