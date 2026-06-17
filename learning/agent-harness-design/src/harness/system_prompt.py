"""System prompt construction — the harness's standing instructions to the model.

The harness owns this string (12-Factor: "own your prompts"). It declares the
role, lists tools with their descriptions (how the model knows what to call),
and injects environment context.
"""
from __future__ import annotations


def build_system_prompt(role: str, tools, env: dict | None = None) -> str:
    lines = [role, "", "## Tools"]
    if tools:
        for t in tools:
            ro = " (read-only)" if getattr(t, "read_only", False) else ""
            lines.append(f"- {t.name}: {t.description}{ro}")
    else:
        lines.append("(no tools available)")

    if env:
        lines.append("")
        lines.append("## Environment")
        for k, v in env.items():
            lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Behavior")
    lines.append("- Call a tool when you need information or an action; otherwise answer directly.")
    lines.append("- When the task is complete, reply with a final answer and no tool call.")
    return "\n".join(lines)


def _self_test() -> None:
    from .tools import Tool

    tools = [
        Tool("read_config", "read the team config", lambda: None, read_only=True),
        Tool("write_report", "save a markdown report", lambda: None),
    ]
    sp = build_system_prompt(
        "You are a budgeting assistant.", tools, env={"cwd": "/work", "user": "infra-team"}
    )
    assert "budgeting assistant" in sp
    assert "read_config" in sp and "(read-only)" in sp
    assert "write_report" in sp
    assert "cwd: /work" in sp and "## Environment" in sp
    assert "final answer" in sp

    empty = build_system_prompt("Role.", [])
    assert "no tools" in empty
    print("[OK] harness.system_prompt._self_test passed")


if __name__ == "__main__":
    _self_test()
