"""Claude Agent SDK-style - built-in tools + permission modes + hooks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from common import mock_search


PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "plan"}


BUILTIN_TOOLS = {
    "Read": lambda args: f"<file {args.get('path','?')} contents>",
    "Write": lambda args: f"wrote {len(args.get('content',''))} chars to {args.get('path','?')}",
    "Bash": lambda args: f"$ {args.get('command','?')} (stdout)",
    "WebSearch": lambda args: mock_search(args.get("query", ""), k=3),
    "WebFetch": lambda args: f"<page {args.get('url','?')}>",
    "Task": lambda args: f"[subagent for: {args.get('description','?')}]",
}


DANGEROUS_TOOLS = {"Bash", "Write"}


@dataclass
class HookContext:
    tool: str
    args: dict


@dataclass
class HookResult:
    allowed: bool = True
    message: str = ""


@dataclass
class AgentRunResult:
    messages: list[dict] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    permission_mode: str = "default"


def query(
    prompt: str,
    options: dict | None = None,
    mock_llm: Callable[[str, list], list[dict]] | None = None,
) -> AgentRunResult:
    options = options or {}
    mode = options.get("permissionMode", "default")
    if mode not in PERMISSION_MODES:
        raise ValueError(f"unknown permission mode: {mode}")

    allowed_tools = set(options.get("allowedTools", list(BUILTIN_TOOLS.keys())))
    custom_tools = options.get("customTools", {})
    pre_hook = options.get("preToolUse")

    tools = {**{n: f for n, f in BUILTIN_TOOLS.items() if n in allowed_tools}, **custom_tools}

    result = AgentRunResult(permission_mode=mode)
    result.messages.append({"role": "user", "content": prompt})

    if mock_llm is None:
        result.messages.append({"role": "assistant", "content": f"(default reply to: {prompt[:50]})"})
        return result

    for step in mock_llm(prompt, result.messages):
        tool_name = step.get("tool")
        if tool_name is None:
            result.messages.append({"role": "assistant", "content": step.get("text", "")})
            break
        if tool_name not in tools:
            result.messages.append({"role": "tool", "name": tool_name, "content": "ERROR: tool not allowed"})
            result.blocked_tools.append(tool_name)
            continue
        if pre_hook is not None:
            ctx = HookContext(tool=tool_name, args=step.get("args", {}))
            hook_result = pre_hook(ctx)
            if not hook_result.allowed:
                result.messages.append({"role": "tool", "name": tool_name,
                                         "content": f"BLOCKED by hook: {hook_result.message}"})
                result.blocked_tools.append(tool_name)
                continue
        if mode == "plan":
            result.messages.append({"role": "tool", "name": tool_name,
                                     "content": "(plan mode - not executed)"})
            continue
        try:
            tool_out = tools[tool_name](step.get("args", {}))
        except Exception as e:
            tool_out = f"ERROR: {e}"
        result.messages.append({"role": "tool", "name": tool_name, "content": str(tool_out)})

    return result


def _self_test() -> None:
    def llm1(prompt, history):
        return [
            {"tool": "WebSearch", "args": {"query": "ReAct"}},
            {"text": "Found info"},
        ]
    res = query("Search ReAct", options={"allowedTools": ["WebSearch"]}, mock_llm=llm1)
    assert any(m["role"] == "tool" and m["name"] == "WebSearch" for m in res.messages)

    def llm2(prompt, history):
        return [{"tool": "Bash", "args": {"command": "rm -rf /"}}]

    def safety_hook(ctx: HookContext) -> HookResult:
        if ctx.tool == "Bash" and "rm -rf" in ctx.args.get("command", ""):
            return HookResult(allowed=False, message="dangerous rm")
        return HookResult(allowed=True)

    res2 = query("Delete all", options={"preToolUse": safety_hook, "allowedTools": ["Bash"]}, mock_llm=llm2)
    assert "Bash" in res2.blocked_tools
    assert any("BLOCKED" in m["content"] for m in res2.messages if m["role"] == "tool")

    def llm3(prompt, history):
        return [{"tool": "Write", "args": {"path": "x.txt", "content": "hi"}}]
    res3 = query("Write", options={"permissionMode": "plan", "allowedTools": ["Write"]}, mock_llm=llm3)
    assert any("plan mode" in m["content"] for m in res3.messages if m["role"] == "tool")

    try:
        query("X", options={"permissionMode": "invalid"})
        assert False, "should have raised"
    except ValueError:
        pass
    print("[OK] claude_agent_sdk_style._self_test passed")


if __name__ == "__main__":
    _self_test()
