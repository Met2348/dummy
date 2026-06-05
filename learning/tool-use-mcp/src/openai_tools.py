"""OpenAI function calling JSON parsing."""
from __future__ import annotations
import json
from common import ToolCall, ToolResult, ToolSchema


def to_openai_tool(schema: ToolSchema) -> dict:
    """Convert internal schema to OpenAI tools format."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.input_schema,
        },
    }


def parse_openai_tool_call(raw: dict) -> ToolCall:
    """Parse {'id':'call_x','function':{'name':'x','arguments':'{...}'}}."""
    if "function" not in raw:
        raise ValueError("missing function key")
    fn = raw["function"]
    args_str = fn.get("arguments", "{}")
    if isinstance(args_str, str):
        args = json.loads(args_str) if args_str else {}
    else:
        args = args_str
    return ToolCall(id=raw.get("id", "call_0"), name=fn["name"], arguments=args)


def execute_tool_call(call: ToolCall, registry: dict[str, callable]) -> ToolResult:
    if call.name not in registry:
        return ToolResult(call_id=call.id, ok=False, error=f"unknown tool {call.name}")
    try:
        out = registry[call.name](call.arguments)
        return ToolResult(call_id=call.id, ok=True, content=out)
    except Exception as e:  # noqa: BLE001
        return ToolResult(call_id=call.id, ok=False, error=str(e))


def _self_test() -> None:
    s = ToolSchema(name="add", description="add", input_schema={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}}
    })
    ot = to_openai_tool(s)
    assert ot["type"] == "function"
    assert ot["function"]["name"] == "add"

    raw = {"id": "call_1", "function": {"name": "add", "arguments": '{"a":3,"b":4}'}}
    tc = parse_openai_tool_call(raw)
    assert tc.name == "add" and tc.arguments == {"a": 3, "b": 4}

    registry = {"add": lambda args: args["a"] + args["b"]}
    r = execute_tool_call(tc, registry)
    assert r.ok and r.content == 7

    bad = ToolCall(id="c2", name="nope", arguments={})
    r2 = execute_tool_call(bad, registry)
    assert not r2.ok and "unknown" in r2.error
    print("[OK] openai_tools._self_test passed")


if __name__ == "__main__":
    _self_test()
