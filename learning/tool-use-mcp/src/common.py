"""Common types for tool-use-mcp."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    ok: bool
    content: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        if self.ok:
            return {"tool_call_id": self.call_id, "content": str(self.content)}
        return {"tool_call_id": self.call_id, "content": f"ERROR: {self.error}", "is_error": True}


def _self_test() -> None:
    s = ToolSchema(name="x", description="d", input_schema={"type": "object"})
    assert s.name == "x"

    c = ToolCall(id="c1", name="x", arguments={"a": 1})
    r = ToolResult(call_id="c1", ok=True, content=42)
    assert r.to_dict()["content"] == "42"

    bad = ToolResult(call_id="c1", ok=False, error="bad")
    assert bad.to_dict()["is_error"] is True
    print("[OK] common._self_test passed")


if __name__ == "__main__":
    _self_test()
