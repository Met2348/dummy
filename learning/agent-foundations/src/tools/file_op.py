"""Mock file op - in-memory virtual fs."""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import Tool, ActionResult


_VIRTUAL_FS: dict[str, str] = {
    "notes.txt": "ReAct paper by Yao 2022.",
    "todo.md": "1. Learn agent\n2. Build RAG\n3. Ship product",
}


def _file_op(args: dict) -> ActionResult:
    action = args.get("action", "read")
    path = args.get("path") or args.get("input") or ""
    if not path:
        return ActionResult(ok=False, error="missing path")
    if action == "read":
        if path in _VIRTUAL_FS:
            return ActionResult(ok=True, value=_VIRTUAL_FS[path])
        return ActionResult(ok=False, error=f"file not found: {path}")
    if action == "write":
        content = args.get("content", "")
        _VIRTUAL_FS[path] = content
        return ActionResult(ok=True, value=f"wrote {len(content)} chars to {path}")
    if action == "list":
        return ActionResult(ok=True, value=sorted(_VIRTUAL_FS.keys()))
    return ActionResult(ok=False, error=f"unknown action: {action}")


file_op_tool = Tool(
    name="file_op",
    description="In-memory virtual fs. action=read/write/list, path=name.",
    schema={"action": "read|write|list", "path": "string", "content": "string (write only)"},
    func=_file_op,
)


def _self_test() -> None:
    r = file_op_tool.func({"action": "read", "path": "notes.txt"})
    assert r.ok and "ReAct" in r.value, r
    r = file_op_tool.func({"action": "list", "path": "_"})
    assert r.ok and "notes.txt" in r.value, r
    r = file_op_tool.func({"action": "write", "path": "x.txt", "content": "hi"})
    assert r.ok, r
    r = file_op_tool.func({"action": "read", "path": "x.txt"})
    assert r.ok and r.value == "hi", r
    print("[OK] file_op._self_test passed")


if __name__ == "__main__":
    _self_test()
