"""Minimal in-memory MCP server."""
from __future__ import annotations
from typing import Callable
from mcp_protocol import (
    make_response, make_error, ERROR_CODES, PROTOCOL_VERSION,
)


class MCPServer:
    def __init__(self, name: str, version: str = "0.1"):
        self.name = name
        self.version = version
        self.tools: dict[str, dict] = {}

    def add_tool(self, name: str, description: str, input_schema: dict,
                 func: Callable[[dict], object]) -> None:
        self.tools[name] = {
            "description": description,
            "input_schema": input_schema,
            "func": func,
        }

    def handle(self, request: dict) -> dict:
        req_id = request.get("id", 0)
        method = request.get("method", "")
        if method == "initialize":
            return self._initialize(req_id, request.get("params", {}))
        if method == "tools/list":
            return self._tools_list(req_id)
        if method == "tools/call":
            return self._tools_call(req_id, request.get("params", {}))
        return make_error(req_id, ERROR_CODES["method_not_found"],
                          f"Method not found: {method}")

    def _initialize(self, req_id: int, params: dict) -> dict:
        return make_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {"tools": {}},
        })

    def _tools_list(self, req_id: int) -> dict:
        return make_response(req_id, {
            "tools": [
                {"name": n, "description": t["description"], "inputSchema": t["input_schema"]}
                for n, t in self.tools.items()
            ]
        })

    def _tools_call(self, req_id: int, params: dict) -> dict:
        name = params.get("name")
        if not name or name not in self.tools:
            return make_error(req_id, ERROR_CODES["invalid_params"],
                              f"Unknown tool: {name}")
        args = params.get("arguments", {})
        try:
            result = self.tools[name]["func"](args)
        except Exception as e:  # noqa: BLE001
            return make_error(req_id, ERROR_CODES["internal_error"], str(e))
        return make_response(req_id, {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        })


def _self_test() -> None:
    srv = MCPServer("test")
    srv.add_tool("echo", "echo back",
                 {"type": "object", "properties": {"msg": {"type": "string"}}},
                 lambda args: args.get("msg", ""))

    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert "result" in init
    assert init["result"]["capabilities"]["tools"] == {}

    listed = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert "result" in listed
    assert listed["result"]["tools"][0]["name"] == "echo"

    called = srv.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                         "params": {"name": "echo", "arguments": {"msg": "hi"}}})
    assert called["result"]["content"][0]["text"] == "hi"

    err = srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                      "params": {"name": "no_such_tool"}})
    assert "error" in err and err["error"]["code"] == -32602

    unk = srv.handle({"jsonrpc": "2.0", "id": 5, "method": "weird/method"})
    assert "error" in unk and unk["error"]["code"] == -32601
    print("[OK] mcp_server._self_test passed")


if __name__ == "__main__":
    _self_test()
