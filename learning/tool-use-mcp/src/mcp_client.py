"""Minimal in-process MCP client."""
from __future__ import annotations
from mcp_protocol import make_request
from mcp_server import MCPServer


class MCPClient:
    def __init__(self, server: MCPServer):
        self.server = server
        self.next_id = 1
        self.capabilities: dict = {}
        self.tools_cache: list[dict] = []

    def _send(self, method: str, params: dict | None = None) -> dict:
        req = make_request(self.next_id, method, params)
        self.next_id += 1
        return self.server.handle(req)

    def initialize(self) -> dict:
        resp = self._send("initialize", {"protocolVersion": "2024-11-05"})
        if "error" in resp:
            raise RuntimeError(f"init failed: {resp['error']}")
        self.capabilities = resp["result"]["capabilities"]
        return resp["result"]

    def list_tools(self) -> list[dict]:
        resp = self._send("tools/list")
        if "error" in resp:
            raise RuntimeError(f"list_tools failed: {resp['error']}")
        self.tools_cache = resp["result"]["tools"]
        return self.tools_cache

    def call_tool(self, name: str, arguments: dict) -> tuple[bool, str]:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            return False, f"{resp['error']['code']} {resp['error']['message']}"
        result = resp["result"]
        text = result["content"][0]["text"]
        return not result.get("isError", False), text


def _self_test() -> None:
    srv = MCPServer("test")
    srv.add_tool("double", "double the number",
                 {"type": "object", "properties": {"n": {"type": "number"}}},
                 lambda args: args["n"] * 2)

    client = MCPClient(srv)
    info = client.initialize()
    assert "serverInfo" in info

    tools = client.list_tools()
    assert len(tools) == 1 and tools[0]["name"] == "double"

    ok, text = client.call_tool("double", {"n": 21})
    assert ok and text == "42"

    ok, err = client.call_tool("missing", {})
    assert not ok and "-32602" in err
    print("[OK] mcp_client._self_test passed")


if __name__ == "__main__":
    _self_test()
