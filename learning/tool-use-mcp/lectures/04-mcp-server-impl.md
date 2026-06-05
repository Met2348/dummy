# L04 · 实现 MCP Server

## 5 步实现

```
1. Register tools (name + schema + func)
2. Listen on transport (stdio JSON 行)
3. Parse JSON-RPC request
4. Dispatch by method
5. Send JSON-RPC response
```

## 真实 MCP SDK (Python)

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [Tool(name="echo", description="echo back", inputSchema={...})]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "echo":
        return [TextContent(type="text", text=arguments["msg"])]

# 启动:
# server.run(stdin_stream=..., stdout_stream=...)
```

## 我们手写版（in-memory，`mcp_server.py` 预告）

```python
class MCPServer:
    def __init__(self, name):
        self.name = name
        self.tools = {}

    def add_tool(self, name, description, schema, func):
        self.tools[name] = {"description":description, "schema":schema, "func":func}

    def handle(self, request: dict) -> dict:
        method = request["method"]
        if method == "initialize":
            return self._initialize(request)
        if method == "tools/list":
            return self._list_tools(request)
        if method == "tools/call":
            return self._call_tool(request)
        return self._error(request, code=-32601, message="Method not found")
```

## Capabilities 协商

```python
def _initialize(self, request):
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": "0.1"},
            "capabilities": {"tools": {}},
        },
    }
```

## tools/list 响应

```python
def _list_tools(self, request):
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {
            "tools": [
                {"name": n, "description": t["description"], "inputSchema": t["schema"]}
                for n, t in self.tools.items()
            ]
        },
    }
```

## tools/call 响应

```python
def _call_tool(self, request):
    params = request.get("params", {})
    name = params["name"]
    args = params.get("arguments", {})
    if name not in self.tools:
        return self._error(request, -32602, f"Unknown tool: {name}")
    try:
        result = self.tools[name]["func"](args)
    except Exception as e:
        return self._error(request, -32603, str(e))
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        },
    }
```

## 错误码（JSON-RPC 标准）

| Code | 含义 |
|------|------|
| -32700 | Parse error |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |

## 真 stdio 模式（生产）

```python
import sys, json
while line := sys.stdin.readline():
    req = json.loads(line)
    resp = server.handle(req)
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
```

## 退出条件

- 知道 4 method (initialize / tools/list / tools/call / errors)
- 能写一个 handle 函数
- 知道 JSON-RPC 5 错误码

## 一句话

> MCP server = JSON-RPC 2.0 dispatch + tool 注册表 + capabilities 协商 — 50 行能跑。
