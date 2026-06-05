# L05 · 实现 MCP Client

## Client 5 步

```
1. Spawn server (stdio: subprocess.Popen)
2. Send initialize → 取 capabilities
3. Send tools/list → 缓存可用工具
4. (LLM 选 tool 时) Send tools/call → 收 result
5. 关 server
```

## 我们手写版（in-process，`mcp_client.py` 预告）

```python
class MCPClient:
    def __init__(self, server):
        self.server = server   # in-process direct call
        self.next_id = 1
        self.capabilities = {}
        self.tools_cache = []

    def _send(self, method, params=None):
        req = {"jsonrpc":"2.0","id":self.next_id,"method":method,"params":params or {}}
        self.next_id += 1
        return self.server.handle(req)

    def initialize(self):
        resp = self._send("initialize", {"protocolVersion":"2024-11-05"})
        self.capabilities = resp["result"]["capabilities"]
        return self.capabilities

    def list_tools(self):
        resp = self._send("tools/list")
        self.tools_cache = resp["result"]["tools"]
        return self.tools_cache

    def call_tool(self, name, arguments):
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]["content"]
```

## 真生产 stdio 模式

```python
import subprocess, json
proc = subprocess.Popen(
    ["python", "my_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

def call(method, params=None):
    req = {"jsonrpc":"2.0","id":1,"method":method,"params":params or {}}
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())
```

## Client 把 MCP tools 转给 LLM

```python
def to_openai_format(mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in mcp_tools
    ]
```

→ MCP 标准 schema 直接喂给 LLM 用。

## 多 server 拼合

实际 agent 接 N 个 MCP server：
```python
clients = [MCPClient(spawn("fs-server")), MCPClient(spawn("git-server")), ...]
all_tools = []
for c in clients:
    c.initialize()
    all_tools.extend(c.list_tools())
```

→ Routing：tool name → server （多 server 时记 mapping）。

## Claude Desktop config 例

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/Users/me"]
    },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-github"]
    }
  }
}
```

启动 Claude Desktop 自动 spawn 这些 server。

## 退出条件

- 能写 in-process MCP client
- 知道 to_openai_format 转换
- 知道多 server 路由

## 一句话

> MCP client = JSON-RPC 客户端 + tool cache + transport (stdio/SSE) — 90 行能跑通完整 round-trip。
