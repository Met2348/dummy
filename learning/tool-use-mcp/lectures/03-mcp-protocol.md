# L03 · MCP 协议（Anthropic 2024.11）⭐⭐⭐⭐⭐

## 30 秒核心

> MCP (Model Context Protocol) = **client-server JSON-RPC 2.0 协议**，把"工具 / 资源 / 提示"标准化为可发现可调用的 endpoint。

类比：USB-C 之于充电 —— "一根线插所有"。

## 三大 primitive

| Primitive | 用途 |
|-----------|------|
| **tools** | 可执行函数 (action) |
| **resources** | 可读数据 (file / url / db row) |
| **prompts** | 预先写好的 prompt 模板 |

## 通讯层

```
Client (Claude Desktop / IDE / agent framework)
  ↕ JSON-RPC 2.0
  ↕ transport: stdio / SSE / streamable HTTP
Server (filesystem / git / postgres / ...)
```

### JSON-RPC 2.0 envelope

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {"name":"read_file","description":"...","inputSchema":{...}}
    ]
  }
}
```

## 标准方法

| Method | 用途 |
|--------|------|
| `initialize` | 握手 + capabilities |
| `tools/list` | 列工具 |
| `tools/call` | 调工具 |
| `resources/list` | 列资源 |
| `resources/read` | 读资源 |
| `prompts/list` | 列 prompt 模板 |
| `prompts/get` | 取 prompt 模板 |

## Initialize 握手

```
Client → Server: {"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{...}}}
Server → Client: {"result":{"protocolVersion":"...","capabilities":{"tools":{},"resources":{}}}}
```

→ 协商版本和 capability。

## Transport 三选一

| Transport | 何时用 |
|-----------|--------|
| **stdio** | 本地进程间 (Claude Desktop 启动 server 作子进程) |
| **SSE** | 旧 web (Server-Sent Events) |
| **Streamable HTTP** | 2025 新增，全双工 |

## MCP 生态（2025 上半年）

| Server 类 | 例 |
|-----------|---|
| 开发 | filesystem / git / github / sqlite / postgres |
| 协作 | slack / linear / gdrive / notion |
| Web | fetch / brave-search / puppeteer |
| 数据 | memory / time / sequential-thinking |
| 1000+ open-source servers |

## 安全模型

| 风险 | 解 |
|------|---|
| 工具被滥用 | client 端 allowlist + 用户确认 |
| Prompt injection | client 端做 (server 不感知) |
| 越权 | server 自己 ACL |

## MCP 与 function calling 比较

| 维度 | function calling | MCP |
|------|------------------|-----|
| 协议 | 每家不同 | 统一 |
| 工具发现 | 静态 (prompt 里) | 动态 (tools/list) |
| 跨进程 | 不 | 是 |
| Resource | 没有概念 | 一等公民 |
| Prompt 模板 | 没有概念 | 一等公民 |
| 复用 | 难 | 一个 server 多 client 用 |

## 实现 (`mcp_protocol.py` 预告)

我们手写一个内存版 MCP server + client：
- JSON-RPC 2.0 envelope
- `initialize` / `tools/list` / `tools/call`
- 1000 行代码内

## 退出条件

- 默写 3 primitive
- 默写 5 method 名
- 知道 stdio / SSE / Streamable HTTP 三 transport

## 一句话

> MCP = USB-C of LLM tools — JSON-RPC 2.0 + 3 primitive (tools/resources/prompts) + 3 transport，Anthropic 主推标准。
