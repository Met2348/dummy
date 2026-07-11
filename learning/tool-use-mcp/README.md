# Topic 3: Tool Use & MCP（工具协议化）

> Module 7 第 3 专题 · 12 lectures · ~12h
>
> OpenAI function calling → **Anthropic MCP** (2024.11) → **Google A2A** (2025) → Computer Use → Sandbox 安全执行

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | Tool calling 范式 | `toolformer_toy.py`（前置：Toolformer 论文筛选机制，见 `paper/guide_01_toolformer.md`） |
| L02 | OpenAI function calling | `openai_tools.py` |
| L03 | **MCP** 协议 ⭐ | `mcp_protocol.py` |
| L04 | MCP server impl | `mcp_server.py` |
| L05 | MCP client impl | `mcp_client.py` |
| L06 | **A2A** 协议 ⭐ | `a2a_minimal.py` |
| L07 | Computer Use | `computer_use_mock.py` |
| L08 | Sandbox / e2b / pyodide | `sandbox_mock.py` |
| L09 | Streaming + interrupt | `streaming_tools.py` |
| L10 | Tool error 重试 | `tool_retry.py` |
| L11 | Tool 安全 (injection) | `tool_injection_demo.py` |
| L12 | **Capstone**: 手写 MCP server + 3 工具 | `capstone_mcp_stack.py` |

## Tags

- `tool-use-mcp` — Module 7 第 3 专题

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（13/13，纯 CPU 秒级）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules tool-use-mcp
> ```

13 个脚本全是**纯 stdlib 手写 mock**（零外部依赖、CPU 秒级；MCP/A2A 都是 in-process 直接函数调用，
无真实 stdio/网络/子进程 transport）。每个直跑都会跑内置 `_self_test`（真断言，非硬编码打印）。
脚本无 argparse，直接 `python <脚本>` 即可（harness 会自动把 `src/` 加进 `PYTHONPATH`，故裸 import 可解析）：

```powershell
# L01 前置：Toolformer 筛选玩具复现（见 paper/guide_01_toolformer.md）
python learning/tool-use-mcp/src/toolformer_toy.py
# L02 OpenAI function calling schema 转换/解析/执行
python learning/tool-use-mcp/src/openai_tools.py
# L03 MCP JSON-RPC 2.0 envelope
python learning/tool-use-mcp/src/mcp_protocol.py
# L04 MCP server（initialize/tools-list/tools-call）
python learning/tool-use-mcp/src/mcp_server.py
# L05 MCP client（initialize+list_tools+call_tool round-trip）
python learning/tool-use-mcp/src/mcp_client.py
# L06 A2A 协议（Agent Card + skill + task 生命周期）
python learning/tool-use-mcp/src/a2a_minimal.py
# L07 Computer Use mock（screenshot/click/type/key）
python learning/tool-use-mcp/src/computer_use_mock.py
# L08 Sandbox（AST 白名单 + 受限 exec）
python learning/tool-use-mcp/src/sandbox_mock.py
# L09 流式工具 + cancel hook
python learning/tool-use-mcp/src/streaming_tools.py
# L10 指数退避重试 + 熔断器
python learning/tool-use-mcp/src/tool_retry.py
# L11 Prompt/Tool-output injection 检测 + 清洗
python learning/tool-use-mcp/src/tool_injection_demo.py
```

> 共享类型库（`ToolSchema`/`ToolCall`/`ToolResult`）也可独立 self-test：
> `python learning/tool-use-mcp/src/common.py`

**Capstone（L12）：手写 MCP server(3 工具) + client discover + 4 次调用**

```powershell
$env:PYTHONIOENCODING="utf-8"
python learning/tool-use-mcp/src/capstone_mcp_stack.py
```

> 直跑即打印 handshake + 3 个发现的工具 + 4 次调用结果（3 次成功 + 1 次预期错误 `-32602 Unknown tool`）的 markdown 报告。

**测试（V2）**

```powershell
python learning/tool-use-mcp/src/tests/test_tools.py
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules tool-use-mcp --tests
```

> 注：`test_tools.py` 是脚本式聚合器（汇总 13 个模块的 `_self_test`），无 `test_` 函数；
> 经 harness 时 pytest 收集为空会**自动回退**按脚本直跑，预期 `13/13 modules passed`。

**关键坑注记**

- **MCP/A2A 都是 in-process mock**：`mcp_server.py`/`mcp_client.py` 靠 `.handle()` 直接函数调用模拟协议，
  不是真的 stdio subprocess/网络 transport（真实生产写法见 `lectures/04-mcp-server-impl.md`/`05-mcp-client-impl.md`
  里明确标注"真 stdio 模式（生产）"的对照代码块，仅供阅读，不在本模块运行范围内）。
- `toolformer_toy.py` 不训练模型，只复现论文的 `L_minus - L_plus` 筛选打分逻辑，用来解释"模型何时该学会调用工具"，
  和 L02-L12 讲的"运行时如何发现/调用/报错工具"是互补关系（对应 `paper/guide_01_toolformer.md` 第 24-25 节）。

## 关键文献 / 标准

- OpenAI function calling (2023.06)
- Anthropic MCP spec (2024.11) — JSON-RPC 2.0 over stdio / SSE
- Google A2A protocol (2025)
- Anthropic Computer Use (2024.10)
- e2b sandbox / Pyodide

## 一句话

> Tool 三代演化：OpenAI function calling → MCP (互操作) → A2A (多 agent 互操作) — 手写一遍 100% 跑通。
