# L04 · Tool Stack 选型

## DRA 需要 5 类 tool

| 类 | 例 | 来源 |
|---|---|------|
| **Web search** | google / brave / serper | API |
| **Web fetch** | fetch URL → markdown | API / Playwright |
| **Citation** | format [1] (Author, 2025) | 内部 |
| **File write** | save md report | 内部 / FS |
| **Python exec** | 算 / 解析 | sandbox (e2b) |

## 我们 capstone 用 5 mock tool

| Tool | 实现 |
|------|------|
| `search` | canned KB |
| `fetch` | KB lookup |
| `cite` | 生成 [N] |
| `file_write` | in-memory FS |
| `python` | sandbox AST exec |

全部 stdlib，无外部依赖。

## 真生产 tool stack

| 选项 | 用 |
|------|---|
| **search** | Brave Search API / Tavily / Serper |
| **fetch** | Firecrawl / Mendable / Playwright |
| **cite** | 自写 |
| **file write** | FS / S3 |
| **python** | e2b / Modal / Daytona |

## MCP 化

所有 tool 通过 MCP 暴露：
- 易切换 vendor
- 多 server 拼合
- 跨 framework 互操作

## Permission

DRA 通常用 `acceptEdits` mode（不需用户每步确认），但保留：
- 大文件 write → confirm
- shell exec → block
- 外网访问 → log

## 退出条件

- 能列 5 tool 类
- 能列 5 商业 vendor
- 知道 MCP 优势

## 一句话

> DRA tool stack = search + fetch + cite + file + python — 5 tool 解决 95% deep research 需要。
