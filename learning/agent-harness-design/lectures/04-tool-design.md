# L04 · 工具设计原则

## 工具是给模型用的 API,不是给人用的

设计 agent 工具,直觉常常错。给人的 API 求灵活、求全;给模型的工具求**少、求明确、求省 token**。

## 原则清单

| 原则 | 反例 | 正例 |
|------|------|------|
| **粒度适中** | 一个 `do_everything(action, params)` | `read_file` / `write_file` 各管一件 |
| **名字自解释** | `proc()` / `handle()` | `search_code` / `send_email` |
| **描述决定选用** | "处理数据" | "搜索代码库,返回匹配的文件路径与行号" |
| **token 经济** | 返回整个文件 10万行 | 返回匹配段 + 行号,大输出裁剪(L05) |
| **错误可操作** | "failed" | "file not found: /x; did you mean /y?" |
| **read_only 标注** | 全部当可写 | 标 `read_only` 供权限层用(L09) |

## 描述是工具的灵魂

模型靠 `description` 判断"这个场景该不该用我"。[tools.py](../src/harness/tools.py) 的 `Tool.description` 会被 [system_prompt.py](../src/harness/system_prompt.py) 注入系统提示:

```python
for t in tools:
    ro = " (read-only)" if t.read_only else ""
    lines.append(f"- {t.name}: {t.description}{ro}")
```

描述含糊 → 模型选错工具(design-patterns 专题的 **tool sprawl** 反模式)。

## agent-facing vs human-facing 的取舍

| 维度 | human-facing API | agent-facing tool |
|------|-----------------|-------------------|
| 参数 | 多、可选、灵活 | 少、必填、明确 |
| 返回 | 完整结构 | 精炼、可直接喂回模型 |
| 错误 | 错误码 + 文档 | 自然语言、可操作 |
| 数量 | 越多越强 | **越少越好**(防选错) |

## token 经济:大输出必须裁剪

工具返回的大 blob(文件读、命令输出)进 context 前要裁剪——见 [context-management](05-context-management.md) 的 `trim` 思路。一个 `read_file` 把 10 万行全灌进去,既烧钱又触发 context rot。

## 与 tool-use-mcp 专题的关系

`tool-use-mcp` 讲的是工具调用的**协议**(MCP/JSON-RPC/A2A);本节讲的是**给定协议下怎么把工具设计好**。协议是管道,设计是水质。

## 退出条件
- [ ] 说出至少 5 条工具设计原则
- [ ] 理解 description 如何影响模型选工具
- [ ] 区分 agent-facing 与 human-facing 的取向差异
