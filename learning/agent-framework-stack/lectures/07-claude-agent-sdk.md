# L07 · Claude Agent SDK ⭐⭐⭐⭐⭐

## 30 秒核心

> Claude Agent SDK = Anthropic **官方 agent framework** (2025)，从 Claude Code 沉淀的"agent OS" 抽象出 SDK。

2025 上半年公测。

## 核心抽象

| 抽象 | 类比 |
|------|------|
| Agent | 主对象 |
| Tool | function calling 抽象 |
| Subagent | nested agent |
| Skill | 可复用 capability |
| Hooks | lifecycle 钩子 |
| Permission mode | 工具权限 (acceptEdits / askPermission) |

## 极简例（TypeScript）

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

const result = await query({
  prompt: 'Read package.json and tell me dependencies',
  options: {
    permissionMode: 'acceptEdits',
    allowedTools: ['Read', 'Bash'],
  },
});

for await (const message of result) {
  console.log(message);
}
```

Streaming response，messages 包含 tool call + text。

## 内置工具

| Tool | 用 |
|------|---|
| `Read` | 读文件 |
| `Edit` / `Write` | 写文件 |
| `Bash` | shell |
| `Grep` / `Glob` | 搜索 |
| `WebFetch` / `WebSearch` | web |
| `Task` | sub-agent |
| `TodoWrite` | task list |

Claude Code 自己用的工具直接暴露给 agent SDK。

## Custom tool

```typescript
import { tool } from '@anthropic-ai/claude-agent-sdk';

const myTool = tool({
  name: 'my_search',
  description: 'Custom search',
  inputSchema: z.object({ q: z.string() }),
  execute: async ({ q }) => ({ results: [...] }),
});
```

## Subagent

```typescript
const result = await query({
  prompt: 'Spawn a subagent to research X, then summarize.',
  options: {
    allowedTools: ['Task'],   // 启用 subagent
  },
});
```

`Task` tool 启动 nested agent — 独立 context，隔离 token budget。

## Hooks

```typescript
const result = await query({
  prompt: '...',
  options: {
    hooks: {
      preToolUse: async ({ tool, args }) => {
        if (tool === 'Bash' && args.command.includes('rm -rf')) {
          return { allowed: false, message: 'Blocked' };
        }
      },
    },
  },
});
```

→ 拦截 tool 前/后，policy / logging。

## Permission mode

| Mode | 行为 |
|------|------|
| `default` | 危险 tool 问用户 |
| `acceptEdits` | 自动批准编辑 |
| `bypassPermissions` | 全自动（dangerous） |
| `plan` | 只规划，不执行 |

## 与其他 framework 区别

| 维度 | Claude Agent SDK | LangGraph | OpenAI Agents SDK |
|------|------------------|-----------|-------------------|
| Vendor | Anthropic only | 任 vendor | OpenAI only |
| 内置 tool | 多 (filesystem/bash) | 无 | 一些 |
| Permission | 4 mode | HITL via interrupt | basic |
| Subagent | Task tool | sub-graph | handoff |
| MCP | ✓ first-class | ✓ | ✓ |

## 退出条件

- 能列 7 built-in tool
- 知道 4 permission mode
- 能写 preToolUse hook

## 一句话

> Claude Agent SDK = Claude Code 内核抽离成 SDK — built-in 工具 + permission mode + hooks，Anthropic 全栈首选。
