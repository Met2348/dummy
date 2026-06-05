# L06 · Vercel AI SDK

## 30 秒核心

> Vercel AI SDK = **TypeScript-first** agent framework，专为 Edge / streaming UI 优化。

Vercel 2023.06 推出，v4 (2024.12) 重大更新。

## 4 layer 设计

| Layer | 包 | 用 |
|-------|---|---|
| **Core** | `ai` | LLM SDK 抽象 |
| **UI** | `@ai-sdk/react` | useChat / useCompletion hook |
| **Providers** | `@ai-sdk/anthropic` 等 | LLM vendor |
| **Tools** | 内置 + custom | tool calling |

## 极简例

```typescript
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const result = await generateText({
  model: anthropic('claude-sonnet-4'),
  prompt: 'What is ReAct?',
});
console.log(result.text);
```

## Tool use

```typescript
import { generateText, tool } from 'ai';
import { z } from 'zod';

const result = await generateText({
  model: anthropic('claude-sonnet-4'),
  tools: {
    getWeather: tool({
      description: 'Get weather',
      parameters: z.object({ city: z.string() }),
      execute: async ({ city }) => ({ temp: 22 }),
    }),
  },
  maxSteps: 5,  // multi-step agent loop
  prompt: 'Weather in Tokyo?',
});
```

zod schema 类似 Pydantic。

## Streaming UI

```typescript
// API route
import { streamText } from 'ai';
export async function POST(req: Request) {
  const result = streamText({ model: ..., messages: ... });
  return result.toDataStreamResponse();
}

// React component
'use client';
import { useChat } from '@ai-sdk/react';
function Chat() {
  const { messages, input, handleSubmit } = useChat();
  return <>{messages.map(m => ...)}</>;
}
```

→ Next.js + Edge runtime 友好。

## Multi-step agent (v4)

```typescript
const result = await generateText({
  model,
  tools: { search, calc },
  maxSteps: 10,  // ← agent loop
  prompt: 'Research Tokyo weather and convert C to F',
});
```

自动 ReAct loop，maxSteps 控制深度。

## 与 Python framework 对比

| 维度 | Vercel AI SDK | LangChain Python |
|------|--------------|------------------|
| 语言 | TS 优先 | Python 优先 |
| UI 集成 | React hooks 一等 | Streamlit / Gradio |
| Edge runtime | ✓ 设计为此 | 不天然 |
| Streaming | 默认开 | 需 explicit |
| 数据 framework | 弱 | 强 |
| Multi-agent | 弱 | 强 |

## 适合谁

| 适合 | 不适合 |
|------|--------|
| Next.js 团队 | Python 团队 |
| Edge / serverless | 长 task |
| Chat / streaming UI | 复杂 graph workflow |
| TS 生态 | Heavy data pipeline |

## 退出条件

- 能讲 4 layer
- 能默写 generateText + tool
- 知道 maxSteps 启用 agent loop

## 一句话

> Vercel AI SDK = TS-first edge-native streaming agent — Next.js + Anthropic 友好，maxSteps 一键 multi-step。
