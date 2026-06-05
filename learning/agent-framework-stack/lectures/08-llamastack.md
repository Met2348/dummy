# L08 · LlamaStack（Meta 2024.09）

## 30 秒核心

> LlamaStack = Meta 推出的**全栈 agent framework**：
> Meta 自己用 Llama 全套 (inference / safety / memory / agents / eval / telemetry)。

2024.09 公布，对标 OpenAI 全栈。

## 5 layer

| Layer | 提供 |
|-------|------|
| **Inference** | LLM serving (vLLM / Together / Fireworks / Groq / Cerebras) |
| **Safety** | Llama Guard 集成 |
| **Memory** | 向量 store + faiss |
| **Agents** | tool use + ReAct |
| **Eval** | 标准 benchmark runner |
| **Telemetry** | OTLP trace |

## SDK 调用

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:5000")

response = client.agents.create(
    agent_config={
        "model": "Llama-3.3-70B-Instruct",
        "tools": [...],
        "input_shields": ["llama-guard"],
        "output_shields": ["llama-guard"],
    }
)
```

## Built-in shields

```python
"input_shields": ["llama-guard"],
"output_shields": ["llama-guard"],
```

→ 内置 Llama Guard 3，输入输出双过滤。

## Provider 模型

```
LlamaStack 是 spec + multiple provider implementations.

Provider:
  - meta-reference (Meta 实现)
  - ollama, together, fireworks, vllm
  - remote-hf 等
```

→ "K8s 之于 cloud" 那种 spec + impl 模型。

## 优势

| 优势 | 解释 |
|------|------|
| 全栈一站 | 不用拼 5 框架 |
| Llama 优化 | Meta 自己 LLM |
| Safety 内置 | 默认开 |
| Provider 灵活 | 切 vendor 简单 |

## 弱势

| 弱 | 解释 |
|----|----|
| 2024 新，社区小 | 文档不全 |
| Meta-centric | 别家 LLM 不一等 |
| 重 | "全栈"意味学曲线陡 |

## 适合谁

| 适合 | 不适合 |
|------|--------|
| Meta 全栈用户 | 已有 LangChain 投资 |
| 多 provider 切换 | 单一 vendor 简单需求 |
| Safety 强需求 | 快速 PoC |

## 退出条件

- 知道 5 layer
- 能列 input/output shields
- 知道 spec + provider 模型

## 一句话

> LlamaStack = Meta 全栈 agent framework spec — 5 layer + Llama Guard 内置 + 多 provider，对标 OpenAI 全栈。
