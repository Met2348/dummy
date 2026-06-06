# L07 · DRA Deployment

## 部署形态

| 形态 | 例 |
|------|---|
| **CLI tool** | Claude Code / Aider |
| **Web app** | Perplexity / ChatGPT |
| **API** | OpenAI Assistants API |
| **MCP server** | 暴露为 MCP tool 给其他 agent |
| **Slack/Discord bot** | Internal use |
| **Browser extension** | 网页边栏 |

## 关键技术栈

| Layer | 选项 |
|-------|------|
| Backend | FastAPI / Next.js API route |
| Stream | SSE / WebSocket |
| Auth | JWT / OAuth |
| State | Postgres / Redis |
| Queue | Celery / BullMQ |
| Vector | Pinecone / Chroma |
| LLM | Anthropic / OpenAI / Bedrock |
| Trace | LangSmith / Phoenix |

## FastAPI 例

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.post("/research")
async def research(query: str):
    async def stream():
        async for chunk in dra_run(query):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
```

## 后台 task 模式

长任务 (3-30 分钟)：
```
POST /research → 返 task_id
GET /research/{task_id}/status → polling
GET /research/{task_id}/result → 完成后
GET /research/{task_id}/stream → SSE 实时
```

## Scaling

| 用户量 | 架构 |
|-------|------|
| <100 QPD | 单机 |
| 100-1k QPD | LB + worker queue |
| 1k+ QPD | K8s + autoscale + multi-region |
| 10k+ | 自己 LLM serving (vLLM/SGLang) |

## Cost 控制

| 措施 | 节省 |
|------|------|
| Prompt cache | 80% input |
| Tool result cache | 50% tool calls |
| Model tier (Haiku 当 sub-agent) | 30-50% |
| Streaming → 不计未完 step | 跑死了不收费 |
| Per-user quota | 防滥用 |

## 退出条件

- 能列 6 部署形态
- 能写 FastAPI streaming endpoint
- 知道 scaling 4 阶段

## 一句话

> DRA 部署 6 形态 + FastAPI/Next.js streaming + 4 阶段 scaling — cost 控制是核心。
