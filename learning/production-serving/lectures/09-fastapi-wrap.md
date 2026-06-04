# L09 · FastAPI + SSE 流式包装

## 1 · 为什么 FastAPI
- 异步原生（async/await）
- pydantic 自动校验
- 自动 OpenAPI 文档
- uvicorn 高性能

## 2 · 基本结构
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(stream_gen(req), media_type="text/event-stream")
    return ChatResponse(...)
```

## 3 · SSE 流式
```python
from sse_starlette.sse import EventSourceResponse

async def stream_gen(req):
    async for tok in model.stream(req.messages):
        yield {"data": json.dumps({"choices":[{"delta":{"content":tok}}]})}
    yield {"data": "[DONE]"}

@app.post("/v1/chat/completions")
async def chat(req):
    return EventSourceResponse(stream_gen(req))
```

## 4 · 错误处理
```python
@app.exception_handler(ValidationError)
async def validation_handler(req, exc):
    return JSONResponse(status_code=422, content={"error": {"message": str(exc)}})
```

## 5 · 并发
- uvicorn `--workers 4` 多进程
- 每 worker async event loop
- 避免阻塞操作（用 `run_in_executor` for sync code）

## 6 · 路由
- nginx / Caddy 反代
- 健康检查 endpoint `/health`
- 版本路由 `/v1/`, `/v2/`

## 7 · 实现：[openai_api_server.py](../src/openai_api_server.py)
完整 FastAPI server，含 SSE 流式。
