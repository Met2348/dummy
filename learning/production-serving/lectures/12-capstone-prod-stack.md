# L12 · Capstone — 生产栈完整部署

## 1 · 目标
拼装一套**最小可用 production**：
- FastAPI OpenAI 兼容 API
- Mock LLM backend（教学；真用 vllm.AsyncLLMEngine）
- Prometheus metrics
- 健康检查 / 错误处理
- 100 QPS 稳定测试

## 2 · 组件
```
client → FastAPI → mock LLM → response
              ↘ metrics (/metrics)
              ↘ health (/health)
```

## 3 · 启动
```bash
uvicorn src.openai_api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 4 · 端到端测试
```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
client.chat.completions.create(model="mock", messages=[{"role":"user","content":"hi"}])
```

## 5 · 退出条件
- /v1/chat/completions 兼容 openai-python
- /v1/models 返回 model list
- /health 返回 200
- /metrics 返回 Prometheus 格式
- 测试 stream / non-stream 两种模式
- 测试 错误格式
- 实测：单 worker 100 QPS, p99 < 500 ms (mock)

## 6 · 真栈替换
```python
# 替换 mock_backend
from vllm import AsyncLLMEngine, SamplingParams
engine = AsyncLLMEngine.from_engine_args(...)

async def llm_generate(messages, params):
    async for output in engine.generate(...):
        yield output.outputs[0].text
```

## 7 · 实现
- [openai_api_server.py](../src/openai_api_server.py) FastAPI
- [streaming_sse.py](../src/streaming_sse.py) SSE helper
- [metrics_prometheus.py](../src/metrics_prometheus.py) metrics
- [tests/test_e2e.py](../src/tests/test_e2e.py) 端到端 httpx test
