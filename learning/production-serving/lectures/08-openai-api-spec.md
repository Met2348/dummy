# L08 · OpenAI-Compatible API 规范

## 1 · 为什么 OpenAI API 成标准
- ChatGPT 2022 出，开发者已用了 2 年
- 客户端 SDK 极完善（openai-python / openai-js / langchain）
- 所有开源 server 都支持它

## 2 · 主要 endpoints
| endpoint | 用 |
|---------|---|
| POST /v1/chat/completions | chat |
| POST /v1/completions | 完成（旧）|
| POST /v1/embeddings | embedding |
| GET /v1/models | 列表 |

## 3 · chat/completions 请求
```json
{
    "model": "qwen2.5-7b",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "1+1?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": false
}
```

## 4 · 响应
```json
{
    "id": "chatcmpl-xxx",
    "object": "chat.completion",
    "created": 1730000000,
    "model": "qwen2.5-7b",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "2"},
        "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
}
```

## 5 · 流式响应（SSE）
```
data: {"id":"...", "choices":[{"delta":{"content":"2"},"index":0}]}
data: {"id":"...", "choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## 6 · 错误格式
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": null}}
```

## 7 · 兼容性挑战
- function_call / tool 格式
- json_mode / response_format
- vision / multimodal
- 不同后端实现 50% 兼容

## 8 · 实现：[openai_api_server.py](../src/openai_api_server.py)
- 最简 FastAPI 实现
- /v1/models + /v1/chat/completions（含 stream）
- mock backend（不依赖真模型）
