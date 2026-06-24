"""OpenAI-compatible API surface as a plain dict transformer.

We split the protocol concerns from the FastAPI layer so the same logic is
testable without a running server.  `app` is only built when FastAPI is
installed; otherwise the module exposes the pure helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import json
import time
import uuid


MOCK_TOKENS = ["Hello", " there", ",", " how", " can", " I", " help", "?", ""]


# ---- Protocol shapes -------------------------------------------------------


def build_models_response() -> Dict:
    return {
        "object": "list",
        "data": [
            {"id": "mock-7b", "object": "model", "created": 1730000000, "owned_by": "mock"},
        ],
    }


def build_completion_response(req: Dict, full_text: str, prompt_tokens: int, out_tokens: int) -> Dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(1730000000),
        "model": req["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": out_tokens,
            "total_tokens": prompt_tokens + out_tokens,
        },
    }


def build_stream_chunk(model: str, delta: Dict, finish: Optional[str] = None) -> Dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(1730000000),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def build_error(message: str, type_: str = "invalid_request_error", code: Optional[str] = None) -> Dict:
    return {"error": {"message": message, "type": type_, "code": code}}


# ---- Validation ------------------------------------------------------------


def validate_chat_request(req: Dict) -> Optional[str]:
    if "model" not in req:
        return "missing 'model'"
    if "messages" not in req or not isinstance(req["messages"], list) or not req["messages"]:
        return "missing or empty 'messages'"
    for m in req["messages"]:
        if "role" not in m or "content" not in m:
            return "message missing role/content"
        if m["role"] not in ("user", "assistant", "system", "tool"):
            return f"invalid role: {m['role']}"
    return None


# ---- Mock backend ----------------------------------------------------------


def mock_stream(req: Dict) -> Iterable[str]:
    for tok in MOCK_TOKENS:
        yield tok


def mock_generate(req: Dict) -> tuple[str, int, int]:
    text = "".join(mock_stream(req))
    prompt_tokens = sum(len(m["content"].split()) for m in req["messages"])
    out_tokens = len(text.split())
    return text, prompt_tokens, out_tokens


# ---- FastAPI hook ---------------------------------------------------------


def make_app():
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:
        return None

    app = FastAPI(title="mock-llm")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return build_models_response()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        err = validate_chat_request(body)
        if err:
            return JSONResponse(status_code=400, content=build_error(err))
        if body.get("stream"):
            async def event_gen():
                for tok in mock_stream(body):
                    chunk = build_stream_chunk(body["model"], {"content": tok})
                    yield f"data: {json.dumps(chunk)}\n\n"
                done = build_stream_chunk(body["model"], {}, finish="stop")
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_gen(), media_type="text/event-stream")
        text, prompt_tokens, out_tokens = mock_generate(body)
        return build_completion_response(body, text, prompt_tokens, out_tokens)

    return app


app = make_app()


def demo() -> None:
    """演示协议层（构造/校验/mock 生成），**不起阻塞服务**。"""
    print("=== OpenAI 兼容 API（协议层 demo，不起服务）===")
    req = {"model": "mock-7b", "messages": [{"role": "user", "content": "hi there"}]}
    print("validate(good):", validate_chat_request(req) or "OK")
    print("validate(bad) :", validate_chat_request({"messages": []}))
    text, pt, ot = mock_generate(req)
    resp = build_completion_response(req, text, pt, ot)
    print(f"completion content: {resp['choices'][0]['message']['content']!r}")
    print(f"usage: {resp['usage']}")
    print(f"\nFastAPI app 已构建: {app is not None}")
    print("起真服务：  cd learning/production-serving/src; uvicorn openai_api_server:app --port 8000")


if __name__ == "__main__":
    demo()
