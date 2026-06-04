"""SSE helper for streaming OpenAI-style responses."""
from __future__ import annotations

from typing import Iterable, Iterator, Dict
import json


def sse_encode(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def chunks_to_sse(chunks: Iterable[Dict]) -> Iterator[str]:
    for c in chunks:
        yield sse_encode(c)
    yield sse_done()


def parse_sse_line(line: str) -> Dict | None:
    line = line.strip()
    if not line.startswith("data: "):
        return None
    body = line[len("data: ") :]
    if body == "[DONE]":
        return {"_done": True}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None
