"""Mock VLM serving - image encoder + LLM offline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VlmRequest:
    text: str
    images: List[str] = None   # paths or base64 placeholders


def encode_image_mock(path_or_b64: str) -> List[float]:
    """Always return exactly 8 floats, padding short inputs."""
    padded = path_or_b64.ljust(8, "0")[:8]
    return [float(ord(c)) for c in padded]


def vlm_generate(req: VlmRequest) -> str:
    visual = []
    if req.images:
        for img in req.images:
            visual += encode_image_mock(img)
    return f"<vlm response to '{req.text}' with {len(visual)//8} images>"
