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


def demo() -> None:
    print("=== Mock VLM 服务（图像编码 + 文本生成）===")
    req = VlmRequest(text="describe these", images=["cat.jpg", "dog.png"])
    print(f"请求: text={req.text!r}, {len(req.images)} 张图")
    print(f"图像编码(每张恒 8 维): {encode_image_mock(req.images[0])}")
    print(f"VLM 输出: {vlm_generate(req)}")
    print(f"纯文本请求: {vlm_generate(VlmRequest(text='hello'))}")


if __name__ == "__main__":
    demo()
