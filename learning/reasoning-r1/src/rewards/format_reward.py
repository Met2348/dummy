"""R1-Zero 的 format reward — 检查 response 是否符合 <think>...</think><answer>...</answer> 格式."""
from __future__ import annotations

import re

# 严格 regex
_FORMAT_RE = re.compile(
    r"^<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$",
    re.DOTALL,
)


def format_reward(response: str) -> float:
    """1.0 if 严格符合 <think>...</think><answer>...</answer>；0 otherwise."""
    if not isinstance(response, str):
        return 0.0
    m = _FORMAT_RE.match(response.strip())
    if m and len(m.group(1).strip()) > 0 and len(m.group(2).strip()) > 0:
        return 1.0
    return 0.0


def extract_answer(response: str) -> str | None:
    """提取 <answer>...</answer> 内容。失败返回 None."""
    m = re.search(r"<answer>(.+?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _self_test():
    good = "<think>I need to add 2+3=5</think><answer>5</answer>"
    print(f"good: {format_reward(good)}")    # 1.0
    print(f"  answer: {extract_answer(good)!r}")

    no_think = "<answer>5</answer>"
    print(f"no_think: {format_reward(no_think)}")    # 0

    empty_think = "<think></think><answer>5</answer>"
    print(f"empty_think: {format_reward(empty_think)}")    # 0


if __name__ == "__main__":
    _self_test()
