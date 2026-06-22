"""
provider.py — L03: 把"真模型"抽象成一个可替换的 Provider。

教学核心: harness 不该和某个具体模型 API 焊死。模型只是一个"无状态 token 预测器",
harness 通过一个**窄接口** (stream) 和它对话; 换 Anthropic / OpenAI / 开源 / Mock,
harness 主体一行不改。这就是 harness engineering 的"模型可替换"原则。

默认 MockProvider: 确定性、模拟流式 + tool-call、无需 API key, 让所有 notebook 在
Windows native 上可跑。真 provider (AnthropicProvider/OpenAIProvider) 给出接口骨架。
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def approx_tokens(text: str) -> int:
    """粗略 token 估计 (跨模块统一口径): ~4 字符/token, 中文按字符。够教学/对照用。"""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ----------------------------------------------------------------- 流式 chunk
@dataclass
class Chunk:
    """provider 流式输出的最小单元。harness 消费它们来驱动 loop。"""
    kind: str                      # "text" | "tool_call" | "done"
    text: str = ""                 # kind=="text" 时的增量文本
    tool_name: str = ""            # kind=="tool_call"
    tool_args: dict = field(default_factory=dict)
    stop: bool = False             # kind=="done" 时, 模型是否认为任务可结束


# ----------------------------------------------------------------- 抽象基类
class Provider(ABC):
    """harness 看到的唯一模型接口。把模型当无状态函数: messages -> chunk 流。"""

    @abstractmethod
    def stream(self, messages: list[dict], tools: Optional[list[dict]] = None) -> Iterator[Chunk]:
        ...

    def count_tokens(self, messages: list[dict]) -> int:
        return sum(approx_tokens(m.get("content", "")) for m in messages)


# ----------------------------------------------------------------- Mock (默认)
@dataclass
class Turn:
    """MockProvider 的一次回合脚本: 先吐 text, 再发若干 tool_call, 最后 done(stop=?)。"""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{"name":..., "args":{...}}]
    stop: bool = False


class MockProvider(Provider):
    """确定性替身模型。

    两种用法:
      1) 给 script=[Turn, Turn, ...]: 每调用一次 stream 消费下一个 Turn (驱动多步 loop)。
      2) 不给 script: 默认策略 —— 若 user 文本里提到工具名则发一个 echo tool_call, 否则纯文本回答。
    """

    def __init__(self, script: Optional[list[Turn]] = None):
        self.script = list(script) if script else None
        self._i = 0
        self.calls = 0

    def stream(self, messages: list[dict], tools: Optional[list[dict]] = None) -> Iterator[Chunk]:
        self.calls += 1
        if self.script is not None:
            turn = self.script[min(self._i, len(self.script) - 1)]
            self._i += 1
            yield from self._emit(turn)
            return
        # 默认策略 (无脚本): 确定性, 依据最后一条 user 消息
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        content = last.get("content", "")
        tool_names = [t.get("name") for t in (tools or [])]
        used = next((n for n in tool_names if n and n in content), None)
        if used:
            yield from self._emit(Turn(text=f"我将调用 {used}。", tool_calls=[{"name": used, "args": {"q": content}}]))
        else:
            yield from self._emit(Turn(text=f"(mock) 已读取 {approx_tokens(content)} tokens 的输入并作答。", stop=True))

    @staticmethod
    def _emit(turn: Turn) -> Iterator[Chunk]:
        # 模拟流式: 文本拆成几段增量吐出
        if turn.text:
            words = turn.text.split()
            step = max(1, len(words) // 3)
            for i in range(0, len(words), step):
                yield Chunk(kind="text", text=" ".join(words[i:i + step]) + " ")
        for tc in turn.tool_calls:
            yield Chunk(kind="tool_call", tool_name=tc["name"], tool_args=tc.get("args", {}))
        yield Chunk(kind="done", stop=turn.stop)


# ----------------------------------------------------------------- 真 provider 骨架
class AnthropicProvider(Provider):
    """接 Anthropic 的骨架。需 `pip install anthropic` + ANTHROPIC_API_KEY。

    教学占位: 真实实现把 messages 转成 Anthropic 的 messages 格式, 用流式 API,
    把 content_block_delta -> Chunk(text), tool_use -> Chunk(tool_call)。
    本专题默认不走它 (无 key), 仅展示"换 provider 不动 harness 主体"。
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def stream(self, messages, tools=None):  # pragma: no cover - 需真实 key
        try:
            import anthropic  # noqa: F401
        except Exception as e:
            raise RuntimeError("接真模型需 `pip install anthropic` 并设 ANTHROPIC_API_KEY") from e
        raise NotImplementedError("教学骨架: 在此把 Anthropic 流式事件映射为 Chunk。默认用 MockProvider。")
