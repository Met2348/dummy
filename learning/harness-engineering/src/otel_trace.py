"""
otel_trace.py — L09: 生产级可观测性 (OpenTelemetry 风格)。

2026 生产标准: 用 OpenTelemetry 的 span 模型描述 agent 运行 ——
  每个 reasoning step 是一个 span; 每个 tool call 是它的 child span; 窗口/子 agent 再套一层。
当 8 小时长任务在第 7 小时挂掉, 有结构化 trace 才能"精确知道哪一步、哪一个工具、什么输入"出的事,
而不是从头重放整段会话。

为保证 notebook 输出可复现, 这里用**逻辑时钟** (单调计数器) 代替墙钟时间。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


@dataclass
class Span:
    name: str
    kind: str                       # "window" | "reasoning" | "tool" | "subagent"
    start: int
    end: Optional[int] = None
    attributes: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration(self) -> int:
        return (self.end if self.end is not None else self.start) - self.start


class Tracer:
    """最小 OTel 风格 tracer: 支持嵌套 span、逻辑时钟、导出树/字典/文本。"""

    def __init__(self):
        self._clock = 0
        self.roots: list[Span] = []
        self._stack: list[Span] = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def start(self, name: str, kind: str, **attrs) -> Span:
        span = Span(name=name, kind=kind, start=self._tick(), attributes=dict(attrs))
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.roots.append(span)
        self._stack.append(span)
        return span

    def end(self, span: Optional[Span] = None, **attrs) -> None:
        if not self._stack:
            return
        cur = self._stack.pop()
        cur.end = self._tick()
        cur.attributes.update(attrs)

    # 上下文管理器糖: with tracer.span("reason","reasoning"): ...
    def span(self, name: str, kind: str, **attrs):
        tracer = self

        class _Ctx:
            def __enter__(self_):
                self_.s = tracer.start(name, kind, **attrs)
                return self_.s

            def __exit__(self_, *exc):
                tracer.end(self_.s)
                return False
        return _Ctx()

    # --- 导出 ---
    def to_dict(self) -> list[dict]:
        def conv(s: Span) -> dict:
            return {"name": s.name, "kind": s.kind, "start": s.start, "end": s.end,
                    "duration": s.duration, "attributes": s.attributes,
                    "children": [conv(c) for c in s.children]}
        return [conv(r) for r in self.roots]

    def stats(self) -> dict:
        """聚合: 各 kind 的 span 数与总时长 (逻辑单位)。"""
        agg: dict[str, dict] = {}

        def walk(s: Span):
            a = agg.setdefault(s.kind, {"count": 0, "duration": 0})
            a["count"] += 1
            a["duration"] += s.duration
            for c in s.children:
                walk(c)
        for r in self.roots:
            walk(r)
        return agg

    def render(self) -> str:
        """文本树, 便于 notebook / 终端查看。"""
        lines: list[str] = []
        icon = {"window": "▭", "reasoning": "◆", "tool": "→", "subagent": "▣"}

        def walk(s: Span, depth: int):
            pad = "  " * depth
            attr = ""
            if s.attributes:
                kv = ", ".join(f"{k}={v}" for k, v in list(s.attributes.items())[:3])
                attr = f"  [{kv}]"
            lines.append(f"{pad}{icon.get(s.kind,'·')} {s.name} (Δ{s.duration}){attr}")
            for c in s.children:
                walk(c, depth + 1)
        for r in self.roots:
            walk(r, 0)
        return "\n".join(lines)
