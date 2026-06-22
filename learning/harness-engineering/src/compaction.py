"""
compaction.py — L04: 5 阶段渐进式 context compaction。

逆向 Claude Code 揭示的生产级做法: 上下文逼近预算时, 不是一刀切 summarize, 而是
**逐级升级**, 能用轻手段解决就不用重手段, 尽量保留信息:

  Stage 1 budget reduction  截断单条超大 tool 输出 (最轻)
  Stage 2 snip              丢弃最老的低价值消息 (旧 tool_result)
  Stage 3 microcompact      把一段较老消息压成一条短摘要
  Stage 4 context collapse  仅保留最近 K 条, 其余塌缩成一条运行摘要
  Stage 5 auto-compact      整窗重置: [system, 全量摘要, 最近几条] (最重)

每条消息: {"role","content","kind"?,"pinned"?}。system / pinned 永不丢弃。
无真 LLM 时摘要用确定性占位 (只为演示机制与 token 曲线; 真实接 LLM 时换成模型摘要)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from provider import approx_tokens

RECENT_PROTECT = 3            # 最近 N 条受保护, 早期阶段不动
SINGLE_MSG_CAP = 400          # 单条消息 token 上限 (stage 1)


def total_tokens(messages: list[dict]) -> int:
    return sum(approx_tokens(m.get("content", "")) for m in messages)


def _is_pinned(m: dict) -> bool:
    return m.get("role") == "system" or m.get("pinned")


@dataclass
class CompactionEvent:
    stage: int
    name: str
    tokens_before: int
    tokens_after: int
    note: str = ""

    @property
    def freed(self) -> int:
        return self.tokens_before - self.tokens_after


def _summarize(messages: list[dict], ratio: float = 0.18) -> dict:
    """确定性占位摘要: 把若干消息压成一条, token 约为原来的 ratio。

    真实 harness 在这里调一次 LLM 做语义摘要; 这里只演示'信息被有损压缩'这件事本身。
    """
    src_tok = total_tokens(messages)
    target = max(20, int(src_tok * ratio))
    roles = [m.get("role", "?") for m in messages]
    head = (messages[0].get("content", "")[:60]) if messages else ""
    body = f"[摘要] 压缩了 {len(messages)} 条消息(角色:{','.join(roles[:6])}{'…' if len(roles)>6 else ''}); 起始: {head}…"
    # 把 body 补/截到 target token 量级, 保证确实变小
    while approx_tokens(body) < target:
        body += " …(保留要点)"
    body = body[: target * 4]
    return {"role": "system", "content": body, "kind": "summary", "pinned": True}


class Compactor:
    """对一段对话历史做渐进式压缩, 直到落到 max_tokens 以下。"""

    def __init__(self, max_tokens: int, recent_protect: int = RECENT_PROTECT):
        self.max_tokens = max_tokens
        self.recent_protect = recent_protect

    # --- 各阶段: 返回 (新 messages, 是否动了手) ---
    def _stage1_budget(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        changed = False
        out = []
        for m in msgs:
            if not _is_pinned(m) and approx_tokens(m.get("content", "")) > SINGLE_MSG_CAP:
                c = m.get("content", "")
                m = {**m, "content": c[: SINGLE_MSG_CAP * 4] + " …[截断]"}
                changed = True
            out.append(m)
        return out, changed

    def _stage2_snip(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        # 丢最老的一条"低价值"消息 (tool_result/旧 user/旧 assistant), 保护 system 与最近 N
        protect_idx = set(range(len(msgs) - self.recent_protect, len(msgs)))
        for i, m in enumerate(msgs):
            if i in protect_idx or _is_pinned(m):
                continue
            if m.get("kind") == "tool_result" or m.get("role") in ("user", "assistant"):
                return msgs[:i] + msgs[i + 1:], True
        return msgs, False

    def _stage3_microcompact(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        # 把"较老的一段" (前半段里的非 pinned) 压成一条摘要
        movable = [i for i, m in enumerate(msgs)
                   if not _is_pinned(m) and i < len(msgs) - self.recent_protect]
        if len(movable) < 2:
            return msgs, False
        block_idx = movable[: max(2, len(movable) // 2)]
        block = [msgs[i] for i in block_idx]
        summary = _summarize(block, ratio=0.25)
        keep = [m for i, m in enumerate(msgs) if i not in set(block_idx)]
        # 摘要插到原块位置 (放在最前的 pinned 之后)
        insert_at = sum(1 for m in keep if _is_pinned(m) and m.get("kind") != "summary")
        return keep[:insert_at] + [summary] + keep[insert_at:], True

    def _stage4_collapse(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        # 仅保留最近 K 条 + pinned, 其余全部塌缩成一条运行摘要
        recent = msgs[-self.recent_protect:]
        pinned = [m for m in msgs[:-self.recent_protect] if _is_pinned(m)]
        collapsible = [m for m in msgs[:-self.recent_protect] if not _is_pinned(m)]
        if not collapsible:
            return msgs, False
        summary = _summarize(collapsible, ratio=0.12)
        return pinned + [summary] + recent, True

    def _stage5_autocompact(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        # 整窗重置: system + 全量摘要 + 最近几条
        system = [m for m in msgs if m.get("role") == "system" and m.get("kind") != "summary"]
        recent = msgs[-2:]
        rest = [m for m in msgs if m not in system and m not in recent]
        summary = _summarize(rest or msgs, ratio=0.08)
        return system + [summary] + recent, True

    STAGES = [
        (1, "budget_reduction", "_stage1_budget"),
        (2, "snip", "_stage2_snip"),
        (3, "microcompact", "_stage3_microcompact"),
        (4, "context_collapse", "_stage4_collapse"),
        (5, "auto_compact", "_stage5_autocompact"),
    ]

    def compact(self, messages: list[dict]) -> tuple[list[dict], list[CompactionEvent]]:
        """渐进式压缩直到 <= max_tokens 或手段用尽。返回 (新 messages, 事件列表)。"""
        msgs = list(messages)
        events: list[CompactionEvent] = []
        if total_tokens(msgs) <= self.max_tokens:
            return msgs, events
        for stage, name, fn in self.STAGES:
            # 同一阶段可重复施用 (如 snip 一次丢一条), 直到本阶段无效或已达标
            guard = 0
            while total_tokens(msgs) > self.max_tokens and guard < 200:
                guard += 1
                before = total_tokens(msgs)
                new, changed = getattr(self, fn)(msgs)
                if not changed:
                    break
                after = total_tokens(new)
                events.append(CompactionEvent(stage, name, before, after))
                msgs = new
            if total_tokens(msgs) <= self.max_tokens:
                break
        return msgs, events
