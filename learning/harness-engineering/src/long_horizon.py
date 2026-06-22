"""
long_horizon.py — L05: long-horizon 自治 (loop-with-hook + 文件系统 state)。

2026 公认最难的问题: 让 agent 跨多个上下文窗口连续工作数小时。今天的模型有三个病:
  ① early stopping (没干完就想收工)  ② 复杂任务分解差  ③ 跨窗口失忆 (上下文一换就忘)。

harness 的解法 (loop-with-hook):
  - 把"真相"放到**文件系统** (而非上下文): todo / 进度 / 笔记落盘。
  - 每个窗口**从干净状态起步**, 但开头从文件系统**读回 state** —— 上下文会忘, 文件不会。
  - 一个 **hook 拦截模型的"收工"动作**: 若 completion-goal 未达成, 不让它停, 把目标重新注入一个
    新窗口, 继续干。每窗口干净起步、靠文件系统接力, 直到目标真正达成。

本模块提供: FileStateStore (落盘 state) + ToolRegistry + run_long_horizon (带 hook 的外循环)。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from provider import Provider, approx_tokens
from compaction import Compactor, total_tokens


# ----------------------------------------------------------------- 文件系统 state
class FileStateStore:
    """把 agent 的"真相"持久到磁盘: 上下文窗口会换、会忘, 这里不会。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = self.path / "state.json"
        if not self.file.exists():
            self._write({"progress": 0, "notes": [], "todo": []})

    def _write(self, state: dict) -> None:
        self.file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8"))

    def save(self, state: dict) -> None:
        self._write(state)

    def summary(self) -> str:
        """给下一个窗口的"读回"摘要 —— 跨窗口记忆的载体。"""
        s = self.load()
        notes = "; ".join(s.get("notes", [])[-3:])
        return f"进度={s.get('progress',0)}/{s.get('goal_total','?')}; 最近笔记: {notes or '(无)'}"


# ----------------------------------------------------------------- 工具
class ToolRegistry:
    """name -> fn(args, state)->str。工具可读写 state (即文件系统真相)。"""

    def __init__(self):
        self._tools: dict[str, Callable[[dict, dict], str]] = {}

    def register(self, name: str, fn: Callable[[dict, dict], str]) -> None:
        self._tools[name] = fn

    def specs(self) -> list[dict]:
        return [{"name": n} for n in self._tools]

    def dispatch(self, name: str, args: dict, state: dict) -> str:
        if name not in self._tools:
            return f"[error] 未知工具 {name}"
        return self._tools[name](args, state)


# ----------------------------------------------------------------- 运行记录
@dataclass
class WindowRecord:
    index: int
    steps: int = 0
    tokens_peak: int = 0
    compactions: int = 0
    stop_intercepted: bool = False


@dataclass
class RunResult:
    success: bool
    windows: list[WindowRecord] = field(default_factory=list)
    total_steps: int = 0
    final_state: dict = field(default_factory=dict)
    context_tokens_total: int = 0      # 累计喂给模型的上下文 token (成本代理)
    aborted_early: bool = False        # 无 hook 时被 early-stop 中断

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def _seed_messages(goal: str, store: FileStateStore) -> list[dict]:
    """每个新窗口的干净起点: system + 目标 + 从文件系统读回的 state 摘要。"""
    return [
        {"role": "system", "content": "你是一个长任务执行 agent。每步调用一个工具推进, 完成后才停。", "pinned": True},
        {"role": "system", "content": f"目标: {goal}", "pinned": True},
        {"role": "user", "content": f"当前进度 (从文件系统读回): {store.summary()}。请继续。"},
    ]


def run_long_horizon(
    provider: Provider,
    goal: str,
    tools: ToolRegistry,
    store: FileStateStore,
    goal_met: Callable[[dict], bool],
    *,
    compactor: Optional[Compactor] = None,
    tracer=None,
    max_windows: int = 6,
    max_steps_per_window: int = 6,
    hook: bool = True,
) -> RunResult:
    """带 hook 的外循环: 跨窗口续跑, 直到 goal_met 或窗口用尽。

    hook=False 时关掉 early-stop 拦截 (对照组): 模型一收工就结束, 长任务多半失败。
    """
    res = RunResult(success=False)

    for w in range(max_windows):
        if goal_met(store.load()):
            res.success = True
            break

        messages = _seed_messages(goal, store)
        rec = WindowRecord(index=w)
        win_span = tracer.start(f"window-{w}", "window") if tracer else None
        aborted = False

        for _ in range(max_steps_per_window):
            rec.steps += 1
            res.total_steps += 1
            res.context_tokens_total += total_tokens(messages)   # 成本: 本步喂入的上下文

            # 一步 reasoning = 一个 span; 其中的 tool 调用 = child span
            r_span = tracer.start(f"reason@w{w}.s{rec.steps}", "reasoning") if tracer else None
            text_parts, tool_calls, stop = [], [], False
            for ch in provider.stream(messages, tools.specs()):
                if ch.kind == "text":
                    text_parts.append(ch.text)
                elif ch.kind == "tool_call":
                    tool_calls.append((ch.tool_name, ch.tool_args))
                elif ch.kind == "done":
                    stop = ch.stop
            messages.append({"role": "assistant", "content": "".join(text_parts)})

            for name, args in tool_calls:
                if tracer:
                    with tracer.span(f"tool:{name}", "tool"):
                        out = _mutate(store, name, args, tools)
                else:
                    out = _mutate(store, name, args, tools)
                messages.append({"role": "user", "content": out, "kind": "tool_result"})

            if tracer:
                tracer.end(r_span)

            # 窗口预算 → compaction
            rec.tokens_peak = max(rec.tokens_peak, total_tokens(messages))
            if compactor:
                messages, events = compactor.compact(messages)
                rec.compactions += len(events)

            # hook: 模型想收工?
            if stop:
                if goal_met(store.load()):
                    res.success = True
                elif hook:
                    rec.stop_intercepted = True   # ★ 拦截 early-stop, 触发换窗续跑
                else:
                    res.aborted_early = True       # 对照组: 不拦截, 整个 run 中断
                    aborted = True
                break

        if tracer and win_span:
            tracer.end(win_span)
        res.windows.append(rec)
        if res.success or aborted:
            break

    res.final_state = store.load()
    return res


def _mutate(store: FileStateStore, name: str, args: dict, tools: ToolRegistry) -> str:
    """读 state → 派发工具(可改 state) → 写回。把'文件系统即真相'落实。"""
    state = store.load()
    out = tools.dispatch(name, args, state)
    store.save(state)
    return out


# ----------------------------------------------------------------- 演示装配 (供 N2/测试复用)
def demo_setup(tmp_dir: str | Path, total_steps: int = 6, early_stop_at: int = 2):
    """构造一个'需要 total_steps 步、但模型会在第 early_stop_at 步过早收工'的确定性场景。

    返回 (provider, goal, tools, store, goal_met)。用来演示 loop-with-hook 如何救回 early-stop。
    """
    from provider import MockProvider, Turn

    store = FileStateStore(tmp_dir)
    st = store.load()
    st["goal_total"] = total_steps
    store.save(st)

    tools = ToolRegistry()

    def do_step(args, state):
        state["progress"] = state.get("progress", 0) + 1
        state.setdefault("notes", []).append(f"完成第 {state['progress']} 步")
        # 模拟一次"大 tool 输出"(如文件读取/搜索结果): 让窗口内上下文真的累积,
        # 这样开/关 compaction 才会在成本上拉开差距 (小任务上看不出差别)。
        bulky = "x" * 1200   # ~300 tokens 的工具输出
        return f"已完成第 {state['progress']}/{total_steps} 步。工具原始输出: {bulky}"
    tools.register("do_step", do_step)

    # 模型脚本: 每回合调 do_step; 在 early_stop_at 处错误地 stop (制造 early-stopping 病),
    # 之后每回合继续 do_step, 真正完成时才 stop。
    script = []
    for i in range(1, total_steps + 3):
        stop = (i == early_stop_at) or (i >= total_steps)
        script.append(Turn(text=f"推进第 {i} 步。", tool_calls=[{"name": "do_step", "args": {}}], stop=stop))
    provider = MockProvider(script=script)

    def goal_met(state: dict) -> bool:
        return state.get("progress", 0) >= total_steps

    return provider, f"完成 {total_steps} 个子步骤", tools, store, goal_met
