"""
harness_eval.py — L10: 评测 harness 本身。

关键事实: 固定同一个模型, 只换 harness, SWE-bench 能差几十分。所以"agent 强不强"很大程度是
"harness 强不强"。本模块把同一个长任务, 在不同 **harness 配置**下各跑一遍, 对照成功率与成本,
让"harness 是自变量"这件事可被测量。

两个旋钮 (来自 long_horizon):
  - compaction: 开/关 5 阶段压缩 → 影响上下文成本 (token)
  - hook:       开/关 early-stop 拦截 → 影响长任务成功率
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from compaction import Compactor
from long_horizon import run_long_horizon, demo_setup


@dataclass
class HarnessConfig:
    name: str
    compaction: bool = True
    window_budget: int = 600       # compaction 触发的 token 预算
    max_windows: int = 6
    hook: bool = True

    def describe(self) -> str:
        return f"compaction={'on' if self.compaction else 'off'}, hook={'on' if self.hook else 'off'}"


def run_with_config(cfg: HarnessConfig, work_dir: str | Path,
                    total_steps: int = 6, early_stop_at: int = 2) -> dict:
    """用一个干净任务实例跑某配置, 返回一行结果。"""
    run_dir = Path(work_dir) / cfg.name.replace(" ", "_")
    provider, goal, tools, store, goal_met = demo_setup(
        run_dir, total_steps=total_steps, early_stop_at=early_stop_at)
    compactor = Compactor(max_tokens=cfg.window_budget) if cfg.compaction else None

    res = run_long_horizon(
        provider, goal, tools, store, goal_met,
        compactor=compactor, max_windows=cfg.max_windows, hook=cfg.hook,
    )
    return {
        "harness": cfg.name,
        "config": cfg.describe(),
        "success": res.success,
        "windows": res.n_windows,
        "steps": res.total_steps,
        "context_tokens": res.context_tokens_total,
        "aborted_early": res.aborted_early,
    }


def evaluate(configs: list[HarnessConfig], work_dir: str | Path,
             total_steps: int = 6, early_stop_at: int = 2) -> list[dict]:
    """跑一组配置, 返回结果列表 (notebook 里转 pandas 对照)。"""
    return [run_with_config(c, work_dir, total_steps, early_stop_at) for c in configs]


def default_configs() -> list[HarnessConfig]:
    """三个有教学意义的对照配置。"""
    return [
        HarnessConfig("A_naive",          compaction=False, hook=False),  # 玩具: 没 hook 没压缩
        HarnessConfig("B_hook_only",      compaction=False, hook=True),   # 救回长任务, 但成本高
        HarnessConfig("C_hook_compaction", compaction=True, hook=True),   # 生产级: 成功且省
    ]
