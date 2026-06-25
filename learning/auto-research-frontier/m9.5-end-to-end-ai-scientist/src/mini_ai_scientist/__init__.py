"""mini_ai_scientist —— 端到端 AI Scientist 的诚实缩小版（M9.5 教学模块）。

把 The AI Scientist 的五阶段（ideation → experiment → analysis → writeup → review）
做成一个**在 3080Ti 上真跑小实验**的闭环：指标来自真实训练，不是 mock 分数。

刻意焊进两个批判教学点：
- ideation-execution gap：新点子可能真的不涨点 → 诚实返回"未支持"。
- grading-its-own-homework：自动评审可被 game → 不可全信。

子模块：
    experiment  真训练（torch MLP on make_moons），确定性可复现
    ideation    生成研究 idea（模板化默认，可插真 LLM）
    analysis    对照比较 + 判定 verdict + 画图
    writeup     自动写 1 页报告（数字源自真实指标，不硬编码）
    review      mock 自动评审（故意可被刷，教 grading-own-homework）
    pipeline    串起五阶段
"""
from __future__ import annotations

__all__ = ["experiment", "ideation", "analysis", "writeup", "review", "pipeline"]
