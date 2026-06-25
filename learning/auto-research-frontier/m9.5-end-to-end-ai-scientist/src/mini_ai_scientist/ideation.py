"""阶段 1：创意 / 假设生成。

默认是**模板化** idea 库（确定性、无需联网/API），但留了 `llm` 插口——
后续可换成真 LLM 生成 idea。每个 idea 自带一个**幼稚的自评 novelty**，
故意和真实实验结果脱钩，好在最后演示 ideation-execution gap
（"看着新颖/合理" ≠ "真能涨点"）。

每个 idea = 一个假设 + 如何实例化 baseline vs treatment（在 ExperimentConfig 上做 override）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Idea:
    id: str
    title: str
    hypothesis: str
    baseline: Dict          # ExperimentConfig 字段 override
    treatment: Dict
    naive_rationale: str
    self_novelty: float     # idea agent 的自评（0–1），故意不可靠


# 模板 idea 库：刻意覆盖"真阳/真阴/无定论"光谱，用于教 ideation-execution gap
IDEA_BANK: List[Idea] = [
    Idea(
        id="add-depth",
        title="给线性模型加一个隐藏层",
        hypothesis="在非线性任务上，加一个隐藏层能显著提升测试精度。",
        baseline={"depth": 0},
        treatment={"depth": 1},
        naive_rationale="moons 非线性可分，线性模型欠拟合，加非线性应有用。",
        self_novelty=0.30,   # 自评"不新"，但实际是真阳——novelty 与结果脱钩的反例
    ),
    Idea(
        id="go-deeper",
        title="1 层不够，3 层更好",
        hypothesis="把隐藏层从 1 加到 3 还能进一步提升精度。",
        baseline={"depth": 1},
        treatment={"depth": 3},
        naive_rationale="更深=更强表达力，应该更好。",
        self_novelty=0.55,
    ),
    Idea(
        id="crank-lr",
        title="把学习率调大就训得更好",
        hypothesis="学习率从 0.05 提到 2.0 会让模型更快更好地收敛。",
        baseline={"lr": 0.05},
        treatment={"lr": 2.0},
        naive_rationale="大学习率=快收敛，听起来很合理。",
        self_novelty=0.40,   # 真阴：过大 lr 反而更差
    ),
    Idea(
        id="widen",
        title="加宽隐藏层提升容量",
        hypothesis="隐藏维度从 8 加到 64 能提升精度。",
        baseline={"hidden_dim": 8},
        treatment={"hidden_dim": 64},
        naive_rationale="更宽=更大容量。",
        self_novelty=0.50,
    ),
    Idea(
        id="relu-vs-tanh",
        title="ReLU 优于 Tanh",
        hypothesis="把激活从 Tanh 换成 ReLU 能提升精度。",
        baseline={"activation": "tanh"},
        treatment={"activation": "relu"},
        naive_rationale="ReLU 是现代默认，应该更好。",
        self_novelty=0.45,
    ),
]


def generate_ideas(
    k: Optional[int] = None,
    llm: Optional[Callable[[str], List[Idea]]] = None,
    topic: str = "small MLP on a nonlinear 2D classification task",
) -> List[Idea]:
    """返回研究 idea。

    默认走模板库（确定性）。传入 `llm`（一个 prompt->ideas 的可调用）即可换成
    真 LLM 生成——这是把本模块接到真模型的插口。
    """
    if llm is not None:                       # 插口：真 LLM ideation
        ideas = llm(topic)
    else:
        ideas = list(IDEA_BANK)
    return ideas[:k] if k else ideas


def get_idea(idea_id: str) -> Idea:
    for idea in IDEA_BANK:
        if idea.id == idea_id:
            return idea
    raise KeyError(f"unknown idea id: {idea_id!r}; have {[i.id for i in IDEA_BANK]}")
