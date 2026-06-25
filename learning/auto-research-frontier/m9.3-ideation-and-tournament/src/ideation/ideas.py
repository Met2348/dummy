"""点子库：每个 idea = 一段宣传文案 + 一组真实训练配置 + 来源（self/other）。

- `text`：评委只看得到这个（含"novel/exotic/fancy"等热词，决定它的 novelty 评分）。
- `config`：真训练配置，决定它的真实 feasibility（评委**看不到**，要执行才知道）。
- `source`：'self'=评委所属模型自己生成的点子（用来演示自偏好）；'other'=人/别的来源。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Idea:
    id: str
    text: str
    config: dict
    source: str   # "self" | "other"


IDEA_BANK = (
    # —— 朴素但真有效（other 来源、文案平淡、feasibility 高）——
    Idea("more-steps", "多训练一会儿，把步数加上去。",
         {"steps": 2000}, source="other"),
    Idea("momentum", "给梯度下降加个动量项。",
         {"momentum": 0.9, "steps": 600}, source="other"),
    # —— 中庸对照（other、平淡、feasibility 中）——
    Idea("tiny-lr", "保守一点，用很小的学习率慢慢走。",
         {"lr": 0.01, "steps": 200}, source="other"),
    # —— 花哨但真没用（self 来源、文案高新颖、feasibility 低）——
    Idea("exotic-init", "用一种新颖的 exotic 重参数化大尺度初始化，突破常规。",
         {"init_scale": 150.0, "steps": 60}, source="self"),
    Idea("fancy-sched", "上一个 cutting-edge 的全新自适应学习率调度（训练中放大 lr）。",
         {"lr_grows": True, "lr": 2.0}, source="self"),
    Idea("bold-lr", "大胆激进地把学习率拉到很大，加速收敛。",
         {"lr": 50.0}, source="self"),
)


def get_idea(idea_id: str) -> Idea:
    for it in IDEA_BANK:
        if it.id == idea_id:
            return it
    raise KeyError(idea_id)
