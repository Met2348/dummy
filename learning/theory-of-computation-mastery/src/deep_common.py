"""Discrete Math Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）
+ 打分/抽题工具。

这是老师指出"和科班CS本科相比还缺哪些能力"之后新开的六个专题(离散数学/计算理论/算法理论证明向/
计算机体系结构/编译原理/安全密码学基础)之一，与 software-engineering-mastery 等既有五个 CS 基础
专题同名文件是同一套已验证设计的独立副本（各 mastery track 自成一体，不跨 track 相互 import）。
沿用 `explain` 字段设计：每个知识点先给一段讲仔细的系统性讲解（是什么/为什么/怎么用/常见误区），
再接 DeepPoint.chain 三层追问链或 ScenarioPoint.rubric 场景判断，兼顾"系统学会"和"面试接得住"。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepPoint:
    id: str
    cat: str
    trigger: str
    explain: str
    chain: tuple[tuple[str, str, tuple[str, ...]], ...]
    pitfall: str
    real_world_link: str = ""


def categories(bank: list[DeepPoint]) -> list[str]:
    seen: list[str] = []
    for dp in bank:
        if dp.cat not in seen:
            seen.append(dp.cat)
    return seen


def grade_chain(dp: DeepPoint, answers: list[str]) -> list[float]:
    scores: list[float] = []
    for i, (_question, _ref, keys) in enumerate(dp.chain):
        if i >= len(answers) or not keys:
            scores.append(0.0)
            continue
        ans = answers[i].lower()
        hit = sum(1 for k in keys if k.lower() in ans)
        scores.append(hit / len(keys))
    return scores


def drill(bank: list[DeepPoint], cat: str | None = None, n: int | None = None) -> list[DeepPoint]:
    pool = [dp for dp in bank if cat is None or dp.cat == cat]
    return pool if n is None else pool[:n]


@dataclass(frozen=True)
class ScenarioPoint:
    id: str
    cat: str
    trigger: str
    explain: str
    rubric: tuple[str, ...]
    trap: str
    real_world_link: str = ""


def grade_scenario(sp: ScenarioPoint, answer: str) -> float:
    if not sp.rubric:
        return 0.0
    ans = answer.lower()
    hit = sum(1 for k in sp.rubric if k.lower() in ans)
    return hit / len(sp.rubric)


def _self_test() -> None:
    sample = [
        DeepPoint(
            id="dp-test-01",
            cat="test",
            trigger="你说鸽笼原理，具体怎么用它证明一个存在性结论？",
            explain="鸽笼原理（Pigeonhole Principle）说的是：如果把n+1个物品放进n个容器，至少有一个"
                    "容器装了2个以上的物品。它的核心价值不在于这个陈述本身有多复杂，而在于它能把很多"
                    "看似需要构造性证明的存在性问题，转化成一个简单的计数论证——不需要具体指出是哪个"
                    "容器装多了，只需要证明容器数量少于物品数量。常见误区是只会在物品和容器都摆在眼前"
                    "的场景里应用它，遇到需要自己构造'容器'划分方式的题目就想不到用鸽笼原理。",
            chain=(
                ("鸽笼原理最基本的陈述是什么？", "n+1个物品放进n个容器，至少有一个容器装了2个以上的物品", ("n+1个物品放进n个容器", "至少有一个容器装了2个以上的物品")),
                ("它的价值主要体现在解决哪类问题上？", "把存在性问题转化成计数论证，不需要具体构造出是哪个容器装多了", ("把存在性问题转化成计数论证", "不需要具体构造出是哪个容器装多了")),
                ("很多人在什么情况下想不到用它？", "遇到需要自己构造容器划分方式的题目就想不到用鸽笼原理", ("需要自己构造容器划分方式的题目", "想不到用鸽笼原理")),
            ),
            pitfall="很多人只会背诵鸽笼原理的陈述，遇到需要自己设计'容器'划分方式的存在性证明题目就"
                    "想不到套用这个工具。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["n+1个物品放进n个容器，至少有一个容器装了2个以上的物品", "把存在性问题转化成计数论证，不需要具体构造出是哪个容器装多了"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果一个证明里反复出现'显然''容易验证'却没有给出具体推导，你会怎么判断这是不是有效证明？",
            explain="这是典型的证明完整性问题：'显然'和'容易验证'经常被用来掩盖作者自己也没有仔细"
                    "验证、或者需要额外条件才成立的步骤。判断一个证明是否有效，需要检查每一步跳跃"
                    "是否真的可以由前面的条件机械地推出，而不是依赖读者的'直觉认同'。",
            rubric=("识别出'显然'掩盖的跳跃步骤", "要求补全被省略的中间推导", "检查是否隐含额外未声明的假设"),
            trap="只会说'这个证明有问题'，说不清具体是哪一步的逻辑跳跃、需要补充什么条件才能让论证"
                 "站得住脚。",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "识别出'显然'掩盖的跳跃步骤，要求补全被省略的中间推导")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
