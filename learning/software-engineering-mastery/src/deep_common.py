"""Software Engineering Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）
+ 打分/抽题工具。

与 onsite-mastery/foundation-model-mastery/federated-learning-mastery/diffusion-mastery 同名文件是
同一套已验证设计的独立副本（各 mastery track 自成一体，不跨 track 相互 import）。这次相对之前三个
track 新增一个 `explain` 字段：因为软件工程/设计模式/数据库/网络/OS 这五个专题是用户完全没有系统学过
的领域（不像 FM/FL/Diffusion 那样已有研究生级基础只需整理成追问链），所以每个知识点先给一段讲仔细的
系统性讲解（是什么/为什么/怎么用/常见误区），再接 DeepPoint.chain 三层追问链或 ScenarioPoint.rubric
场景判断，兼顾"系统学会"和"面试接得住"两个目标。
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
            trigger="你提到用了单例模式，具体怎么保证线程安全？",
            explain="单例模式（Singleton）保证一个类在整个程序运行期间只有一个实例，并提供一个全局访问点。"
                    "它的核心动机是控制某些资源（如数据库连接池、配置管理器）的唯一性，避免重复创建带来的"
                    "资源浪费或状态不一致。最常见的误区是只在单线程环境下测试通过就认为实现正确，实际上"
                    "在多线程环境下，如果不加同步控制，多个线程可能同时判断实例不存在从而各自创建一份，"
                    "破坏了'唯一实例'这个核心约束。",
            chain=(
                ("最简单的线程安全写法是什么？", "在获取实例的方法上加synchronized关键字，每次调用都加锁", ("synchronized", "每次调用都加锁")),
                ("这种写法有什么性能问题？", "每次调用都要竞争锁，即使实例已经创建好了也要付出加锁开销", ("每次调用都要竞争锁", "已经创建好了也要付出加锁开销")),
                ("怎么优化这个性能问题？", "双重检查锁定，只在实例为空时才加锁，配合volatile防止指令重排序", ("双重检查锁定", "volatile", "指令重排序")),
            ),
            pitfall="很多人只会说'加个锁就行'，答不出双重检查锁定为什么还需要volatile来防止指令重排序。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["加synchronized每次调用都加锁", "每次调用都要竞争锁，即使已经创建好了也要付出加锁开销"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果发现一个类到处都在被if-else判断类型来决定行为，你会怎么重构？",
            explain="这是典型的可以用策略模式（Strategy）替代的信号：当一段代码里出现大量针对同一个变量的"
                    "if-else或switch分支，且每个分支对应一种独立的算法/行为时，说明这些行为应该被抽取成"
                    "独立的策略类，通过多态替代条件分支，这样新增一种行为只需要新增一个策略类而不用修改"
                    "已有代码，符合开闭原则。",
            rubric=("识别出条件分支对应独立算法", "抽取为策略接口和具体策略类", "用多态替代if-else分支"),
            trap="只会说'用设计模式重构'，说不清具体是哪个模式、为什么适用、重构后新增行为怎么做",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "识别出条件分支对应独立算法，抽取为策略接口和具体策略类")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
