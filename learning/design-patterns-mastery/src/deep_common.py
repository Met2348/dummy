"""Design Patterns Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）
+ 打分/抽题工具。

与 software-engineering-mastery/onsite-mastery/foundation-model-mastery/federated-learning-mastery/
diffusion-mastery 同名文件是同一套已验证设计的独立副本（各 mastery track 自成一体，不跨 track 相互
import）。DeepPoint/ScenarioPoint 都带 `explain` 字段：因为设计模式对用户是完全没系统学过的领域，
每个知识点先给一段讲仔细的系统性讲解（是什么/为什么/怎么用/常见误区），再接三层追问链或场景判断，
兼顾"系统学会"和"面试接得住"两个目标。
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
            trigger="你提到用了观察者模式，具体怎么避免内存泄漏？",
            explain="观察者模式（Observer）定义了一种一对多的依赖关系，当一个对象（Subject）的状态发生变化"
                    "时，所有依赖它的观察者（Observer）都会自动收到通知并更新。它的核心动机是解耦事件的"
                    "发布方和订阅方，发布方不需要知道具体有哪些订阅方存在。最常见的误区是只关注注册"
                    "（subscribe）逻辑，忽略了取消注册（unsubscribe）——如果观察者的生命周期短于被观察者，"
                    "而观察者又没有主动取消订阅，被观察者会一直持有观察者的引用，导致观察者对象无法被垃圾"
                    "回收，造成内存泄漏。",
            chain=(
                ("观察者不取消订阅会有什么后果？", "被观察者持有观察者引用导致观察者无法被垃圾回收造成内存泄漏", ("被观察者持有观察者引用", "无法被垃圾回收", "内存泄漏")),
                ("怎么从设计上规避这个问题？", "用弱引用持有观察者列表，或者约定观察者销毁时必须显式取消订阅", ("弱引用", "显式取消订阅")),
                ("弱引用方案有什么隐藏代价？", "弱引用可能在通知发生前就被垃圾回收导致通知丢失，需要权衡及时性和安全性", ("弱引用可能在通知发生前就被垃圾回收", "权衡及时性和安全性")),
            ),
            pitfall="很多人只会说'观察者模式就是发布订阅'，答不出不取消订阅会导致内存泄漏这个具体机制。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["被观察者持有观察者引用导致观察者无法被垃圾回收造成内存泄漏", "用弱引用配合显式取消订阅"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果一个类里到处都是针对同一个变量的if-else分支决定行为，你会怎么重构？",
            explain="这是典型的可以用策略模式（Strategy）替代的信号：当一段代码里出现大量针对同一个变量的"
                    "if-else或switch分支，且每个分支对应一种独立的算法/行为时，说明这些行为应该被抽取成"
                    "独立的策略类，通过多态替代条件分支，新增一种行为只需要新增一个策略类而不用修改已有"
                    "代码，符合开闭原则。",
            rubric=("识别出条件分支对应独立算法", "抽取为策略接口和具体策略类", "用多态替代if-else分支"),
            trap="只会说'用设计模式重构'，说不清具体是哪个模式、为什么适用、重构后新增行为怎么做",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "识别出条件分支对应独立算法，抽取为策略接口和具体策略类，用多态替代if-else分支")
    assert sp_score == 1.0, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
