"""OS Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）+ 打分/
抽题工具。

与 software-engineering-mastery/design-patterns-mastery/database-mastery/networking-mastery 同名
文件是同一套已验证设计的独立副本（各 mastery track 自成一体，不跨 track 相互 import）。
DeepPoint/ScenarioPoint 都带 `explain` 字段：因为操作系统对用户是完全没系统学过的领域，每个
知识点先给一段讲仔细的系统性讲解（是什么/为什么/怎么用/常见误区），再接三层追问链或场景判断，
兼顾"系统学会"和"面试接得住"两个目标。这是老手要求的五个CS基础专题(软件工程/设计模式/数据库/
网络/OS)队列的最后一个。
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
            trigger="你说进程和线程不一样，具体区别在哪？",
            explain="进程（Process）是操作系统进行资源分配的基本单位，每个进程拥有独立的地址空间"
                    "（虚拟内存、代码段、堆栈）；线程（Thread）是操作系统进行调度的基本单位，同一个"
                    "进程内的多个线程共享该进程的地址空间，但每个线程有自己独立的栈和寄存器状态。"
                    "最常见的误区是以为线程和进程只是'轻量级'和'重量级'的区别，而没有理解到线程之所以"
                    "轻量，本质原因是它不需要像进程切换那样重新加载整个地址空间（页表），只需要切换"
                    "寄存器和栈指针，这才是线程切换开销远小于进程切换的根本原因。",
            chain=(
                ("进程和线程各自是什么单位？", "进程是操作系统进行资源分配的基本单位线程是操作系统进行调度的基本单位", ("进程是操作系统进行资源分配的基本单位", "线程是操作系统进行调度的基本单位")),
                ("为什么线程切换比进程切换快？", "线程切换不需要重新加载整个地址空间的页表只需要切换寄存器和栈指针", ("不需要重新加载整个地址空间的页表", "只需要切换寄存器和栈指针")),
                ("同一进程内的多个线程共享什么，各自独立拥有什么？", "共享该进程的地址空间但每个线程有自己独立的栈和寄存器状态", ("共享该进程的地址空间", "每个线程有自己独立的栈和寄存器状态")),
            ),
            pitfall="很多人只会说'线程比进程轻量'，答不出轻量的根本原因是不需要切换页表。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["进程是操作系统进行资源分配的基本单位，线程是操作系统进行调度的基本单位", "不需要重新加载整个地址空间的页表，只需要切换寄存器和栈指针"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果线上服务CPU使用率突然飙升到100%，你会怎么排查？",
            explain="CPU飙升排查的常规思路是先定位是哪个进程占用了CPU（top/htop），再定位是进程里"
                    "哪个线程占用了CPU（top -H或者jstack这类工具），最后结合线程堆栈判断是死循环、"
                    "锁竞争还是GC频繁等具体原因。",
            rubric=("先定位是哪个进程占用CPU", "再定位进程里哪个线程占用CPU", "结合线程堆栈判断具体原因"),
            trap="只会说'看CPU占用'，说不清具体怎么从进程级别下钻到线程级别再到代码级别",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "先定位是哪个进程占用CPU，再定位进程里哪个线程占用CPU")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
