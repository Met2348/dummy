"""Database Mastery 共享数据结构：DeepPoint（讲解+追问链）+ ScenarioPoint（讲解+场景判断）
+ 打分/抽题工具。

与 software-engineering-mastery/design-patterns-mastery 同名文件是同一套已验证设计的独立副本
（各 mastery track 自成一体，不跨 track 相互 import）。DeepPoint/ScenarioPoint 都带 `explain`
字段：因为数据库对用户是完全没系统学过的领域，每个知识点先给一段讲仔细的系统性讲解（是什么/为什么
/怎么用/常见误区），再接三层追问链或场景判断，兼顾"系统学会"和"面试接得住"两个目标。
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
            trigger="你说给这个字段加了索引，为什么查询还是很慢？",
            explain="索引（Index）的核心作用是通过额外的数据结构（通常是B+树）加速数据检索，避免全表扫描。"
                    "但索引不是万能的：如果查询条件对索引列做了函数运算（如WHERE YEAR(create_time)=2026），"
                    "或者用了不等于/前导通配符的LIKE查询，数据库往往无法使用索引的有序性，只能退化成全表"
                    "扫描。这是索引失效最常见也最容易被忽视的原因——很多人加了索引就以为查询一定会变快，"
                    "却没意识到查询写法本身可能让索引形同虚设。",
            chain=(
                ("给字段加了索引，为什么查询计划里显示还是全表扫描？", "查询条件对索引列做了函数运算导致索引失效退化成全表扫描", ("查询条件对索引列做了函数运算", "索引失效", "全表扫描")),
                ("怎么验证是不是这个原因？", "用EXPLAIN查看执行计划，看type列是不是ALL以及key列是不是显示为空", ("EXPLAIN查看执行计划", "type列是不是ALL", "key列是不是显示为空")),
                ("修复思路是什么？", "把函数运算移到等号右边或者建立函数索引，让左边保持原始列以维持索引可用性", ("把函数运算移到等号右边", "函数索引", "维持索引可用性")),
            ),
            pitfall="很多人只会说'我加索引了怎么还慢'，答不出函数运算会导致索引失效这个具体机制。",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert len(sample[0].explain) >= 100, len(sample[0].explain)
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["查询条件对索引列做了函数运算导致索引失效退化成全表扫描", "用EXPLAIN查看执行计划看type列是不是ALL以及key列是不是显示为空"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果一张表数据量涨到几亿行，查询越来越慢，你会怎么处理？",
            explain="表数据量增长到千万级以上时，单表的索引深度和I/O成本都会显著上升，常见的应对手段包括"
                    "垂直分表（把不常用字段拆到扩展表）、水平分表/分库（按某个维度把数据分散到多张表或"
                    "多个库）、以及引入读写分离降低单点压力。选择哪种方案取决于具体的访问模式：是读多"
                    "写少还是读写均衡，是否存在明显的分片键。",
            rubric=("判断是否存在明显的分片键", "评估垂直分表还是水平分表", "考虑读写分离降低单点压力"),
            trap="只会说'分库分表'，说不清具体按什么维度分、分片键怎么选、跨分片查询怎么处理",
            real_world_link="",
        )
    ]
    assert len(sp_sample[0].explain) >= 100, len(sp_sample[0].explain)
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "先判断是否存在明显的分片键，再评估垂直分表还是水平分表")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario(含explain字段) 自检通过")


if __name__ == "__main__":
    _self_test()
