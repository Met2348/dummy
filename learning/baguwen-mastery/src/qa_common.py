"""八股问答卡片的单一数据源：QA dataclass + categories()/quiz()/grade()。

所有 ai_qa/*.py、backend_qa/*.py 模块从这里导入，不各自重复定义，
避免 23 个文件里出现 23 份几乎一样但可能悄悄不一致的 grade() 实现。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QA:
    id: str                      # 模块内唯一，全局也唯一（前缀区分，如 "ai-opt-01" / "be-os-03"）
    cat: str                     # 类别中文名
    q: str                       # 问题
    a: str                       # 标准答案
    keys: tuple[str, ...]        # 采分关键词，供 grade() 打分
    follow_ups: tuple[str, ...]  # 面试官常见的第二、三问（只列问题，不展开答案）


def categories(bank: list[QA]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for qa in bank:
        if qa.cat not in seen:
            seen.add(qa.cat)
            out.append(qa.cat)
    return out


def quiz(bank: list[QA], cat: str | None = None, n: int | None = None) -> list[QA]:
    """确定性抽题：给 cat 则只取该类，n 截断前 n 道。"""
    pool = [qa for qa in bank if cat is None or qa.cat == cat]
    return pool[:n] if n else pool


def grade(answer: str, qa: QA) -> float:
    """按采分关键词命中率打分（0..1）。命中即答案里出现该关键词（大小写不敏感）。"""
    low = answer.lower()
    hit = sum(1 for kw in qa.keys if kw.lower() in low)
    return hit / len(qa.keys)


def _self_test() -> None:
    sample = QA(
        id="test-01", cat="测试类", q="示例问题？", a="示例答案包含关键词A和关键词B。",
        keys=("关键词A", "关键词B"), follow_ups=("追问1？",),
    )
    bank = [sample]
    assert categories(bank) == ["测试类"]
    assert quiz(bank, cat="测试类") == [sample]
    assert quiz(bank, cat="不存在的类") == []
    assert grade(sample.a, sample) == 1.0
    assert grade("", sample) == 0.0
    assert grade("只含关键词A", sample) == 0.5
    print("[PASS] qa_common: QA数据结构 + categories/quiz/grade 三件套")


if __name__ == "__main__":
    _self_test()
