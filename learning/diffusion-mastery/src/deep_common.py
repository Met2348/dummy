"""Diffusion Mastery 共享数据结构：DeepPoint（追问链）+ ScenarioPoint（场景判断）+ 打分/抽题工具。

与 onsite-mastery/foundation-model-mastery/federated-learning-mastery 同名文件是同一套已验证
设计的独立副本（各 mastery track 自成一体，不跨 track 相互 import），DeepPoint.chain 每层带
参考答案+采分词，ScenarioPoint.rubric 是"没有唯一正确答案"场景题的要点覆盖率打分。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepPoint:
    id: str
    cat: str
    trigger: str
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
            trigger="你提到用了 LoRA，具体 rank 怎么选的？",
            chain=(
                ("rank 怎么选的？", "从 8 起步做消融，兼顾显存和拟合能力", ("消融", "显存")),
                ("如果 rank 选太小会怎样？", "欠拟合，任务复杂度超过低秩子空间容量", ("欠拟合", "子空间")),
                ("有没有理论依据支撑这个观察？", "intrinsic dimension 假说，但样本量小时结论不稳", ("intrinsic", "样本量")),
            ),
            pitfall="很多人第2层就只会说'试出来的'，接不住'为什么'",
            real_world_link="",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["我做了消融实验对比显存", "会欠拟合因为子空间不够"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert scores[2] == 0.0, scores
    assert drill(sample, cat="test", n=1) == sample

    sp_sample = [
        ScenarioPoint(
            id="sc-test-01",
            cat="test",
            trigger="如果上线后效果变差，你怎么定位？",
            rubric=("对比线上线下分布差异", "灰度回滚止损", "分层排查数据/模型/服务"),
            trap="只会说'看日志'，说不清具体先看哪一层、怎么止损",
            real_world_link="",
        )
    ]
    assert categories(sp_sample) == ["test"]
    sp_score = grade_scenario(sp_sample[0], "先对比线上线下分布差异，同时灰度回滚止损")
    assert sp_score == 2 / 3, sp_score
    assert grade_scenario(sp_sample[0], "") == 0.0
    print("[PASS] deep_common: DeepPoint/grade_chain/drill/ScenarioPoint/grade_scenario 自检通过")


if __name__ == "__main__":
    _self_test()
