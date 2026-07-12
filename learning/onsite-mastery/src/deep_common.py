"""终面深水区共享数据结构：DeepPoint（追问链）+ 打分/抽题工具。

与 baguwen-mastery 的 QA(+follow_ups) 的关键区别：follow_ups 只列问题，逼你现场应对；
DeepPoint.chain 每一层都带参考答案+采分词，逼你对照自查"有没有真接住"，而不是自我感觉良好。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepPoint:
    id: str
    cat: str
    trigger: str
    # chain: 每层 = (追问题, 参考答案, 采分关键词)，层数递进加深（>=3层），最后一层常是诚实的开放问题/局限性
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
    """对 chain 每一层分别打分：答案里命中该层采分关键词的比例（不区分大小写子串匹配）。
    answers 长度可以短于 chain（用户没答到那么深），未作答的层记 0.0。
    """
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
    """确定性抽题（不用 random，保证可复现）：按 bank 原始顺序过滤+截取。"""
    pool = [dp for dp in bank if cat is None or dp.cat == cat]
    return pool if n is None else pool[:n]


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
            real_world_link="learning/dpo-family",
        )
    ]
    assert len(sample[0].chain) >= 3
    assert categories(sample) == ["test"]
    scores = grade_chain(sample[0], ["我做了消融实验对比显存", "会欠拟合因为子空间不够"])
    assert scores[0] == 1.0, scores
    assert scores[1] == 1.0, scores
    assert scores[2] == 0.0, scores
    assert drill(sample, cat="test", n=1) == sample
    print("[PASS] deep_common: DeepPoint/grade_chain/drill 自检通过")


if __name__ == "__main__":
    _self_test()
