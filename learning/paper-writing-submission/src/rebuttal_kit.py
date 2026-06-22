"""
rebuttal_kit.py — 把一组审稿意见分类、定优先级, 生成结构化 rebuttal 骨架.

为什么需要它: 收到审稿意见 (尤其有 negative 的) 时, 新手容易情绪化、逐条辩驳、或漏掉关键点。
高效的 rebuttal 是**系统工程**: 先把每条意见分类 (它要的是什么), 再按"对录用影响"定优先级,
在有限字数里优先回应能翻盘的点。这个工具把这套流程代码化。

意见分类 (决定回应策略):
  - factual_error   审稿人说错了 (基于误读) → 礼貌澄清 + 指向论文位置
  - clarification   审稿人没看懂 → 解释 + 承诺改写
  - add_experiment  要补实验 → 能补则补(rebuttal 期最有力), 不能补则说明
  - weakness        真实弱点 → 承认 + 缓解 + 说明影响有限
  - scope           说超范围/要更多 → 划定贡献边界, 礼貌拒绝扩张
  - positive        正面评价 → 致谢, 不浪费字数

纯 stdlib (基于关键词的轻量规则分类; 真实使用时人工复核)。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 分类 → (关键词信号, 回应策略, 对录用影响权重)
CATEGORIES = {
    "factual_error": (["incorrect", "wrong", "错误", "misread", "实际上", "其实"],
                      "礼貌澄清 + 指向论文具体位置/数据", 3),
    "add_experiment": (["experiment", "baseline", "ablation", "实验", "补", "compare", "missing",
                        "should also", "评测", "benchmark"],
                       "能补就补(rebuttal 期补实验最有力); 不能补则说明计划", 3),
    "clarification": (["unclear", "confusing", "not clear", "看不懂", "解释", "clarify", "definition",
                       "notation", "符号"],
                      "解释 + 承诺正文改写, 引用具体段落", 2),
    "weakness": (["weak", "limitation", "concern", "弱", "不足", "problem", "issue", "局限"],
                 "承认 + 缓解措施 + 说明对主张影响有限", 2),
    "scope": (["out of scope", "beyond", "超出", "范围", "more general", "也应该", "为什么不"],
              "划定贡献边界, 礼貌指出属 future work", 1),
    "positive": (["good", "novel", "interesting", "strong", "well", "好", "新颖", "扎实", "clear"],
                 "简短致谢, 不耗字数", 0),
}


def classify(comment: str) -> str:
    """把一条审稿意见归类. 取信号词命中最多的类; 都没命中归为 clarification (保守)."""
    text = comment.lower()
    best, best_hits = "clarification", 0
    for cat, (kws, _, _) in CATEGORIES.items():
        hits = sum(1 for k in kws if k.lower() in text)
        if hits > best_hits:
            best, best_hits = cat, hits
    return best


def triage(comments: list[str]) -> list[dict]:
    """对一组意见分类 + 标策略 + 标优先级 (权重高的先回), 返回排序后的列表."""
    out = []
    for c in comments:
        cat = classify(c)
        _, strategy, weight = CATEGORIES[cat]
        out.append({"comment": c, "category": cat, "strategy": strategy, "priority": weight})
    out.sort(key=lambda x: x["priority"], reverse=True)
    return out


def budget_words(triaged: list[dict], total_words: int = 500) -> list[dict]:
    """按优先级把有限字数分配给各条 (positive 几乎不给字数). 返回带 word_budget 的列表."""
    weight_sum = sum(max(t["priority"], 0.2) for t in triaged) or 1
    for t in triaged:
        share = max(t["priority"], 0.2) / weight_sum
        t["word_budget"] = round(total_words * share)
    return triaged


def build_skeleton(comments: list[str], total_words: int = 500) -> str:
    """生成一份结构化 rebuttal 骨架 (按优先级排序, 每条配策略 + 字数预算)."""
    triaged = budget_words(triage(comments), total_words)
    lines = ["# Rebuttal 骨架 (auto)\n",
             "> 原则: 先谢评审; 按对录用影响排序优先回应; 能补实验是 rebuttal 期最强武器;",
             "> 礼貌、就事论事、对每条都回应(哪怕一句)。绝不情绪化、不无视、不空泛。\n",
             "感谢各位审稿人的细致意见。下面逐条回应 (R=Reviewer)。\n"]
    cat_label = {"factual_error": "澄清误读", "add_experiment": "补充实验",
                 "clarification": "澄清说明", "weakness": "承认并缓解",
                 "scope": "范围界定", "positive": "致谢"}
    for i, t in enumerate(triaged, 1):
        lines.append(f"## [{i}] [{cat_label[t['category']]}] (≈{t['word_budget']} 词)")
        lines.append(f"> 意见: {t['comment']}")
        lines.append(f"> 策略: {t['strategy']}")
        lines.append("> 回应: ____________\n")
    return "\n".join(lines)


if __name__ == "__main__":
    comments = [
        "The baseline DPO seems under-tuned; you should compare against a properly tuned DPO.",
        "It is unclear how the robust weighting is computed. The notation in Eq.3 is confusing.",
        "This is an interesting and well-motivated paper.",
        "The method only tested on 7B; why not also show it scales to 70B? Feels out of scope otherwise.",
        "Claim of robustness is a real concern given only one noise type is studied.",
    ]
    for t in triage(comments):
        print(f"[P{t['priority']}] {t['category']:14} | {t['comment'][:50]}...")
    print("\n--- 骨架预览 ---")
    print(build_skeleton(comments)[:500])
