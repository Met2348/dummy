"""
review_kit.py — 结构化同行评审辅助: 一份 rubric, 把"审别人"和"审自己"用同一套维度打通.

为什么需要它: 给别人审稿是博士的义务, 也是**反哺自己写作的最佳训练** —— 站到评判者那一边,
你会突然看清自己论文的破绽。但新手审稿常犯两个错: ① 评审不结构化 (写几句感想就打分);
② 太苛刻或太宽松 (没有一致的标准)。这个工具给你一份固定 rubric, 让评审**结构化、可复用、
建设性**, 并能把同一套维度调转枪口审你自己的工作 (接 9.3 攻击清单 / 9.7 narrative_audit)。

评审维度 (复用 9.3 的找-洞视角):
  - summary        先复述论文 (证明你读懂了, 也是建设性评审的前提)
  - soundness      方法/实验是否站得住 (公平 baseline? 多种子? 9.4/9.5)
  - novelty        贡献是否新 (相对 9.2 领域地图)
  - clarity        是否讲清楚 (9.6 图 / 9.7 写作)
  - reproducibility 能否复现 (9.5)
  每维度: 评分 + 具体证据 + 建设性建议 (不是"this is bad", 而是"X 不清楚, 建议补 Y")

纯 stdlib。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 评审维度: key -> (名称, 该问什么, 对应的前序专题)
DIMENSIONS = {
    "soundness": ("方法/实验扎实度", "baseline 公平吗? 多种子报方差吗? 消融充分吗?", "9.4/9.5"),
    "novelty": ("新颖性/贡献", "相对已有工作真的新吗? 贡献是否被夸大?", "9.2/9.3"),
    "clarity": ("清晰度", "讲清楚了吗? 图/写作是否到位?", "9.6/9.7"),
    "reproducibility": ("可复现性", "代码/超参/数据齐吗? 能重现吗?", "9.5"),
    "significance": ("重要性", "解决的问题重要吗? 谁会在意?", "9.3"),
}


def blank_review() -> dict:
    """返回一份空白结构化评审表."""
    return {
        "summary": "",  # 先复述论文 (建设性评审的前提)
        "dimensions": {k: {"score": None, "evidence": "", "suggestion": ""} for k in DIMENSIONS},
        "overall_score": None,  # 1-5 或 1-10, 看会议
        "confidence": None,     # 1-5, 你对评审的把握
    }


def constructiveness_check(review: dict) -> dict:
    """检查这份评审是否建设性: 有没有复述、每个负面是否带可操作建议、是否有证据."""
    issues = []
    if not review.get("summary", "").strip():
        issues.append("缺 summary —— 先复述论文证明你读懂了, 这是建设性评审的前提")
    for k, d in review["dimensions"].items():
        name = DIMENSIONS[k][0]
        if d.get("score") is not None and d["score"] <= 2:  # 给了低分
            if not d.get("suggestion", "").strip():
                issues.append(f"「{name}」给了低分却没给改进建议 —— 评审要可操作, 不是只批评")
            if not d.get("evidence", "").strip():
                issues.append(f"「{name}」低分缺具体证据 —— 指出'哪里', 不是泛泛'不好'")
    return {"issues": issues, "constructive": not issues}


def flip_to_self(review_dims: list[str] | None = None) -> list[str]:
    """把审稿维度调转枪口: 返回'用这些维度审视自己工作'的自检问句 (审别人→反哺自己)."""
    dims = review_dims or list(DIMENSIONS)
    return [f"[{DIMENSIONS[k][0]}] {DIMENSIONS[k][1]} (对照我的工作, 诚实回答 / 见 {DIMENSIONS[k][2]})"
            for k in dims]


def render(review: dict) -> str:
    lines = ["=== 结构化评审 ===",
             f"Summary: {review['summary'] or '(待填)'}\n"]
    for k, (name, q, ref) in DIMENSIONS.items():
        d = review["dimensions"][k]
        lines.append(f"[{name}] 分={d['score']}  ({ref})")
        if d["evidence"]:
            lines.append(f"   证据: {d['evidence']}")
        if d["suggestion"]:
            lines.append(f"   建议: {d['suggestion']}")
    lines.append(f"\n总分: {review['overall_score']}  置信度: {review['confidence']}")
    return "\n".join(lines)


if __name__ == "__main__":
    r = blank_review()
    r["summary"] = "提出 Robust-DPO, 在噪声偏好下提升对齐 win-rate。"
    r["dimensions"]["soundness"] = {"score": 4, "evidence": "8 种子, 公平 baseline",
                                    "suggestion": "建议补不同噪声类型"}
    r["dimensions"]["novelty"] = {"score": 2, "evidence": "", "suggestion": ""}  # 低分但没证据/建议
    r["overall_score"] = 3; r["confidence"] = 4
    print(render(r))
    print("\n--- 建设性自检 ---")
    chk = constructiveness_check(r)
    for i in chk["issues"]:
        print("⚠", i)
    print("\n--- 审别人 → 反哺自己 (调转枪口) ---")
    for q in flip_to_self()[:3]:
        print(" ?", q)
