"""引用忠实度核查：每句话回查"被引文献是否真支持它的论断"。

对比两种检查，看清 9.4 的核心区分：
- `existence_only`：只问"引的 id 在不在库"——这是 9.2 守住的那一关（防幻觉引用）。
- `check_sentence`：问"被引文献到底支不支持这句的 claim"——**忠实度**，严格更难。
一个引用可以**存在却不忠实**：引了真论文，但那论文压根没说这事。
"""
from __future__ import annotations

from .corpus import BY_ID


def check_sentence(sentence, by_id=BY_ID) -> str:
    """faithful / unfaithful / dangling / uncited。"""
    if sentence.cited_doc is None:
        return "uncited"
    doc = by_id.get(sentence.cited_doc)
    if doc is None:
        return "dangling"      # 引了不存在的 id（9.2 的存在性失败）
    return "faithful" if sentence.claim_tokens <= doc.supports else "unfaithful"


def existence_only(sentence, by_id=BY_ID) -> bool:
    """naive 检查：只看被引 id 在不在库——抓不到"存在但不忠实"。"""
    return sentence.cited_doc in by_id


def audit(report, by_id=BY_ID) -> dict:
    verdicts = [(s, check_sentence(s, by_id)) for s in report.sentences]
    unfaithful = [s for s, v in verdicts if v == "unfaithful"]
    faithful = [s for s, v in verdicts if v == "faithful"]
    dangling = [s for s, v in verdicts if v == "dangling"]
    exist_pass = sum(1 for s in report.sentences if existence_only(s, by_id))
    return {
        "verdicts": verdicts,
        "total": len(report.sentences),
        "faithful": len(faithful),
        "unfaithful": unfaithful,
        "dangling": dangling,
        "existence_pass": exist_pass,   # naive 检查"通过"的句数
    }
