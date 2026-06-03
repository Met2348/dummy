"""Generative RM (GenRM) — RM 用 CoT 推理打分.

idea: 不直接 head 出 scalar，而是让 LLM 生成"为什么是好/坏"再打分.
Pipeline:
    1. prompt: "Question: ... Answer: ... Critique step-by-step and rate 1-10"
    2. generate critique + score
    3. score 提取作为 reward
"""
from __future__ import annotations

import re

import torch


GENRM_PROMPT = """You are evaluating an answer to a math question.

Question: {question}
Answer: {answer}

First, critique the answer step by step:
- Is the reasoning correct?
- Are there errors?
- Is the final answer right?

Then give a score from 1 to 10. End with "Score: N".
"""


def parse_genrm_score(text: str) -> float | None:
    """从 generation 中提 'Score: N'，归一化到 [0, 1]."""
    m = re.search(r"Score:\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    score = float(m.group(1))
    return min(max(score / 10.0, 0.0), 1.0)


def genrm_score(
    judge_llm,
    question: str,
    answer: str,
    n_samples: int = 1,
) -> float:
    """用 LLM 当 judge，多次采样取均值更稳."""
    prompt = GENRM_PROMPT.format(question=question, answer=answer)
    scores = []
    for _ in range(n_samples):
        out = judge_llm(prompt)
        s = parse_genrm_score(out)
        if s is not None:
            scores.append(s)
    return sum(scores) / max(len(scores), 1)


def compare_scalar_vs_genrm(scalar_rm_fn, genrm_judge_fn, samples: list[tuple[str, str]]):
    print(f"{'Q (head)':25s} {'A (head)':25s} {'scalar':8s} {'genrm':8s}")
    print("-" * 80)
    for q, a in samples:
        s = scalar_rm_fn(q, a)
        g = genrm_score(genrm_judge_fn, q, a)
        print(f"{q[:22]:25s} {a[:22]:25s} {s:.3f}    {g:.3f}")


if __name__ == "__main__":
    print("GenRM minimal — mock\n" + "=" * 50)
    def mock_scalar_rm(q, a):
        return 0.6 if "7" in a else 0.3

    def mock_judge(prompt):
        if "16-3-6=7" in prompt or "7" in prompt.split("Answer:")[-1][:15]:
            return "The reasoning is correct. Score: 9"
        return "Wrong reasoning. Score: 2"

    samples = [
        ("Q: 16-3-6=?", "A: 7"),
        ("Q: 16-3-6=?", "A: I think it's 5"),
        ("Q: 5*8=?", "A: 40"),
    ]
    compare_scalar_vs_genrm(mock_scalar_rm, mock_judge, samples)
    print("\nGenRM 优势:")
    print("  ✓ 可解释 (输出 critique)")
    print("  ✓ 利用 LLM 推理能力 (强 base 强 RM)")
    print("  ✗ 推理慢 ~10x")
    print("  ✗ judge 模型本身需要训")
