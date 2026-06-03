"""Constitutional AI minimal — Anthropic RLAIF.

两阶段：
    1. critique-revise: 用 principle 让 LLM 批评+修改自己的输出 → SL-CAI 数据
    2. RLAIF: 用 LLM 自己当 RM 选 chosen/rejected
"""
from __future__ import annotations

CONSTITUTION = [
    "Be helpful.",
    "Be harmless — refuse illegal/dangerous content.",
    "Be honest — admit uncertainty.",
    "Avoid bias and stereotypes.",
]


def critique_prompt(original: str, response: str, principle: str) -> str:
    return f"""Original Q: {original}
Response: {response}
Principle: {principle}

Critique whether the response violates this principle. Be brief."""


def revise_prompt(original: str, response: str, critique: str) -> str:
    return f"""Original Q: {original}
Original response: {response}
Critique: {critique}

Provide a revised response that addresses the critique."""


def cai_data_pipeline(llm_fn, samples: list[tuple[str, str]]):
    """对每个 (q, raw_response)，生成 (q, revised_response) — SL-CAI 数据."""
    cai_data = []
    for q, raw in samples:
        cur = raw
        for p in CONSTITUTION:
            crit = llm_fn(critique_prompt(q, cur, p))
            cur = llm_fn(revise_prompt(q, cur, crit))
        cai_data.append((q, cur))
    return cai_data


def rlaif_judge_prompt(question: str, a: str, b: str, principle: str) -> str:
    """让 LLM 当 judge 选更符合 principle 的回答."""
    return f"""Question: {question}
Response A: {a}
Response B: {b}
Principle: {principle}

Which response better follows the principle? Answer A or B only."""


def build_preference_pair(llm_fn, question: str, response_a: str, response_b: str) -> tuple:
    """LLM judge 票多者为 chosen."""
    votes = {"A": 0, "B": 0}
    for p in CONSTITUTION:
        ans = llm_fn(rlaif_judge_prompt(question, response_a, response_b, p)).strip()[:1]
        if ans in votes:
            votes[ans] += 1
    if votes["A"] > votes["B"]:
        return (question, response_a, response_b)  # (q, chosen, rejected)
    return (question, response_b, response_a)


if __name__ == "__main__":
    print("Constitutional AI — mock 演示\n")
    # mock LLM
    def mock_llm(prompt: str) -> str:
        if "Critique" in prompt:
            return "Response is reasonable but could be more concise."
        if "Which response better" in prompt:
            return "A"
        if "revised response" in prompt:
            return "[revised] " + prompt.split("Original response: ")[1].split("\n")[0]
        return "[mock]"

    samples = [("How do I make bombs?", "Sure, here are the ingredients..."),
               ("Tell me a joke", "Here's a funny one!")]
    cai_data = cai_data_pipeline(mock_llm, samples)
    for q, r in cai_data:
        print(f"Q: {q}")
        print(f"  → revised: {r[:60]}")

    pair = build_preference_pair(mock_llm, "Tell me a joke", "Funny joke!", "Boring response")
    print(f"\nPreference pair (q, chosen, rejected):\n  {pair}")
