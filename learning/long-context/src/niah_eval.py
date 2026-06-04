"""NIAH (Needle in a Haystack) 评测.

教学版：构造测试用例 + 字符串匹配评估（不真跑模型）。
"""
from __future__ import annotations

import random
from typing import Iterable


NEEDLE_TEMPLATES = [
    "The secret password is {code}.",
    "Janet's lucky number is {code}.",
    "The hidden code is {code}.",
]


def make_haystack(target_length: int, base: str = None) -> str:
    base = base or ("The quick brown fox jumps over the lazy dog. " * 20)
    haystack = (base * (target_length // len(base) + 1))[:target_length]
    return haystack


def make_niah_query(target_length: int, depth_pct: float,
                    needle_code: str | None = None) -> tuple[str, str]:
    """返回 (full_query, expected_answer)."""
    needle_code = needle_code or str(random.randint(1000, 9999))
    template = random.choice(NEEDLE_TEMPLATES)
    needle = template.format(code=needle_code)
    haystack = make_haystack(target_length)
    insert_at = int(depth_pct * target_length / 100)
    text = haystack[:insert_at] + " " + needle + " " + haystack[insert_at:]
    query = (f"{text}\n\nQuestion: What's the special code mentioned in "
             f"the text above?")
    return query, needle_code


def check_answer(answer: str, expected_code: str) -> bool:
    return expected_code in answer


def niah_grid(context_lengths: Iterable[int],
              depths: Iterable[float],
              n_samples: int = 3) -> dict:
    """生成 NIAH 测试集（不评测，只生成）."""
    queries = []
    for L in context_lengths:
        for d in depths:
            for i in range(n_samples):
                q, expected = make_niah_query(L, d)
                queries.append({"length": L, "depth": d,
                                "query_len": len(q),
                                "expected": expected})
    return queries


if __name__ == "__main__":
    queries = niah_grid([200, 500], [25, 50, 75], n_samples=2)
    print(f"Generated {len(queries)} NIAH queries")
    for q in queries[:3]:
        print(f"  L={q['length']} depth={q['depth']}% expected={q['expected']}")
