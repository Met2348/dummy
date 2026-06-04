"""RULER 评测（NVIDIA 2024）— NIAH 升级版.

教学版：实现 4 个子任务的题目生成器（不真跑模型）.
- single-NIAH (S-NIAH)
- multi-key NIAH (MK-NIAH)
- multi-value NIAH (MV-NIAH)
- variable tracking (VT)
"""
from __future__ import annotations

import random
from typing import Iterable


HAYSTACK_BASE = ("The quick brown fox jumps over the lazy dog. "
                 "Lorem ipsum dolor sit amet consectetur adipiscing elit. ")


def _make_haystack(target_length: int) -> str:
    base = HAYSTACK_BASE
    s = (base * (target_length // len(base) + 1))[:target_length]
    return s


def s_niah(target_length: int, depth_pct: float) -> dict:
    """Single NIAH：1 个 needle，1 个 key，1 个 value."""
    code = str(random.randint(10000, 99999))
    needle = f"The magic number is {code}."
    hay = _make_haystack(target_length)
    pos = int(depth_pct * target_length / 100)
    text = hay[:pos] + " " + needle + " " + hay[pos:]
    q = f"{text}\n\nQuestion: What is the magic number?"
    return {"prompt": q, "answer": code, "task": "s-niah"}


def mk_niah(target_length: int, n_keys: int = 4) -> dict:
    """Multi-Key NIAH：多个 needle (key=A,B,C,D)，问其中一个."""
    keys = ["alpha", "beta", "gamma", "delta", "epsilon"][:n_keys]
    codes = {k: str(random.randint(1000, 9999)) for k in keys}
    target_key = random.choice(keys)

    needles = [f"The {k}-code is {v}." for k, v in codes.items()]
    hay = _make_haystack(target_length)
    pieces = [hay]
    for needle in needles:
        pos = random.randint(0, len(hay))
        pieces.insert(random.randint(0, len(pieces)), needle)
    text = " ".join(pieces)
    q = f"{text}\n\nQuestion: What is the {target_key}-code?"
    return {"prompt": q, "answer": codes[target_key], "task": "mk-niah"}


def mv_niah(target_length: int, n_values: int = 3) -> dict:
    """Multi-Value NIAH：一个 key 对应多个 value，问 list."""
    key = "lucky-number"
    values = [str(random.randint(100, 999)) for _ in range(n_values)]
    needles = [f"One {key} is {v}." for v in values]
    hay = _make_haystack(target_length)
    pieces = [hay]
    for needle in needles:
        pieces.insert(random.randint(0, len(pieces)), needle)
    text = " ".join(pieces)
    q = f"{text}\n\nQuestion: List all the {key} values."
    return {"prompt": q, "answer": values, "task": "mv-niah"}


def variable_tracking(target_length: int, n_hops: int = 3) -> dict:
    """变量追踪：x=5, y=x, z=y → 问 z."""
    val = str(random.randint(100, 999))
    chain = ["x", "y", "z", "w", "v"][:n_hops + 1]
    sentences = [f"Let {chain[0]} = {val}."]
    for i in range(1, len(chain)):
        sentences.append(f"Let {chain[i]} = {chain[i - 1]}.")
    hay = _make_haystack(target_length)
    pieces = [hay]
    for s in sentences:
        pieces.insert(random.randint(0, len(pieces)), s)
    text = " ".join(pieces)
    q = f"{text}\n\nQuestion: What is the value of {chain[-1]}?"
    return {"prompt": q, "answer": val, "task": "variable-tracking"}


def check_answer(predicted: str, expected) -> bool:
    """str-in 或 list-all-in."""
    if isinstance(expected, list):
        return all(v in predicted for v in expected)
    return str(expected) in predicted


def ruler_grid(context_lengths: Iterable[int],
               n_per_task: int = 10) -> list:
    """生成 RULER 子集（4 个子任务 × 多长度 × n 样本）."""
    queries = []
    for L in context_lengths:
        for _ in range(n_per_task):
            queries.append(s_niah(L, depth_pct=50))
            queries.append(mk_niah(L))
            queries.append(mv_niah(L))
            queries.append(variable_tracking(L))
    return queries


if __name__ == "__main__":
    qs = ruler_grid([500, 1000], n_per_task=2)
    print(f"Generated {len(qs)} RULER queries")
    tasks = {}
    for q in qs:
        tasks[q["task"]] = tasks.get(q["task"], 0) + 1
    print(f"Distribution: {tasks}")
    print(f"Sample s-niah answer: {qs[0]['answer']}")
    print(f"Sample mv-niah answers: {qs[2]['answer']}")
