"""长 ctx 数据打包 + document-level attention mask.

教学版：纯 PyTorch，演示 pack / mask / 课程切换.
"""
from __future__ import annotations

import torch
from typing import Iterable


def pack_documents(docs: list, max_len: int = 8192) -> list:
    """First-fit decreasing 打包.

    返回 list[list[doc]]，每个内层 list 拼接长度 ≤ max_len.
    """
    chunks = []
    for d in docs:
        if len(d) <= max_len:
            chunks.append(d)
            continue
        for start in range(0, len(d), max_len):
            chunks.append(d[start:start + max_len])
    sorted_docs = sorted(chunks, key=lambda d: -len(d))
    batches = []
    for d in sorted_docs:
        placed = False
        for batch in batches:
            if sum(len(x) for x in batch) + len(d) <= max_len:
                batch.append(d)
                placed = True
                break
        if not placed:
            batches.append([d])
    return batches


def concat_with_lens(batch: list) -> tuple:
    """concat tokens 并返回 (tokens, doc_lens)."""
    tokens = []
    lens = []
    for d in batch:
        tokens.extend(d)
        lens.append(len(d))
    return torch.tensor(tokens, dtype=torch.long), lens


def make_doc_mask(doc_lens: list, total_len: int = None) -> torch.Tensor:
    """生成 block-diagonal mask.

    mask[i, j] = True 仅当 i, j 属于同一 doc.
    """
    total = total_len or sum(doc_lens)
    mask = torch.zeros(total, total, dtype=torch.bool)
    start = 0
    for L in doc_lens:
        end = start + L
        mask[start:end, start:end] = True
        start = end
    return mask


def curriculum_lengths(stage: int) -> tuple:
    """阶段 → (min_len, max_len)."""
    stages = {
        1: (256, 2048),
        2: (2048, 8192),
        3: (8192, 32768),
        4: (32768, 131072),
    }
    return stages.get(stage, (256, 2048))


def filter_by_curriculum(docs: list, stage: int) -> list:
    """按当前 stage 过滤 doc 长度."""
    lo, hi = curriculum_lengths(stage)
    return [d for d in docs if lo <= len(d) <= hi]


def packing_efficiency(batches: list, max_len: int) -> float:
    """计算 packing 效率 = 实际 token / (n_batch × max_len)."""
    total = sum(sum(len(d) for d in b) for b in batches)
    return total / (len(batches) * max_len)


if __name__ == "__main__":
    docs = [
        list(range(100)),
        list(range(500)),
        list(range(2000)),
        list(range(3000)),
        list(range(1500)),
        list(range(800)),
    ]

    batches = pack_documents(docs, max_len=4000)
    print(f"Packed {len(docs)} docs into {len(batches)} batches")
    for i, b in enumerate(batches):
        lens = [len(d) for d in b]
        print(f"  batch {i}: lens={lens}, total={sum(lens)}")

    eff = packing_efficiency(batches, max_len=4000)
    print(f"Packing efficiency: {eff:.2%}")

    tokens, doc_lens = concat_with_lens(batches[0])
    mask = make_doc_mask(doc_lens)
    print(f"\nBatch 0: tokens.shape={tokens.shape}, mask.shape={mask.shape}")
    print(f"Doc lens: {doc_lens}")

    print("\nCurriculum stages:")
    for s in range(1, 5):
        print(f"  stage {s}: {curriculum_lengths(s)}")
