"""Quantization evaluation helpers — PPL / acc / memory."""
from __future__ import annotations

from typing import Callable, List
import math
import torch


def eval_ppl_mock(predict_logits: Callable[[List[int]], torch.Tensor], data: List[List[int]]) -> float:
    """PPL with mock predictor; just to keep the API tidy."""
    nll_total = 0.0
    n_tok = 0
    for seq in data:
        for t in range(1, len(seq)):
            logits = predict_logits(seq[:t])
            logp = torch.log_softmax(logits, dim=-1)
            nll_total += -logp[seq[t]].item()
            n_tok += 1
    return math.exp(nll_total / max(n_tok, 1))


def memory_table(rows: List[dict]) -> str:
    """Format a markdown comparison table."""
    head = "| variant | PPL | acc | mem(GB) | tok/s |\n|---|---|---|---|---|"
    body = "\n".join(
        f"| {r['variant']} | {r['ppl']} | {r['acc']} | {r['mem_gb']} | {r['tok_s']} |"
        for r in rows
    )
    return head + "\n" + body
