"""Batched sampling with temperature / top-k / top-p / penalty.

Designed to match vLLM's `Sampler` behaviour at the math level so a learner
can compare sampled token distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class SamplerConfig:
    temperature: float = 1.0
    top_k: int = 0       # 0 = disable
    top_p: float = 1.0   # 1.0 = disable
    min_p: float = 0.0
    repetition_penalty: float = 1.0


def apply_repetition_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    """Decay logits of tokens already produced (presence-style)."""
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 0, prev_tokens)
    score = torch.where(score > 0, score / penalty, score * penalty)
    out = logits.clone()
    out.scatter_(0, prev_tokens, score)
    return out


def top_k_top_p_mask(logits: torch.Tensor, top_k: int, top_p: float, min_p: float) -> torch.Tensor:
    """Return a boolean mask of tokens KEPT (True = candidate)."""
    keep = torch.ones_like(logits, dtype=torch.bool)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values.min()
        keep &= logits >= kth
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprob = torch.softmax(sorted_logits, dim=0).cumsum(0)
        cutoff = (cumprob - cumprob[0]) < top_p   # always keep top-1
        sorted_keep = cutoff | (torch.arange(len(logits)) == 0)
        nucleus_mask = torch.zeros_like(keep)
        nucleus_mask[sorted_idx] = sorted_keep
        keep &= nucleus_mask
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=0)
        keep &= probs >= min_p * probs.max()
    return keep


def sample_one(logits: torch.Tensor, prev_tokens: torch.Tensor, cfg: SamplerConfig) -> int:
    logits = apply_repetition_penalty(logits, prev_tokens, cfg.repetition_penalty)
    if cfg.temperature <= 0:
        return int(logits.argmax().item())
    logits = logits / cfg.temperature
    mask = top_k_top_p_mask(logits, cfg.top_k, cfg.top_p, cfg.min_p)
    logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=0)
    tok = torch.multinomial(probs, 1).item()
    return int(tok)


def sample_batch(logits: torch.Tensor, prev_tokens: List[torch.Tensor], cfgs: List[SamplerConfig]) -> List[int]:
    out = []
    for i, cfg in enumerate(cfgs):
        out.append(sample_one(logits[i], prev_tokens[i], cfg))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    V = 100
    logits = torch.randn(V)
    cfg = SamplerConfig(temperature=0.7, top_k=10, top_p=0.9)
    print("sampled:", sample_one(logits, torch.tensor([0, 1, 2]), cfg))
