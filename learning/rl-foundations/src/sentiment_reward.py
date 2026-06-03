"""BERT-sentiment 当 reward model.

教学：用现成的 SST-2 finetuned BERT 给 GPT-2 生成的文本打"正向情感"分。

运行（先 smoke）:
    python learning/rl-foundations/src/sentiment_reward.py
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


class SentimentReward:
    """用 distilbert SST-2 当 reward。"""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).to(dtype)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, texts: list[str]) -> torch.Tensor:
        """返回每条文本的"positive" 概率 (B,)."""
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128,
        ).to(self.device)
        logits = self.model(**enc).logits      # (B, 2)
        probs = F.softmax(logits, dim=-1)
        # labels: 0=neg, 1=pos
        return probs[:, 1].cpu()


def _smoke():
    print("Loading SST-2 RM...")
    rm = SentimentReward(device="cpu")
    samples = [
        "This movie was absolutely wonderful, I loved every minute.",
        "What a complete waste of time, terrible acting.",
        "It was an okay movie I guess, nothing special.",
        "Best film I've seen this year!",
    ]
    s = rm.score(samples)
    for txt, score in zip(samples, s.tolist()):
        print(f"  {score:.3f}   {txt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="跑 4 句话测试")
    args = p.parse_args()
    if args.smoke or True:    # 默认就跑 smoke
        _smoke()


if __name__ == "__main__":
    main()
