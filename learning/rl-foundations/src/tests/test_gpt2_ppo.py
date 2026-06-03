"""GPT-2 PPO 单元测试 — 仅验证基础组件能跑（不做完整训练）.

由于 LLM-PPO 单 iter 30s+，这里只测：
1. build_token_rewards 维度与 mask 正确
2. SentimentReward 能加载且对正负样本区分
3. get_log_probs 输出 shape 正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_SRC = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_SRC))


def test_build_token_rewards_shape():
    from ppo_gpt2_minimal import build_token_rewards

    B, T = 4, 10
    raw_rewards = torch.tensor([1.0, 0.5, -0.5, 2.0])
    response_mask = torch.zeros(B, T)
    response_mask[:, 5:] = 1.0
    response_mask[3, 7:] = 0.0     # 第 4 个样本 response 短一些
    log_p_act = torch.randn(B, T)
    log_p_ref = torch.randn(B, T)

    rewards = build_token_rewards(raw_rewards, response_mask, log_p_act,
                                  log_p_ref, beta=0.1)
    assert rewards.shape == (B, T)
    # prompt 段（mask=0）reward 应为 0
    assert torch.all(rewards[:, :5] == 0)


def test_build_token_rewards_end_only():
    """raw_rewards 仅加在 response 末端 token 位置上。"""
    from ppo_gpt2_minimal import build_token_rewards

    B, T = 2, 6
    raw_rewards = torch.tensor([10.0, 20.0])
    response_mask = torch.zeros(B, T)
    response_mask[0, 2:5] = 1   # response 长度 3
    response_mask[1, 1:4] = 1   # response 长度 3
    log_p_act = torch.zeros(B, T)
    log_p_ref = torch.zeros(B, T)

    rewards = build_token_rewards(raw_rewards, response_mask, log_p_act,
                                  log_p_ref, beta=0.0)
    # 末端 token 位 (b=0, t=4) 应有 10.0
    assert rewards[0, 4] == 10.0
    assert rewards[0, 2] == 0.0  # 非末端
    assert rewards[1, 3] == 20.0
    assert rewards[1, 1] == 0.0


@pytest.mark.slow
def test_sentiment_reward_distinguishes():
    """SentimentReward 应能区分正面 vs 负面文本。

    标记为 slow：本测试需下载 BERT 模型。
    """
    from sentiment_reward import SentimentReward
    rm = SentimentReward(device="cpu")
    pos = "This movie was absolutely amazing, I loved it!"
    neg = "This movie was a complete waste of time, terrible."
    scores = rm.score([pos, neg])
    assert scores[0] > scores[1] + 0.3, scores


def test_get_log_probs_shape():
    """get_log_probs 输出 shape (B, T-1)."""
    from ppo_gpt2_minimal import get_log_probs
    import torch.nn as nn

    class FakeLM(nn.Module):
        """伪造 LM，避免下载真实 GPT-2。"""
        def __init__(self, vocab=50, hidden=8) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids, attention_mask=None):
            h = self.emb(input_ids)
            logits = self.head(h)
            return type("Out", (), {"logits": logits})()

    model = FakeLM(vocab=20, hidden=4)
    input_ids = torch.randint(0, 20, (2, 5))
    attn = torch.ones_like(input_ids)
    log_p = get_log_probs(model, input_ids, attn)
    assert log_p.shape == (2, 4), log_p.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "not slow"]))
