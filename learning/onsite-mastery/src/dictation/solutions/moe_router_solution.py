"""moe_router_spec 的参考实现。用 torch，Mixtral 风格的 top-k 重归一化 + Switch 风格 aux loss。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def moe_topk_router_with_aux_loss(logits: torch.Tensor, k: int, num_experts: int):
    # 完整 softmax：这是算 P_i（平均路由概率）和 topk 权重来源都要用到的全量分布
    probs = F.softmax(logits, dim=-1)  # (T, N)

    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)  # (T, k) 各
    # Mixtral 风格：只在被选中的 k 个专家里重新归一化，让权重和为 1
    topk_weight = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    # f_i: 专家 i 被选中的频率，分母是 T*k（不是 T），保证 sum_i f_i == 1
    expert_mask = F.one_hot(topk_idx, num_classes=num_experts).float()  # (T, k, N)
    f = expert_mask.mean(dim=(0, 1))  # (N,)  sum(f) == 1

    # P_i: 完整 softmax 概率在全部 T 个 token 上的平均，sum(P) == 1
    p = probs.mean(dim=0)  # (N,)

    aux_loss = num_experts * (f * p).sum()

    return topk_idx, topk_weight, aux_loss
