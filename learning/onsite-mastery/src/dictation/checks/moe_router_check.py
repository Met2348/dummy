"""moe_router_spec 的纯断言检验：不 import solutions/。用 torch 做结构性 + 数值检验。"""
from __future__ import annotations

import torch


def check(target) -> None:
    torch.manual_seed(0)
    num_experts = 4
    k = 2
    t_len = num_experts  # 4 个 token，配合下面的循环移位构造，让路由恰好均匀

    # ---- 构造"均匀路由"场景：把同一个有明显区分度的 logits 行，循环移位 T 次 ----
    # 因为是完整一圈的循环移位，无论每一行 softmax 出来的分布长什么样，
    # T 行取平均之后每个专家拿到的平均概率 P_i 必然相等（旋转对称）；
    # 同理，每个 token 的 top-k 选择也是同一个选择模式的循环移位，
    # 使得每个专家恰好在 k 个 token 的 top-k 里出现——两者合起来构成一个
    # "整体均匀"的路由场景，用来验证 aux_loss 理论最小值。
    base = torch.tensor([5.0, 3.0, 1.0, -1.0])
    logits = torch.stack([torch.roll(base, shifts=t) for t in range(t_len)])  # (T, N)

    topk_idx, topk_weight, aux_loss = target(logits, k, num_experts)

    # ---- 结构检查 1: 每个 token 恰好选中 k 个互不相同、合法范围内的专家 ----
    assert topk_idx.shape == (t_len, k), f"topk_idx 形状应为 (T,k)=({t_len},{k})，实得 {tuple(topk_idx.shape)}"
    for t in range(t_len):
        idxs = topk_idx[t].tolist()
        assert len(set(idxs)) == k, f"token {t} 选中的专家有重复: {idxs}"
        assert all(0 <= i < num_experts for i in idxs), f"token {t} 专家下标越界: {idxs}"

    # ---- 结构检查 2: topk_weight 在每个 token 内部应重新归一化到和为 1 ----
    weight_sums = topk_weight.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(t_len), atol=1e-4), (
        f"每个 token 的 top-k 门控权重应该重新归一化到和为 1，实得 {weight_sums.tolist()}"
    )
    assert torch.all(topk_weight >= 0), "门控权重不应为负"

    # ---- 数值检查 1: 均匀路由场景下 aux_loss 应约等于理论最小值 1.0(标准形式) ----
    aux_val = float(aux_loss)
    assert abs(aux_val - 1.0) < 1e-3, (
        f"均匀路由(每个专家的分配比例和平均路由概率都均匀)时 aux_loss 理论最小值应为 1.0，"
        f"实得 {aux_val:.6f}（是不是 f_i 忘记除以 k，或者 P_i 用了 top-k 而不是完整 softmax？）"
    )

    # ---- 数值检查 2: 偏斜路由(几乎所有 token 都挤到同一个专家) aux_loss 应显著更高 ----
    skew_logits = torch.zeros(t_len, num_experts)
    skew_logits[:, 0] = 20.0  # softmax 之后 expert-0 独占几乎全部概率质量
    _, _, aux_loss_skew = target(skew_logits, k, num_experts)
    aux_skew_val = float(aux_loss_skew)
    assert aux_skew_val > aux_val * 1.5, (
        f"偏斜路由的 aux_loss({aux_skew_val:.4f}) 应显著高于均匀路由的 aux_loss({aux_val:.4f})"
    )
