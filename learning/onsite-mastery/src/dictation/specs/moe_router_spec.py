"""MoE Top-K 路由 + 负载均衡辅助损失（Switch Transformer / Mixtral 风格），闭卷从零手写。

面试高频度 ****。MoE 岗位/infra 岗位手撕重灾区，经常和"为什么需要 aux loss"、
"路由坍缩(routing collapse)是什么、怎么发现"连续追问。用 torch 写。

接口约定
--------
    moe_topk_router_with_aux_loss(logits, k, num_experts)
        -> (topk_idx, topk_weight, aux_loss)

    logits      : torch.Tensor, 形状 (T, num_experts)，T 是 token 数，未经 softmax 的原始路由 logits
    k           : 每个 token 激活的专家数（Switch Transformer 用 k=1，Mixtral 用 k=2）
    num_experts : 专家总数 N

    返回:
        topk_idx    : LongTensor (T, k)，每个 token 选中的专家下标
        topk_weight : FloatTensor (T, k)，选中专家的门控权重，**在 k 个专家内重新归一化到和为 1**
                      （Mixtral 的做法：只在被选中的 top-k 里做 softmax 归一化，不用完整 softmax
                      的原始概率值，否则 k 个权重加起来达不到 1，FFN 输出的尺度会不对）
        aux_loss    : 标量 tensor，Switch Transformer 风格的负载均衡辅助损失

公式（负载均衡辅助损失，Fedus et al. 2021 式 4-5，推广到任意 k）
-------------------------------------------------------------------
    P_i = 完整 softmax(logits) 在专家 i 上的平均概率（对全部 T 个 token 取平均，
          **不是**只在被选中的 top-k 上算）——衡量路由器"整体倾向"给专家 i 分配多少概率质量
    f_i = 专家 i 被选中的频率 = (token 数 × k 个槽位中，落在专家 i 上的次数) / (T * k)
          ——注意分母是 T*k 不是 T，这样 sum_i f_i = 1，和 sum_i P_i = 1 对齐，
          这也是让"均匀路由时 aux_loss 恰好等于理论最小值 1.0"这条性质在**任意 k**
          下都成立的关键（很多人 k>1 时忘记除以 k，导致均匀路由时 aux_loss 变成 k
          而不是 1.0）
    aux_loss = num_experts * sum_i (f_i * P_i)

why: f 和 P 都是长度 N、和为 1 的分布；当路由完全均匀（f_i = P_i = 1/N）时，
aux_loss = N * N * (1/N)^2 = 1.0，这是这个 loss 形式的**理论下界**（读作"1.0 分是
满分/最均衡"）；路由越偏斜（某几个专家吃掉大部分 token 且路由概率也集中在它们身上），
aux_loss 会显著大于 1.0——这正是训练时监控"专家有没有坍缩到只用其中几个"的信号。

面试常问
--------
- 为什么 aux_loss 要同时用 f（硬分配比例，来自 top-k/argmax，不可导）和 P（软概率，
  可导）两个量相乘，而不是只用 P？—— 只用 P 无法感知"实际 token 被分配到哪"（比如
  两个专家概率都是 0.5 但 token 全被 argmax 分给了其中一个，P 本身看不出这个不均衡），
  f 提供硬分配的真实信号；但 f 本身是 argmax/topk 产生的，对 logits 不可导，训练时
  f 通常被当常量（stop-gradient），梯度只经 P 回传——这是这个 loss 能训练的关键设计。
- k=1（Switch Transformer）和 k>1（Mixtral 等）的路由权重处理有什么区别？—— k=1 时
  "top-1 里重新归一化"退化成权重恒为 1（选中的那个专家概率本身就是全部权重），
  不需要额外操作；k>1 时才需要显式在选中的 k 个里重新做一次归一化。

常见实现陷阱
------------
1. **f_i 忘记除以 k**：只除以 T，导致均匀路由时 sum_i f_i = k 而不是 1，
   aux_loss 理论下界变成 k 而不是 1.0——k=1 时看不出这个 bug（因为此时 k=1
   两者一样），必须用 k>1 的用例才能测出来。
2. **P_i 算错来源**：应该用**完整** softmax(logits) 的概率（对全部 N 个专家都有值），
   不能只对被选中的 top-k 专家的概率取平均——否则 aux_loss 完全无法感知"哪些专家
   从来没被选中过"，起不到监控坍缩的作用。
3. **忘记重新归一化 topk_weight**：直接把全 softmax 的原始概率截出 top-k 当权重用，
   这几个权重加起来通常小于 1（因为其余专家还分走了一部分概率质量），会让 FFN
   输出的整体尺度系统性偏小。
4. **每个 token 选中的专家数不对/有重复**：topk 的下标必须是同一个 token 里
   k 个互不相同的专家（结构性 bug，比如手写 top-k 时下标算重了）。
"""
from __future__ import annotations

import torch


def moe_topk_router_with_aux_loss(logits: torch.Tensor, k: int, num_experts: int):
    """见模块 docstring：返回 (topk_idx, topk_weight, aux_loss)。"""
    raise NotImplementedError("闭卷手写：删除这行 raise，实现 MoE top-k 路由 + 负载均衡辅助损失")
