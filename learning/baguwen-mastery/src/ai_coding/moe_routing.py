"""MoE top-k 门控路由 + 负载均衡辅助损失，纯 stdlib 手写（不用 numpy/torch）。

实现 4 个专家、top-2 门控：对一批 token 的 gate logits 做 softmax，选出概率最高的 top-2
专家当作该 token 实际调用的专家。负载均衡辅助损失沿用 Switch Transformer/GShard 的经典
形式 L_aux = N·Σ_i f_i·P_i（N=专家数，f_i=被路由到专家 i 的 token 比例、P_i=专家 i 的平均
路由概率），并额外算一个"专家使用频率方差"作为直观的负载不均程度诊断量。

_self_test() 用完全确定性的手工构造输入验证一个结构性质：'偏斜'门控(明显偏好 1-2 个专家)
的使用频率方差、辅助损失都应显著大于'均匀'门控(各专家概率接近)的对应数值——这个大小关系
不依赖随机数，纯靠手工构造的输入即可确定性验证。
"""
from __future__ import annotations

import math


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def top_k_indices(probs: list[float], k: int) -> list[int]:
    """选出概率最高的 k 个专家下标，同分时下标小的优先（确定性 tie-break）。"""
    order = sorted(range(len(probs)), key=lambda i: (-probs[i], i))
    return sorted(order[:k])


def route_batch(gate_logits_batch: list[list[float]], num_experts: int, k: int) -> tuple[list[float], list[float]]:
    """对一批 token 的 gate logits 做路由，返回 (f, P)：
    f[i] = 专家 i 被选中的 token 比例；P[i] = 专家 i 在整批上的平均路由概率(softmax 后、选择前)。
    """
    t = len(gate_logits_batch)
    dispatch_count = [0] * num_experts
    prob_sum = [0.0] * num_experts
    for logits in gate_logits_batch:
        assert len(logits) == num_experts
        probs = softmax(logits)
        chosen = top_k_indices(probs, k)
        for i in chosen:
            dispatch_count[i] += 1
        for i in range(num_experts):
            prob_sum[i] += probs[i]
    f = [c / t for c in dispatch_count]
    p = [s / t for s in prob_sum]
    return f, p


def aux_loss(gate_logits_batch: list[list[float]], num_experts: int, k: int) -> float:
    """Switch/GShard 风格负载均衡辅助损失：L_aux = N * sum_i(f_i * P_i)。

    完全均衡时 f_i=P_i=k/N 与 1/N，L_aux 的理论最小值为 k（与 num_experts 无关）。
    """
    f, p = route_batch(gate_logits_batch, num_experts, k)
    return num_experts * sum(fi * pi for fi, pi in zip(f, p))


def usage_variance(f: list[float]) -> float:
    """专家使用频率 f 的（总体）方差，越大代表负载越不均。"""
    mean = sum(f) / len(f)
    return sum((x - mean) ** 2 for x in f) / len(f)


def _self_test() -> None:
    num_experts, k = 4, 2

    # 1) "偏斜"批次：每个 token 的 gate logits 都强烈偏好专家 0、其次专家 1，
    #    专家 2/3 几乎不可能被 top-2 选中 —— 明显的路由坍缩场景。
    skewed_batch = [[10.0, 5.0, -10.0, -10.0] for _ in range(6)]
    f_skew, p_skew = route_batch(skewed_batch, num_experts, k)
    var_skew = usage_variance(f_skew)
    loss_skew = aux_loss(skewed_batch, num_experts, k)
    assert f_skew[0] == 1.0 and f_skew[1] == 1.0, f_skew
    assert f_skew[2] == 0.0 and f_skew[3] == 0.0, f_skew

    # 2) "均匀"批次：4 个 token 的 logits 互为循环移位(cyclic shift)，
    #    每个专家在 4 个 token 里恰好各被选中 2 次(f_i=0.5)、
    #    且由对称性平均路由概率 P_i 对所有专家也完全相等 —— 精确均衡，无需随机数。
    base = [6.0, 3.0, -6.0, -6.0]
    uniform_batch = [base[-i:] + base[:-i] for i in range(num_experts)]
    f_unif, p_unif = route_batch(uniform_batch, num_experts, k)
    var_unif = usage_variance(f_unif)
    loss_unif = aux_loss(uniform_batch, num_experts, k)
    assert all(abs(fi - 0.5) < 1e-9 for fi in f_unif), f_unif
    assert all(abs(pi - p_unif[0]) < 1e-9 for pi in p_unif), p_unif

    # 3) 均衡批次应达到理论最小值 L_aux = k
    assert abs(loss_unif - k) < 1e-9, loss_unif

    # 4) 核心结构性质：偏斜情况下方差、辅助损失都应明显大于均匀情况 —— 确定性大小关系，不依赖随机数。
    assert var_skew > var_unif, (var_skew, var_unif)
    assert loss_skew > loss_unif, (loss_skew, loss_unif)

    print(
        f"[PASS] moe_routing: 偏斜(方差={var_skew:.3f},aux_loss={loss_skew:.3f}) "
        f"> 均匀(方差={var_unif:.3f},aux_loss={loss_unif:.3f}==理论最小值 k={k})"
    )


if __name__ == "__main__":
    _self_test()
