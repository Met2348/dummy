"""解码采样：greedy / temperature / top-k / top-p(nucleus) / beam search。

面试高频度 ★★★★★。得分点：top-k 固定候选数；top-p 固定累计概率质量（自适应候选数）；
temperature<1 更尖锐、>1 更平坦；beam 是"宽度受限的 BFS"，按 logprob 累加。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp <= 0:
        raise ValueError("temperature 必须 > 0（=0 请用 greedy）")
    return logits / temp


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """保留 top-k，其余置 -inf。logits: (..., V)。"""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = logits.topk(k, dim=-1).values[..., -1, None]     # 第 k 大的值
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """nucleus：保留累计概率 >= p 的最小集合，其余置 -inf。"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    # 保留：cum 首次 >= p 的那个也要留（右移一位）
    remove = cum > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # 散射回原顺序
    out = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return out


def greedy_next(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def beam_search(step_fn, start: list[int], n_steps: int, beam_width: int, vocab: int):
    """通用 beam search。step_fn(seq)->logits(V,)（确定性）。返回最优序列。"""
    beams = [(0.0, list(start))]                            # (累计 logprob, 序列)
    for _ in range(n_steps):
        cand = []
        for score, seq in beams:
            logp = F.log_softmax(step_fn(seq), dim=-1)
            topv, topi = logp.topk(beam_width)
            for lp, tok in zip(topv.tolist(), topi.tolist()):
                cand.append((score + lp, seq + [tok]))
        cand.sort(key=lambda x: x[0], reverse=True)
        beams = cand[:beam_width]
    return beams[0][1]


def _self_test() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])

    # greedy 取 argmax
    assert greedy_next(logits).item() == 0

    # top-k=2 恰保留 2 个有限值
    fk = top_k_filter(logits, 2)
    assert torch.isfinite(fk).sum().item() == 2
    assert set(torch.isfinite(fk).nonzero().flatten().tolist()) == {0, 1}

    # top-p：p 很小时只留 top-1；p=1 全留
    fp_small = top_p_filter(logits, 0.01)
    assert torch.isfinite(fp_small).sum().item() == 1
    assert torch.isfinite(top_p_filter(logits, 1.0)).all()

    # temperature：升温使分布更平坦（熵增）
    def entropy(z):
        pr = z.softmax(-1)
        return -(pr * pr.clamp_min(1e-9).log()).sum()
    assert entropy(apply_temperature(logits, 2.0)) > entropy(logits)

    # beam 优于 greedy：构造一个"贪心陷阱"
    # 第一步 token1 略高，但选 token0 后第二步能拿到大奖励
    table = {
        (): torch.tensor([1.0, 1.1, 0.0]),        # 起点：token1 略高
        (0,): torch.tensor([0.0, 0.0, 5.0]),      # 选0后：token2 大奖励
        (1,): torch.tensor([0.0, 0.0, 0.0]),      # 选1后：平淡
    }
    step = lambda seq: table[tuple(seq)]
    greedy_seq = [step([]).argmax().item()]
    greedy_seq.append(step(greedy_seq).argmax().item())
    beam_seq = beam_search(step, [], n_steps=2, beam_width=2, vocab=3)
    assert greedy_seq == [1, 0] or greedy_seq[0] == 1     # 贪心咬住 token1
    assert beam_seq == [0, 2], beam_seq                   # beam 找到全局最优
    print("[PASS] sampling: greedy/top-k/top-p/温度熵/beam 破贪心陷阱")


if __name__ == "__main__":
    _self_test()
