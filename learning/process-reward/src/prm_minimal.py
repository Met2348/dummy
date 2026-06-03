"""PRM (Process Reward Model) minimal — Lightman 2023.

ORM 给整个 response 打 1 个分；PRM 给每步打 1 个分.
训练：标 (step_text, label∈{good, neutral, bad}) -> 3-way cross entropy.
推理：step-by-step gen，每步用 PRM 评分，sum/min/min-last 聚合.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRMHead(nn.Module):
    """在 LM 最后 hidden 上加 3-way classifier."""

    def __init__(self, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_at_step_end: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_at_step_end)


class PRM(nn.Module):
    """PRM = LM backbone + step-end classifier head."""

    def __init__(self, lm, num_labels: int = 3):
        super().__init__()
        self.lm = lm
        hidden = getattr(lm.config, "hidden_size", getattr(lm.config, "n_embd", 768))
        self.head = PRMHead(hidden, num_labels)

    def score_steps(self, input_ids: torch.Tensor, step_end_positions: list[list[int]],
                    attention_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        """每个 step 的结束位置 hidden 状态 → 3-way logits."""
        out = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_h = out.hidden_states[-1]   # (B, T, H)
        result = []
        for b, positions in enumerate(step_end_positions):
            h_at_steps = last_h[b, positions]   # (n_steps, H)
            logits = self.head(h_at_steps)
            result.append(logits)
        return result


def prm_loss(logits_list: list[torch.Tensor],
             labels_list: list[torch.Tensor]) -> torch.Tensor:
    """3-way cross entropy, 平均到 step."""
    losses, n = [], 0
    for logits, labels in zip(logits_list, labels_list):
        losses.append(F.cross_entropy(logits, labels, reduction="sum"))
        n += labels.numel()
    return torch.stack(losses).sum() / max(n, 1)


# ===== aggregation 策略 =====

def aggregate_step_scores(step_logits: torch.Tensor, mode: str = "min_last") -> float:
    """step_logits: (n_steps, 3); 转 prob_good 后聚合."""
    probs = F.softmax(step_logits, dim=-1)
    good_prob = probs[:, 0]   # label 0 = good
    if mode == "mean":
        return good_prob.mean().item()
    if mode == "min":
        return good_prob.min().item()
    if mode == "min_last":
        # 最后一步必须 good (Lightman 2023 推荐)
        return min(good_prob.min().item(), good_prob[-1].item())
    if mode == "product":
        return good_prob.prod().item()
    raise ValueError(mode)


if __name__ == "__main__":
    print("PRM minimal — smoke test\n" + "=" * 50)
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_size": 32})()
            self.emb = nn.Embedding(100, 32)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            h = self.emb(input_ids)
            return type("O", (), {"hidden_states": (h,)})()

    prm = PRM(TinyLM())
    input_ids = torch.randint(0, 100, (2, 20))
    step_end = [[4, 9, 14, 19], [3, 8, 13, 18]]  # 4 步 each
    logits_list = prm.score_steps(input_ids, step_end)
    labels_list = [torch.tensor([0, 0, 2, 0]), torch.tensor([0, 1, 0, 0])]
    L = prm_loss(logits_list, labels_list)
    print(f"  loss = {L.item():.4f}")
    for mode in ("mean", "min", "min_last", "product"):
        s = aggregate_step_scores(logits_list[0], mode)
        print(f"  aggregate({mode}) = {s:.3f}")
