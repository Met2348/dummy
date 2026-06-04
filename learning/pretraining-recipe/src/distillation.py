"""KD loss + 数据合成 demo."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            T: float = 4.0) -> torch.Tensor:
    """soft KL with temperature."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)


def combined_loss(student_logits, teacher_logits, targets,
                   alpha: float = 0.5, T: float = 4.0) -> torch.Tensor:
    ce = F.cross_entropy(student_logits.flatten(0, -2),
                          targets.flatten())
    kd = kd_loss(student_logits, teacher_logits, T)
    return alpha * ce + (1 - alpha) * kd


def data_synthesis_template():
    return """
1. Define synthesis spec:
   - genre: textbook / story / dialogue
   - target tokens: 100k
   - quality: high (peer-reviewed style)

2. Use larger model (GPT-4o / Claude / DeepSeek-V3):
   prompt = '''
   Write a textbook chapter on {topic}
   suitable for grade {grade}. Length {n_words}.
   '''
   responses = call_api(prompt × 1000)

3. Filter:
   - perplexity (smaller is good)
   - n-gram dedup
   - safety classifier

4. Tokenize + add to training set
"""


if __name__ == "__main__":
    s = torch.randn(2, 8, 100)
    t = torch.randn(2, 8, 100)
    y = torch.randint(0, 100, (2, 8))

    kd = kd_loss(s, t, T=4.0)
    print(f"KD loss: {kd.item():.4f}")

    full = combined_loss(s, t, y, alpha=0.5, T=4.0)
    print(f"Combined loss: {full.item():.4f}")

    print("\n=== 数据合成模板 ===")
    print(data_synthesis_template())
