"""轻量 eval (val_loss + tiny multiple-choice)."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def validation_loss(model, data, sample_batch_fn, n_batch: int = 50,
                    seq_len: int = 512, batch: int = 8) -> float:
    model.eval()
    total = 0.0
    for _ in range(n_batch):
        x, y = sample_batch_fn(data, seq_len=seq_len, batch_size=batch)
        x = x.to(next(model.parameters()).device)
        y = y.to(x.device)
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        total += loss.item()
    model.train()
    return total / n_batch


def perplexity(loss: float) -> float:
    return math.exp(loss)


@torch.no_grad()
def multiple_choice_score(model, tokenize, prompt: str,
                          choices: list) -> int:
    """返回得分最高的 choice idx."""
    scores = []
    device = next(model.parameters()).device
    for c in choices:
        ids = tokenize(prompt + " " + c)
        ids = torch.tensor(ids, device=device).unsqueeze(0)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, 1), ids[:, 1:].flatten(),
            reduction="sum",
        )
        scores.append(-loss.item() / max(1, len(tokenize(c))))
    return int(torch.tensor(scores).argmax().item())


TINY_HELLASWAG = [
    {"prompt": "He picked up the basketball and",
     "choices": ["shot it at the hoop.", "ate a sandwich.",
                  "slept on it.", "discussed quantum physics."],
     "answer": 0},
    {"prompt": "The chef cracked an egg into the pan and",
     "choices": ["started juggling.", "fried it sunny-side up.",
                  "wrote a poem.", "called the police."],
     "answer": 1},
    {"prompt": "She turned the key in the ignition and",
     "choices": ["the car started.", "the dog barked.",
                  "snow fell.", "she went to sleep."],
     "answer": 0},
]


def run_tiny_hellaswag(model, tokenize) -> float:
    correct = 0
    for ex in TINY_HELLASWAG:
        pred = multiple_choice_score(model, tokenize,
                                       ex["prompt"], ex["choices"])
        if pred == ex["answer"]:
            correct += 1
    return correct / len(TINY_HELLASWAG)


if __name__ == "__main__":
    print("Eval helpers ready.")
    print(f"  tiny-HellaSwag has {len(TINY_HELLASWAG)} examples")
    print(f"  ppl(loss=2.3) = {perplexity(2.3):.2f}")
