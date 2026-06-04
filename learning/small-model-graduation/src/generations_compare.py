"""5 ckpt 同 prompt 生成对照."""
from __future__ import annotations

from pathlib import Path


PROMPTS = [
    "The quick brown fox jumped over",
    "In a small village, there lived a",
    "Once upon a time, there was a",
    "The chef cracked an egg into",
    "The robot turned on its lights and",
]


MOCK_OUTPUTS = {
    "A": {
        "The chef cracked an egg into":
            "the pan and the the the and the cooked it",
        "Once upon a time, there was a":
            "boy. He went to the school. He saw the boy.",
        "default": "[repetitive, simple]",
    },
    "B": {
        "The chef cracked an egg into":
            "the pan and started cooking dinner.",
        "Once upon a time, there was a":
            "small girl who lived in a village.",
        "default": "[fluent, short]",
    },
    "C": {
        "The chef cracked an egg into":
            "the sizzling pan. The aroma of butter filled "
            "the kitchen as he reached for the spatula.",
        "Once upon a time, there was a":
            "curious young inventor who built strange machines "
            "from scrap metal and old gears.",
        "default": "[contextual, rich]",
    },
    "D": {
        "default": "[same as C, long ctx-aware]",
    },
    "E": {
        "The chef cracked an egg into":
            "the hot skillet. The yolk burst open and "
            "spread across the buttered surface as he "
            "deftly flipped it with practiced ease.",
        "Once upon a time, there was a":
            "young apprentice who dreamed of becoming a great "
            "sorcerer, despite being told she had no magical "
            "talent at all.",
        "default": "[best - detailed + coherent]",
    },
}


def get_output(variant: str, prompt: str) -> str:
    bank = MOCK_OUTPUTS.get(variant, {})
    return bank.get(prompt, bank.get("default", ""))


def write_generations_md(path: str = "generations.md"):
    md = ["# 5 ckpt 同 prompt 输出对照", ""]
    for p in PROMPTS:
        md.append(f"## Prompt: {p!r}\n")
        for v in ["A", "B", "C", "D", "E"]:
            md.append(f"### ckpt {v}")
            md.append(f"> {get_output(v, p)}\n")
    Path(path).write_text("\n".join(md), encoding="utf-8")


def generation_template():
    return """
import torch
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=80, temp=0.8, top_p=0.95):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    for _ in range(max_new_tokens):
        logits = model(ids)
        last = logits[:, -1, :] / temp
        probs = torch.softmax(last, dim=-1)
        sorted_p, sorted_i = probs.sort(descending=True)
        cumsum = sorted_p.cumsum(-1)
        mask = cumsum <= top_p
        if mask.sum() == 0: mask[0] = True
        sorted_p = sorted_p * mask
        sorted_p /= sorted_p.sum(-1, keepdim=True)
        next_id = sorted_i[torch.multinomial(sorted_p, 1)]
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=-1)
    return tokenizer.decode(ids[0])
"""


if __name__ == "__main__":
    print("=== Mock 5-ckpt outputs ===")
    for p in PROMPTS[:2]:
        print(f"\nPrompt: {p!r}")
        for v in ["A", "B", "C", "E"]:
            print(f"  {v}: {get_output(v, p)}")
    print()
    print("Template:")
    print(generation_template())
