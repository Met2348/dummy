"""VLM-R1 (OM-AI Lab 2025.02) — GRPO 直接训 VLM.

VLM-R1 = Qwen2-VL + GRPO + counting/grounding reward.
本文件: 简化 reward 函数 + mock training step (真训需 GPU 配 VLM).
"""
from __future__ import annotations

import re

import torch


# ===== Reward 函数 =====

def counting_reward(response: str, ground_truth: int) -> float:
    """从 response 提数字答案, 与 GT 对比.

    支持 "Answer: 5" / "<answer>5</answer>" / 末尾数字.
    """
    m = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
    if not m:
        m = re.search(r"[Aa]nswer:\s*(\d+)", response)
    if not m:
        m = re.search(r"(\d+)\s*$", response.strip())
    if not m:
        return 0.0
    return 1.0 if int(m.group(1)) == ground_truth else 0.0


def grounding_reward(predicted_bbox: list[float],
                     gt_bbox: list[float],
                     iou_threshold: float = 0.5) -> float:
    """IoU > 0.5 视为 reward = 1."""
    if not predicted_bbox or len(predicted_bbox) != 4:
        return 0.0
    x1 = max(predicted_bbox[0], gt_bbox[0])
    y1 = max(predicted_bbox[1], gt_bbox[1])
    x2 = min(predicted_bbox[2], gt_bbox[2])
    y2 = min(predicted_bbox[3], gt_bbox[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
    a2 = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = a1 + a2 - inter
    iou = inter / max(union, 1e-8)
    return 1.0 if iou > iou_threshold else iou


def format_reward_vlm(response: str) -> float:
    """检查 <think></think><answer></answer>."""
    pat = r"<think>.+?</think>\s*<answer>.+?</answer>"
    return 1.0 if re.search(pat, response, re.DOTALL) else 0.0


def combined_vlm_reward(response: str, gt_count: int,
                         alpha: float = 0.1) -> dict:
    f = format_reward_vlm(response)
    a = counting_reward(response, gt_count)
    return {"format": f, "accuracy": a, "total": alpha * f + (1 - alpha) * a}


# ===== Setup 伪代码 =====

def vlm_r1_setup_code() -> str:
    return """
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

# 4bit + LoRA
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)
lora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# GRPO loop:
for batch in dataloader:
    images = batch["images"]
    questions = batch["questions"]  # "How many red cubes?"
    gt_counts = batch["counts"]
    # rollout k=4
    inputs = processor(images=images, text=questions, return_tensors="pt")
    responses = model.generate(**inputs, num_return_sequences=4,
                                temperature=0.7, max_new_tokens=256)
    rewards = [combined_vlm_reward(r, gt) for r, gt in zip(responses, gt_counts)]
    # GRPO update (同 reasoning-r1 grpo_minimal)
    ...
"""


if __name__ == "__main__":
    print("VLM-R1 minimal smoke test\n" + "=" * 50)
    # Reward 测试
    print("\n[Counting reward]")
    for resp, gt in [
        ("<think>3 red, 2 blue</think><answer>3</answer>", 3),
        ("Answer: 5", 3),
        ("There are 7", 7),
        ("noise", 3),
    ]:
        r = combined_vlm_reward(resp, gt)
        print(f"  GT={gt}: {r} | {resp[:40]}")

    print("\n[Grounding IoU]")
    for pred, gt in [
        ([0, 0, 100, 100], [0, 0, 100, 100]),
        ([0, 0, 50, 50], [25, 25, 75, 75]),
        ([200, 200, 250, 250], [0, 0, 100, 100]),
    ]:
        r = grounding_reward(pred, gt)
        print(f"  pred={pred} gt={gt} → reward={r:.3f}")

    print("\n[Setup code (伪代码)]:")
    print(vlm_r1_setup_code())
