"""Capstone Track B — Qwen-1.5B + GSM8K + 4bit LoRA 挑战轨.

目标: 真训练，看 aha moment 涌现
    base: Qwen2.5-1.5B-Base
    quant: 4bit (bitsandbytes)
    LoRA: r=16, target=[q,k,v,o]
    task: GSM8K-tiny (500 训 + 100 测)
    reward: 0.1 * format + 0.9 * accuracy
    algo: GRPO (k=4 起步, 显存制约)
    显存: 5090 24GB OK (4bit + LoRA)
    时长: 单跑 ~4h
预期: acc 5%→25%, len 100→250, "wait/reconsider" 词频 ≥ 5%
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

REPO_SRC = Path(__file__).parent
sys.path.insert(0, str(REPO_SRC))

from rewards.format_reward import format_reward            # noqa
from rewards.accuracy_reward import gsm8k_extract_answer, gsm8k_reward  # noqa


# ===== aha moment 词频检测 =====

AHA_WORDS = ["wait", "let me reconsider", "let me check", "actually",
             "i made a mistake", "let me try again", "rethink",
             "double-check", "verify", "重新", "等等"]


def aha_word_frequency(responses: list[str]) -> dict:
    counts = {w: 0 for w in AHA_WORDS}
    total_words = 0
    for r in responses:
        low = r.lower()
        total_words += len(low.split())
        for w in AHA_WORDS:
            if w in low:
                counts[w] += 1
    return {
        "total_responses": len(responses),
        "responses_with_aha": sum(1 for r in responses
                                   if any(w in r.lower() for w in AHA_WORDS)),
        "aha_ratio": sum(1 for r in responses
                          if any(w in r.lower() for w in AHA_WORDS)) / max(len(responses), 1),
        "word_counts": {k: v for k, v in counts.items() if v > 0},
    }


def gsm8k_reward_full(response: str, ground_truth: str, alpha: float = 0.1) -> dict:
    f = format_reward(response)
    m = re.search(r"<answer>(.+?)</answer>", response, re.DOTALL)
    a = 0.0
    if m:
        pred = gsm8k_extract_answer(m.group(1))
        a = gsm8k_reward(pred, ground_truth)
    return {"format": f, "accuracy": a, "total": alpha * f + (1 - alpha) * a}


def setup_lora_qwen(model_name: str = "Qwen/Qwen2.5-1.5B"):
    """配置 4bit + LoRA 加载（真实训练用）.

    伪代码（CPU 无法跑）.
    """
    return f"""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
print(f"trainable params: {{model.num_parameters(only_trainable=True):,}}")
"""


def grpo_config_track_b() -> dict:
    return {
        "algo": "GRPO",
        "k": 4,
        "eps": 0.2,
        "beta_kl": 0.04,
        "lr": 5e-6,
        "max_response_len": 256,
        "rollout_batch": 16,
        "ppo_epochs": 2,
        "total_steps": 1000,
        "temperature": 0.7,
    }


if __name__ == "__main__":
    print("Capstone Track B — Qwen-1.5B GSM8K (mock demo)\n" + "=" * 60)
    print("LoRA + 4bit setup 伪代码:")
    print(setup_lora_qwen())
    print("\nGRPO config:")
    for k, v in grpo_config_track_b().items():
        print(f"  {k}: {v}")

    # mock 后期 aha 词频
    sample_responses = [
        "<think>16-3=13, then... wait, let me reconsider, she eats 3 first</think><answer>#### 7</answer>",
        "<think>let me try: 16-3-6=7</think><answer>#### 7</answer>",
        "<think>actually let me double-check: 16-9=7</think><answer>#### 7</answer>",
        "<think>simple subtraction</think><answer>#### 7</answer>",
        "<think>I made a mistake, redo: 16-3=13, 13-6=7</think><answer>#### 7</answer>",
    ]
    stats = aha_word_frequency(sample_responses)
    print(f"\nAha moment 检测 (mock 5 responses):")
    print(f"  with-aha: {stats['responses_with_aha']}/{stats['total_responses']}")
    print(f"  ratio:    {stats['aha_ratio']:.1%}")
    print(f"  words:    {stats['word_counts']}")
