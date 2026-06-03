"""QLoRA peft + bitsandbytes 调包版（GPU only）。

GPU + bitsandbytes 才能跑真量化。CPU 上会 fallback 到 fake-quant 的 minimal 版。

需要：
    pip install bitsandbytes accelerate
    GPU + CUDA
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from common import print_param_summary  # noqa: E402


def build_peft_qlora_tinyllama(
    r: int = 8,
    alpha: int = 16,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
):
    """构造 QLoRA + TinyLlama 模型，GPU only。

    在 GPU 上跑：bitsandbytes 真 NF4 + LoRA
    在 CPU 上跑：报错（bitsandbytes 4-bit 不支持 CPU）
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bitsandbytes 4-bit 量化需要 GPU。CPU 环境请用 qlora_minimal.py 的 fake-quant 版。"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = prepare_model_for_kbit_training(base)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],  # LLaMA attention
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base, config)
    return model, tokenizer


def main() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] 没有 GPU，QLoRA + bitsandbytes 真量化无法运行")
        print("在 CPU 上请用: python qlora_minimal.py（fake-quant 版）")
        return

    try:
        print("尝试加载 TinyLlama-1.1B + 4-bit NF4 量化 + LoRA...")
        model, tokenizer = build_peft_qlora_tinyllama(r=8, alpha=16)
        print_param_summary(model, "QLoRA peft (TinyLlama, NF4, r=8)")

        inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs)
        print(f"\nforward logits.shape: {tuple(out.logits.shape)}")
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        print("\n常见原因：")
        print("  - bitsandbytes 与 torch CUDA 版本不兼容")
        print("  - GPU sm 不被 bitsandbytes 支持（Blackwell sm_120 需要 bnb 0.43+）")
        print("  - HF 模型下载失败（需要联网或本地缓存）")


if __name__ == "__main__":
    main()
