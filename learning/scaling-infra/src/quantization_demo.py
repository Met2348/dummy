"""推理量化 - 显存账本 + GPTQ/AWQ/FP8 调用模板."""
from __future__ import annotations


def memory_table(n_params: int) -> dict:
    return {
        "fp32":  n_params * 4 / 1e9,
        "bf16":  n_params * 2 / 1e9,
        "fp8":   n_params * 1 / 1e9,
        "int8":  n_params * 1 / 1e9,
        "int4":  n_params * 0.5 / 1e9,
    }


def quant_templates():
    print("=== GPTQ ===")
    print("""
from auto_gptq import AutoGPTQForCausalLM
m = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-3-8B-GPTQ",
    device="cuda:0",
)
""")

    print("=== AWQ ===")
    print("""
from awq import AutoAWQForCausalLM
m = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-3-8B-AWQ",
    fuse_layers=True,
)
""")

    print("=== FP8 via vLLM ===")
    print("""
from vllm import LLM
llm = LLM(
    "meta-llama/Llama-3-8B",
    quantization="fp8",
    kv_cache_dtype="fp8",
)
""")

    print("=== bitsandbytes 4-bit ===")
    print("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
m = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb,
    device_map="auto",
)
""")


if __name__ == "__main__":
    print("=== 显存 (8B model) ===")
    for k, v in memory_table(8_000_000_000).items():
        print(f"  {k}: {v:.1f} GB")
    print()
    quant_templates()
