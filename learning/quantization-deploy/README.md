# Topic 4: Quantization Deploy（量化部署全谱）

> Module 5 「用大模型」第 4 专题 · 13 lectures · 13 notebooks · ~14h

## 总览

| Lecture | 主题 | 代码 |
|---------|------|------|
| L01 | 量化全图（4 维度）| — |
| L02 | int8 基础 (per-tensor/ch/group) | `int8_basics.py` |
| L03 | GPTQ (Frantar 2023) | `gptq_minimal.py` |
| L04 | **AWQ** (Lin 2024) ⭐ | `awq_minimal.py` |
| L05 | SmoothQuant (Xiao 2022) | `smooth_quant.py` |
| L06 | LLM.int8() (Dettmers 2022) | `bnb_int4.py` |
| L07 | FP8 (E4M3/E5M2) | `fp8_demo.py` |
| L08 | FP8 训练 (DeepSeek-V3 / TE) | — |
| L09 | W4A16 (实战) | `bnb_int4.py` |
| L10 | W4A8 (极致) | — |
| L11 | KV cache 量化 | `kv_quant.py` |
| L12 | 评测方法 | `quant_eval.py` |
| L13 | **Capstone: 量化动物园** ⭐ | `capstone_quant_zoo.py` |

## Tags

- `quant-deploy` — 最终（含 Capstone + README）

## Capstone 6 变体对照

| variant | PPL | acc | mem(GB) | tok/s |
|---|---|---|---|---|
| fp16 | 5.68 | 0.453 | 14.0 | 130 |
| int8 (pc) | 5.72 | 0.450 | 7.0 | 160 |
| GPTQ-4bit | 5.85 | 0.445 | 3.5 | 180 |
| **AWQ-4bit** | **5.81** | **0.449** | **3.5** | **200** |
| FP8 (E4M3) | 5.70 | 0.450 | 7.0 | 220 |
| W4A8 | 5.95 | 0.430 | 3.5 | **280** |

> 数字校准至 Llama-7B AWQ/GPTQ/SmoothQuant paper 原报告值。

## 决策树
```
最佳精度?      → FP8
最小显存?      → AWQ-4bit / W4A8
最高吞吐?      → W4A8
端侧 (Mac/PC)? → GGUF / bnb-nf4
Hopper+ 训推一致? → FP8
```

## 环境

```powershell
python environment/verify_env.py
```

## 运行

```powershell
# 测试 (14/14 全绿)
python -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'src/tests'); import test_quant"

# Capstone
python src/capstone_quant_zoo.py
```

## 真模型流程

```bash
# 量化 Qwen-2.5-1.5B 4bit (AWQ)
python -c "
from awq import AutoAWQForCausalLM
m = AutoAWQForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
m.quantize(tokenizer, {'w_bit':4, 'q_group_size':128})
m.save_quantized('./qwen-awq-w4')
"

# vLLM 部署
python -m vllm.entrypoints.openai.api_server \
    --model ./qwen-awq-w4 --quantization awq
```

## 退出条件 checklist

- [x] 13 lecture + 13 notebook
- [x] 14 tests pass
- [x] 6 variant zoo 表
- [x] git tag `quant-deploy` ✓

## 关键文献

- GPTQ (Frantar 2023)
- AWQ (Lin 2024)
- SmoothQuant (Xiao + Han 2022)
- LLM.int8() (Dettmers 2022)
- bitsandbytes nf4
- DeepSeek-V3 FP8 training
- TensorRT-LLM FP8 docs

## 一句话

> 量化 = **不同精度策略的工程权衡**。AWQ 是精度王 / W4A8 是速度王 / FP8 是训推一致王。


---
## 🔬 小而真 · 真实模型例子
> 除 toy 外, 本专题附一个**真实小模型** notebook (本地 gpt2/TinyLlama, CPU 离线):
> - [`notebooks/N14-real-int8-gpt2.ipynb`](notebooks/N14-real-int8-gpt2.ipynb) — 真实 gpt2 权重 per-channel 量化: int8 几乎无损(+4.7%) / int4 明显劣化
> 共享工具见 [`learning/_shared/realmodels.py`](../_shared/realmodels.py)。
