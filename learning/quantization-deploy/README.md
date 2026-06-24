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

## 变体对照（Llama-7B 论文参考值）

> ⚠️ 下表是**论文报告的真实大模型指标**（PPL/acc/tok-s 需真模型+真推理引擎，CPU toy
> 跑不出），仅作"该选哪个"的参考。**这不是** `capstone_quant_zoo.py` 的输出——脚本现在
> 在一个 toy 权重层上**真跑** 6 种量化器并打印**计算出的**重建 MSE / 压缩比 / 显存
> （见「运行验证」）。

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

## 运行验证（Runbook）

入口清单见 [`runbook.yaml`](runbook.yaml)。一键验证（V0 静态 + V1 smoke）：

```bash
python scripts/eric_3080ti_env_audit.py --runbook --modules quantization-deploy \
  --json-out docs/local-env/ERIC-3080Ti-runbook-results.json \
  --md-out  docs/local-env/ERIC-3080Ti-runbook-matrix.md
```

### 可直跑入口（2 个，纯 CPU 数值，秒级）

```bash
# Capstone 量化动物园：单层 toy 上真跑 6 种量化器，输出真实 重建MSE / 压缩比 / 显存
python learning/quantization-deploy/src/capstone_quant_zoo.py

# GPTQ 论文形态最小实现：Hessian(X^T X) + 逐列量化 + 误差补偿 vs 朴素 RTN
python learning/quantization-deploy/src/gptq_original_minimal.py
```

`capstone_quant_zoo` 现在**真的调用** `int8_basics` / `gptq_minimal` / `awq_minimal` /
`bnb_int4`(NF4) / `fp8_demo` / `smooth_quant` 六个量化器，对同一个 toy 权重层算出
**真实的输出重建 MSE**（fp16=0，8bit < 4bit，GPTQ 的 Hessian 补偿 < NF4 静态码本）；
`压缩比`/`显存`列由各方法的**真实 bit 宽度**推出（`16/bits`），不是手写常数。
（旧版本是一张硬编码 paper 数字表，未跑任何量化——已重写，见下「关键坑」。）

### 库模块（8 个，无 `__main__`，**不直跑**）

`int8_basics` · `bnb_int4` · `gptq_minimal` · `awq_minimal` · `smooth_quant` ·
`fp8_demo` · `kv_quant` · `quant_eval` 是**库模块**，没有 `__main__`。直接
`python <module>.py` 只会 import 后 no-op 退出 0（**假成功**），所以 runbook 不登记它们。
它们的正确性由测试套件（见下 V2）覆盖；其中 6 个还被上面的 capstone 真实调用演示。
`kv_quant`（KV-cache 量化）与 `quant_eval`（表格渲染）不属于"权重层"对照，仅在测试中验证。

### 测试（V2）

```bash
# 23 tests（test_quant.py 18 + test_gptq_original_minimal.py 5）
python scripts/eric_3080ti_env_audit.py --modules quantization-deploy --tests \
  --json-out /tmp/v2.json --md-out /tmp/v2.md
# 或直接： python -m pytest learning/quantization-deploy/src/tests -q
```

### 关键坑

- **`bnb_int4.py` 是手写 NF4 模拟，不是 bitsandbytes**：它内嵌 NF4 论文的 16 个
  分位码 + 逐块 absmax，**不** import `bitsandbytes`、**不**用 `load_in_4bit`。纯数值
  toy，无 CUDA 依赖（venv 里虽装了 bnb 0.49.2，本模块不用它）。
- **capstone 旧版是硬编码 paper 表**（命中"mock 没真演示出宣称效果"）：原 `VARIANTS`
  是一串手敲的 PPL/acc/mem/tok-s 字面量，未调用任何量化器，`test_capstone_*` 只断言
  字面量在场。已重写为真跑 6 种量化器 + 真实重建误差，测试改为断言真实性质
  （8bit<4bit、GPTQ≤NF4、压缩比=16/bits）。
- **真模型/真引擎路径是可选且当前未装**：`awq`(AutoAWQ) / `vllm` 在本机 venv **缺失**
  （`find_spec` 实证），下面「真模型流程」需先 `pip install` 且需较大权重下载，**不在
  V1 smoke 范围**。

## 真模型流程（可选，需 `pip install autoawq vllm` + GPU + 权重下载）

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
- [x] 23 tests pass
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
