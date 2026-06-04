# L06 · LLM.int8()（Dettmers et al. 2022）

## 1 · 解决问题
"6.7B+ 模型 int8 量化掉精度" → Dettmers 发现是**少数 outlier 通道**导致。

## 2 · idea：outlier 分离
- 90% 通道 (无 outlier) → int8 量化
- 10% 通道 (含 outlier) → fp16 保留

mixed precision matmul:
```
y = X_int8 · W_int8 + X_outlier_fp16 · W_outlier_fp16
```

## 3 · 收益
- 总精度损失 < 1pp on MMLU
- 显存：weight 减半（fp16 → 9-bit avg）
- 速度：**慢**（mixed matmul kernel 不优化）

## 4 · 为什么慢
- 两个 matmul 串行
- bitsandbytes 实现使用 PyTorch + 简单 CUDA
- 显存收益主导，速度可接受（≈ fp16）

## 5 · 部署
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    load_in_8bit=True,
    device_map="auto",
)
```

## 6 · 后续方向
- LLM.int8() → 验证 outlier 假说
- 之后 SmoothQuant / GPTQ 解决 outlier 不同思路
- bitsandbytes 4bit (nf4 / fp4) 进一步压

## 7 · nf4 (NormalFloat 4-bit)
- bitsandbytes 4bit 的 dtype
- 假设权重~正态分布，按 quantile 划 16 个桶
- 比 fp4 均匀分桶精度高

## 8 · 实现：见 `bnb_int4.py`（lib 轨；minimal 部分略）
