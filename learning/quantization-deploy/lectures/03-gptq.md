# L03 · GPTQ（Frantar et al. 2023, IST Austria）

## 1 · 核心 idea
朴素量化每个权重独立 round → 累积误差。
**GPTQ**：用二阶 Hessian 引导**贪心 column-by-column** 量化，每量一列就调整剩余列补偿误差。

## 2 · 算法
```python
# 输入: W [out_features, in_features], H = X^T X
W_q = W.clone()
for j in range(in_features):           # 一列一列处理
    q_j = quantize(W_q[:, j])           # round
    err = (W_q[:, j] - q_j) / H_inv[j, j]
    # 把误差均摊到剩余列
    W_q[:, j+1:] -= err[:, None] * H_inv[j, j+1:][None, :]
    W_q[:, j] = q_j
```

## 3 · 数学：OBS (Optimal Brain Surgeon)
最小化 `||W X - W_q X||²` 等价于：
- 对每个列误差 `err_j`，剩余列调整量 = `err_j · H_inv[j, j+1:] / H_inv[j,j]`
- H = X^T X，从 calibration 数据估计

## 4 · 实战步骤
1. 收集 calibration data（128-512 sample）
2. forward 一遍记录每层 X
3. 算 H = X^T X，cholesky 求逆
4. 量化每个 weight tensor
5. 整模型量好后存 packed int4 + scales

## 5 · 精度 (Llama-7B)
| 方案 | PPL (Wiki) |
|------|-----------|
| fp16 | 5.68 |
| naive int4 | 6.50 |
| **GPTQ 4bit** | **5.85** |
| AWQ 4bit | 5.81 |

GPTQ 几乎无损（PPL 仅升 3%）。

## 6 · 缺点
- 量化慢：7B 在单卡需 1h
- calibration 数据敏感

## 7 · 实现：[gptq_minimal.py](../src/gptq_minimal.py)
- 单 layer GPTQ 主循环（~30 行）
- 与 auto-gptq 对照（lib 轨）
