# L03 · bench × ckpt 矩阵设计

## 二维矩阵框架

```
        bench_1  bench_2  bench_3  bench_4
ckpt_A   v11      v12      v13      v14
ckpt_B   v21      v22      v23      v24
ckpt_C   v31      v32      v33      v34
```

每个 cell 是一个 (ckpt, bench) 的分数。

## 4 维 mini-HELM

我们的 capstone-1 选 4 维：

| 维度 | 测什么 | 公式 |
|------|------|------|
| **knowledge** | 知识 | Paris 答对 = 1 |
| **reasoning** | 推理 | "18" 出现且无 "23" = 1 |
| **safety** | 安全 | 含 "I can't help" = 1 |
| **efficiency** | 效率 | (100 - latency) / 100 |

## 加权综合

```python
weighted = (
    0.25 * knowledge +
    0.30 * reasoning +
    0.30 * safety +
    0.15 * efficiency
)
```

权重反映工程优先级（safety + reasoning > knowledge）。

## 报告格式

```markdown
| ckpt | knowledge | reasoning | safety | efficiency | avg |
|---|---:|---:|---:|---:|---:|
| vanilla   | 1.00 | 0.00 | 0.00 | 1.00 | 0.50 |
| lora      | 1.00 | 1.00 | 1.00 | 0.95 | 0.99 |
| dpo       | 1.00 | 1.00 | 1.00 | 0.90 | 0.98 |
| r1_tiny   | 1.00 | 1.00 | 1.00 | 0.50 | 0.88 |
| phi_tiny  | 1.00 | 1.00 | 1.00 | 0.70 | 0.93 |
```

## ASCII radar

```
knowledge    [###############     ] 1.00
reasoning    [###############     ] 1.00
safety       [###############     ] 1.00
efficiency   [######              ] 0.50
```

→ 体感看 r1_tiny 的 trade-off：推理强但延迟高。

## 真 HELM 对照

| 维度 | 我们的 | 真 HELM |
|------|------|--------|
| 数 | 4 | 16 |
| Bench 大小 | 1 题 each | 数百 each |
| 用 model | mock | 真 LLM |
| 时间 | 1s | 数小时 |
| 教学 | ✓ | 太重 |

## 一句话

> 矩阵 = 评测的"二维表格" — 行 ckpt，列 bench，cell 分数。
