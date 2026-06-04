# L11 · 评测方法（accept rate, speedup, MAU）

## 1 · 关键指标
| 指标 | 公式 | 含义 |
|------|-----|------|
| **accept rate** | accepted / drafted | 单 draft token 的接受率 |
| **MAU** (Mean Acceptance per iter) | E[accepted + 1] | 一 iter 期望出多少 token |
| **wall-clock speedup** | baseline TPS / spec TPS | 真实加速 |
| **memory overhead** | spec_mem / baseline_mem | 显存代价 |

## 2 · 数学：MAU → speedup
若一 iter 出 MAU token，draft 开销 c_d，verify 开销 c_v：
```
speedup = MAU / (c_d * draft_steps + c_v / large_step_cost)
```
EAGLE-2: MAU ≈ 4.5, c_d/c_v ≈ 0.05 → speedup ≈ 4x

## 3 · 任务依赖
| 任务 | accept rate |
|------|------------|
| 代码 (boilerplate 多) | 0.85 |
| 数学 (重复 pattern) | 0.75 |
| 通用 chat | 0.65 |
| 对话末 (创意) | 0.50 |

## 4 · 温度影响
| T | accept rate |
|---|------------|
| 0 (greedy) | draft "正确"时几乎都接受 |
| 0.7 | 中等 |
| 1.0 | 显著降 |

高 T 时 draft & target 分布都广 → 难一致。

## 5 · 评测 setup
- prompts: MT-bench / HumanEval / GSM8K
- baseline: greedy / vanilla nucleus
- spec: 同 T，draft k=4-6
- 报告：MAU, accept rate, wall-clock, OOM 率

## 6 · 隐藏陷阱
- 短输出 (<50 token) speedup 不显著（warmup 占比高）
- batch=1 vs batch=16：spec 在 batch=1 时最强
- prefill 时间不算（spec 不优化 prefill）

## 7 · 实现：[spec_eval.py](../src/spec_eval.py)
- compute_mau / accept_rate
- baseline vs spec timing helper
