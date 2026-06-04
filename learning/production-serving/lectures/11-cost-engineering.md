# L11 · 成本工程

## 1 · 关键公式
`$/M token = GPU_$_per_h / tok_per_s / 3.6`

例：H100 $4/h, 5000 tok/s → $4 / 5000 / 3.6 = **$0.22 / M token**

OpenAI GPT-4 价格 $2.5 / M token → 卖给客户每 M token 利润 $2.28。

## 2 · 优化 lever
| lever | $/M token 降幅 |
|-------|--------------|
| AWQ 4bit (省显存 → 更大 batch) | 30-50% |
| EAGLE-2 投机 | 30% |
| prefix cache | 20-50% (取决于场景) |
| disaggregated P/D | 20-30% |
| FP8 (H100+) | 25% |
| 综合 | **70-85%** |

## 3 · 不同模型 $/M token
| 模型 | 成本 (估) |
|------|---------|
| GPT-4 | $2.50 |
| Claude 3.7 Sonnet | $3.00 |
| Llama-3.3 70B (开源) | $0.50 |
| Qwen-72B (开源) | $0.40 |
| Llama-3 8B | $0.15 |
| **5090 + Qwen-7B AWQ** | **$0.05** |

## 4 · cache 收益
- system prompt cache 命中 → 输入 token 折扣 90%
- OpenAI cached_input 计价 = 1/10 正常

## 5 · cost-aware routing
- 简单 query → 小模型（便宜）
- 复杂 query → 大模型（贵）
- 节省 70% 总 cost

## 6 · 实现：[cost_calc.py](../src/cost_calc.py)
- $/M token 计算器
- batch size 优化
- ROI 分析
