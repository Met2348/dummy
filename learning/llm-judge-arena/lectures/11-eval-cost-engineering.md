# L11 · Eval 成本工程

## 真实数据（2025 价格）

| 平台 | 每 1M token 价 |
|------|---------------|
| GPT-4o (Mar 2025) | $2.50 in / $10 out |
| GPT-4-turbo | $10 / $30 |
| Claude 3.7 Sonnet | $3 / $15 |
| Prometheus 2 7B (本地) | ~$0.05 (电费) |

## 评测全套成本估算

**MT-Bench (80 题 × 2 turns × 8 candidates)**：
- 每 judgement ≈ 1500 token in + 200 token out
- 总 in: 1.6M token, out: 200k token
- 用 GPT-4o: $4 + $2 = **$6**
- 用 GPT-4-turbo: $16 + $6 = **$22**

**Arena-Hard (500 题 × 30 candidates × 2 swap)**：
- 30k pairs × 3k token = 90M token in, 10M out
- GPT-4o: $225 + $100 = **$325**
- GPT-4-turbo: $900 + $300 = **$1200**

**Chatbot Arena**: 100M+ vote → 累计 $$M。

## 5 大省钱策略

### 1. Judge ladder
```
GPT-4o → Prometheus 2 → Llama-3 8B
```
80% case 用便宜 judge，难判的用贵 judge。

### 2. Subset 评测
500 pair → 50 pair：
- 选 model-pair high-variance 的题
- 信息量保留 ~80%

### 3. Cache + reuse
同 model 同 prompt → cache response。
LMSYS 缓存 GPT-4-0314 baseline → 永久免费。

### 4. Batch API
OpenAI Batch API: **50% off**（24h SLA）。
Anthropic 类似。

### 5. 开源 judge
Prometheus 2 / Skywork-Critic 接近 GPT-4 准确率，本地零成本。

## 评测预算决策树

```
question: 评测目的？
├── research paper → 用 GPT-4 全套（可信）
├── leaderboard → Arena-Hard subset + Prometheus 2 (省钱)
├── model 迭代 → Prometheus 2 (快)
└── 在线 A/B → 真实用户 + 简单 metric (无需 judge)
```

## 一句话

> Judge 不便宜，**省钱在 subset + 开源 judge + cache**。
