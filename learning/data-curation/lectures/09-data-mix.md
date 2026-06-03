# L09 · 数据配比 — Doremi / DCLM Ablation

> 22 slides | 65 min | Data Curation 第 9 讲 ⭐⭐⭐⭐⭐

---

## 学习目标

1. 理解为什么"配比"和"质量"同等重要
2. 掌握 Doremi 自动权重学习算法
3. 知道 DCLM 比赛设计与 ablation 范式
4. 能为自定义数据组合做实验设计

---

## Slide 1 · "配比" 是什么

LLM 训练数据通常多源：

```
60% web
15% code
10% wiki + arxiv
10% books
5%  math
```

权重就是 token 抽样概率。不同权重 → 不同模型偏好。

---

## Slide 2 · Llama-1 配方

```
CommonCrawl  67%
C4           15%
GitHub        4.5%
Wikipedia     4.5%
Books         4.5%
ArXiv         2.5%
StackEx       2%
```

Llama-1 论文 Table 1，纯手工经验值。

---

## Slide 3 · GPT-3 配方

```
Common Crawl    60%   (filtered)
WebText2        22%   (Reddit links)
Books1          8%
Books2          8%
Wikipedia       3%
```

注意：filtered CC 才 60%（vs Llama 67%），但精筛后质量高。

---

## Slide 4 · 手工 vs 自动配比

| | 手工 | 自动 |
|---|------|------|
| 例 | Llama-1 / GPT-3 | Doremi / RegMix |
| 工作量 | 大量 ablation | 1 次代理训练 |
| 风险 | 经验依赖 | 代理模型偏差 |
| 适用 | 探索期 | 优化期 |

---

## Slide 5 · Doremi 算法核心

Google 2023：

```
1. 训一个小代理模型（280M）一次
2. 每个 domain 跑代理模型的 ppl
3. 求 worst-case domain (loss 最高)
4. 给 worst-case 加权 → 重训代理
5. 反复，直到权重收敛
```

→ "自动找让所有 domain 都不太差的配比"。

---

## Slide 6 · Doremi 数学

minimax 优化：

```
α* = argmin_α  E_{domain~α} [ L_proxy(domain) ]
       s.t.  α ≥ 0, Σα = 1
```

但难直接求 → DRO（Distributionally Robust Optimization）：交替更新 model & α。

---

## Slide 7 · Doremi 结果

The Pile 22 domain：

```
              MAX of  Domain
手工 weight   = 5%    (Pile-CC 等比)
Doremi weight = 50%   (CC) / 12% (代码) / ...
                     ↓
              下游平均 ppl ↓ 2-3%
```

效果不是革命性，但稳定改善。

---

## Slide 8 · RegMix (2024)

ByteDance：用回归代替 minimax。

```
1. 训 K=64 个小代理 (1B token each)
2. 每个用不同配比
3. 拟合 ppl ↔ 配比 的回归曲面
4. 找回归极值
```

K 大 → 更准；K 小 → 更省。RegMix 1B-token 代理找出 7B 的最优配比。

---

## Slide 9 · DCLM 比赛设计

Apple + UW 2024：
- **给定** 240T raw CC web
- **任务**：参赛者清洗 → 4T 子集
- **裁判**：训 7B 模型 → MMLU + 综合 acc
- **奖**：超过 baseline 上 leaderboard

→ 让"清洗 + 配比"成为可比较任务。

---

## Slide 10 · DCLM-Baseline 配方

DCLM 团队自己的 baseline：

```
- 启发式：URL filter + Gopher rules
- dedup: MinHash threshold 0.7
- 质量：fastText classifier (高/低质量)
- 配比：web 100%（DCLM 仅 web）
- 4 T tokens
```

7B 模型 → 64% MMLU（与 Llama-3 持平）。

---

## Slide 11 · "scaling laws" 解 mix

经典 Chinchilla：

```
L(N, D) = A · N^{-α} + B · D^{-β} + E
```

加入 mix 维度：

```
L(N, D, α_mix) = ...
```

→ 各 domain 独立 scaling，最优 mix 应等化 marginal loss reduction。

---

## Slide 12 · 多目标 mix

不同评测目标 → 不同最优 mix：

```
通用 LLM           → 高 web
数学 / 推理         → 高 math + arxiv
代码               → 高 github
对话               → 高 dialog + reddit
```

实务：**任务定后选最优 mix**。

---

## Slide 13 · 阶段化 mix（curriculum）

Llama-3 / Phi-3 实务：

```
早期    web 50% / code 30% / wiki 10% / math 10%
中期    保持
末期    切高质量子集 (FineWeb-Edu / textbooks) 5-10%
```

最后 5-10% 是"annealing"，决定模型最终能力天花板。

---

## Slide 14 · "mid-training" 概念

```
预训练 (10T) → mid-training (1T) → SFT (100M) → RLHF
                  ↓
            注入新能力（多语言/数学/code）
```

中段训练用专门 mix（高质量、强标）。Llama-3 / Phi-4 / Qwen-2.5 都做。

---

## Slide 15 · ablation 实验设计

最小可靠 ablation：

```
- 代理模型 ~ 1B
- token 量 ~ 30B（chinchilla optimal）
- 评测：MMLU(5-shot) + HellaSwag + GSM8K-tiny
- 重复 3 seed
```

成本 ~ 单卡 5090 24h × 配比数。8 个配比 ~ 1 周。

---

## Slide 16 · "domain"如何分类

实务两种粒度：

```
粗：web / code / wiki / books / math
细：CC-2024-12 / CC-2024-11 / code-py / code-js / arxiv-math / arxiv-bio / ...
```

细粒度更精确但 ablation 维度爆炸；多数研究停在 5-10 domain。

---

## Slide 17 · Mixtral / Phi-4 的 mix

Phi-4：~50% 合成 + 30% web + 20% 其他。打破"以 web 为主"的旧范式。

Mixtral / DeepSeek：仍以 web 为主，但精筛。

→ 2025 后期"合成 ≥ web"是新趋势。

---

## Slide 18 · 合成数据的"配比"维度

```
multi-step reasoning  : Magpie reasoning template
roleplay              : 合成 character & dialog
math                  : MetaMath synth
code                  : self-instruct code
```

每种合成是一个"领域"，权重需独立 ablation。

---

## Slide 19 · 评估配比是否最优

```
原 mix:         MMLU = 0.42
+ 数学 +5pct:   MMLU = 0.43, GSM8K +3pt
+ code +5pct:   MMLU = 0.43, HumanEval +2pt
- web -10pct:   MMLU = 0.41, 综合 -1pt
```

各指标变动 ≤ 0.5pp 不显著（需更多 seed）。

---

## Slide 20 · 配比 → 训练日历

实务 12 month 的 mix 决策：

```
Q1: 手工估配比 + 训 1B 代理 ablation
Q2: 训 7-13B 模型，验证配比
Q3: 优化 mid-training mix
Q4: 上 大型，最终 mix 锁定
```

mix 决定权一般在 Tech Lead 而非 trainer。

---

## Slide 21 · 实务代码框架

```python
class MixSampler:
    def __init__(self, weights: dict[str, float], shard_dirs: dict):
        self.weights = weights
        self.shards  = {d: shuffled_jsonl(shard_dirs[d]) for d in weights}

    def __iter__(self):
        while True:
            d = random.choices(list(self.weights), self.weights.values())[0]
            yield next(self.shards[d])
```

权重定后每个 batch 按概率混合。

---

## Slide 22 · 课后思考

1. Doremi 收敛到 worst-case 真的最优吗？
2. 如果你只能选 5 个 domain 训通用 LLM，怎么排序？
3. 高质量合成数据应占多少？为什么？
4. 配比应在每个 step 固定 vs 阶段化？

---

## 参考

- Doremi: Xie 2023 (Google)
- RegMix: 2024
- DCLM: Apple 2024
- Phi-4 technical report 2024
