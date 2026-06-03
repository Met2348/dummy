# L03 · DeepSeek-R1-Zero — 纯 RL 涌现 aha moment

> 18 slides | 55 min | R1 时代爆发点 ⭐⭐⭐⭐⭐

---

## Slide 1 · 历史时刻

2025-01-20，DeepSeek 同时发布 R1 + R1-Zero。**论文核心震撼**：
> 不要 SFT，纯 RL on base model 也能学出推理.

行业地震。

---

## Slide 2 · 与 OpenAI o1 关系

| 维度 | OpenAI o1 (2024.09) | DeepSeek R1-Zero (2025.01) |
|------|---------------------|----------------------------|
| 闭源/开源 | 闭源 | **开源** ⭐ |
| 训法 | 未公开 | GRPO + rule reward |
| cold start | ? | 无 SFT，base 直 RL |
| 结果 | AIME 74 | AIME 71 |

→ R1-Zero 证明 o1 范式可被开源复现。

---

## Slide 3 · 训练设置（论文）

```
base: DeepSeek-V3-Base (671B MoE)
algo: GRPO (k=64 rollouts)
reward:
    + format reward (正确 <think></think><answer></answer>)
    + accuracy reward (rule-based verifier, 数学)
no SFT, no PRM, no value model
```

---

## Slide 4 · GRPO 在 R1-Zero 的作用

GRPO = PPO 去 critic + group baseline:
- 不需要 value model (省 ¾ 显存)
- group z-score 当 advantage
- KL ref penalty 防漂

→ R1-Zero 因为 671B 不可能再 +critic，GRPO 是必选。

---

## Slide 5 · Reward 设计（极简）

```python
def reward(response, ground_truth):
    fmt = check_format(response)              # 0 or 1
    acc = check_answer(response, ground_truth)  # 0 or 1
    return 0.1 * fmt + 0.9 * acc
```

无 RM，无 PRM。只有规则。

---

## Slide 6 · 训练过程奇景

观察图（论文 Figure 2）：
- step 0-2k: response length 短 (~100 token)
- step 2k-4k: length 缓慢上升
- step 4k+: length **暴涨** (300 → 800 → 1500)
- 同步：accuracy 30% → 60% → 71%

→ 模型自己发现"想更久 = 答得更好"。

---

## Slide 7 · aha moment 涌现

论文图：训练中段，model 开始说：
- "Wait, let me reconsider..."
- "I made a mistake earlier..."
- "Let me check this differently..."

这些 token 在 base model 中没有。**自发涌现**。

---

## Slide 8 · 为什么 base 能这么训

```
1. base model 已经"见过" CoT 数据 (pretraining)
2. base model 已经会"推理"，只是不会"用对格式"
3. GRPO + rule reward 给信号让它"激活"
4. KL ref penalty 防止漂出能力区
```

→ R1-Zero 不是教 base 推理，是让它"用上"已有能力。

---

## Slide 9 · 训练负担

R1-Zero 训练规模:
- ~10k step (DeepSeek V3 用)
- 64 rollouts × 4096 max_resp_len
- 每 step ~1h on 1024 H800
- **总 ~10000 H800-hour**

→ 不是 hobby 工作。但 TinyZero ($30) 证明小规模可行。

---

## Slide 10 · R1-Zero vs R1

R1-Zero: pure RL，会"思考"但"难读"（混语言）。
R1: 加 cold-start SFT → readable，最终模型。

```
base → R1-Zero (pure RL) → R1-Zero-Distill 数据 → SFT → 再 RL → R1
```

R1 是 R1-Zero 的工程化包装。

---

## Slide 11 · R1-Zero 的 readability 问题

R1-Zero 输出常见：
- 中英混合 (中文是 zh: 它会切到中文 reasoning)
- 数学符号与文本混
- 不分段

R1 的 cold-start SFT 修了这个。但 R1-Zero **更接近 raw RL 学到的东西**。

---

## Slide 12 · 复现潮 (2025.01-2025.06)

| 项目 | 团队 | 基座 | 关键贡献 |
|------|------|-----|---------|
| TinyZero | UC Berkeley | Qwen-3B | $30 复现 aha |
| Open-R1 | HuggingFace | Qwen-7B | 完整开源流程 |
| Open-Reasoner-Zero | StepFun | Qwen-7B | 1/10 训练步数 |
| SimpleRL-Zoo | HKUST | Llama/Qwen | 多基座对照 |
| Mini-R1 | Phil Schmid | Qwen-1.5B | 教学版 |

→ 4 个月内 5+ 复现。

---

## Slide 13 · TinyZero 的关键启示

$30 完成 = 单卡 5090 24h:
- base = Qwen2.5-3B
- task = Countdown (比 GSM8K 简单)
- reward = format + answer
- 训 4 小时可见 aha

→ 个人 dev 也可以训。

---

## Slide 14 · R1-Zero 的边界

R1-Zero 范式适用于：
- ✅ 数学 (verifier OK)
- ✅ 代码 (test pass)
- ✅ 形式化推理
- ❌ 开放对话 (无 verifier)
- ❌ 创作 (主观)
- ❌ 多步 tool use (没演示)

→ 这是 R1 路线的内禀限制。

---

## Slide 15 · 数学推理的特殊性

为什么 R1-Zero 在数学涌现 aha：
- 数学：reward 信号 100% 可靠 (verifier)
- 数学：推理路径有多样性（多种解法）
- 数学：pretrain 数据丰富 (math.SE, arxiv)
- 数学：error 可分阶段定位 (rethink 有意义)

→ 不是所有任务都能套这套。

---

## Slide 16 · 工程 takeaway

1. **Verifier first**: 没 rule-based reward 别启动
2. **base model 够好**: < 1B 难看到 aha
3. **GRPO 而非 PPO**: 显存制约
4. **max_response_len 大**: 1024 → 4096 → 8192
5. **温度高**: 0.7-1.0 (要探索)
6. **训长**: 至少 1k step

---

## Slide 17 · 与 DPO/RLHF 对照

| 维度 | RLHF | DPO | R1-Zero |
|------|------|-----|---------|
| 数据 | preference | preference | (question, answer) |
| RM | 训 | 隐式 | rule |
| 适用 | 通用对齐 | 通用对齐 | 推理 |
| 涌现 | 无 | 无 | **aha** ⭐ |

---

## Slide 18 · 一句话总结

> R1-Zero = base model + GRPO + verifier reward = 自发涌现"思考"。**开源版 o1 起点**。

下一讲 L04 — R1 (cold-start + 4 阶段)。
