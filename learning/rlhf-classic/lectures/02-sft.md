# L02 · SFT (Supervised Fine-Tuning)

> 14 slides | 45 min | RLHF 三段管线第一段

---

## Slide 1 · 为什么 SFT 是 RLHF 第一段

InstructGPT 三段：
```
[1] SFT      → 教会 instruction-following
[2] RM       → 学会评分
[3] PPO      → 优化分布
```

- 没 SFT 的 base model 完全不会 instruction
- SFT 是 RL 的"起跳台"——给 PPO 一个像样的起点

---

## Slide 2 · SFT 数据格式

```
{
  "prompt":   "Translate to French: Hello world",
  "response": " Bonjour le monde"
}
```

- prompt 不计 loss（label = -100）
- response 计 next-token NLL

---

## Slide 3 · loss 公式

```
L_SFT = -E[ sum_t log π(y_t | x, y_<t) ]
```

只对 response token 求和。代码：
```python
labels[prompt_len:] = response_ids
labels[:prompt_len] = -100   # ignored by cross_entropy
loss = F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)
```

---

## Slide 4 · 不 mask prompt 会怎样

- 加入 prompt loss → model 学到"复述 prompt"
- 实测：MT-Bench -2-3pp
- HuggingFace TRL `SFTTrainer` 默认 mask prompt

---

## Slide 5 · packed sequence

显存效率高的做法：把多条短样本 pack 进 max_len，用 attention mask 分隔。
```
[Q1] [A1] [EOS] [Q2] [A2] [EOS] [pad...]
```
- 训练快 2-3×
- 但需要 attention mask block-causal（不让 Q2 看 Q1）

---

## Slide 6 · 学习率与 epoch

经验：
- lr = 1e-5 ~ 5e-5 (full fine-tune)
- lr = 1e-4 ~ 5e-4 (LoRA)
- epoch = 2-3 (overfit 后掉 generalization)

---

## Slide 7 · 数据规模

| 模型 | SFT 数据量 |
|------|----------|
| InstructGPT | 13k 人工 |
| LLaMA-2-chat | 27k 高质量人工 |
| Vicuna | 70k ShareGPT |
| WizardLM | 70k Evol-Instruct |

**质 > 量**：Anthropic 实测 5k 精选 > 50k 噪声。

---

## Slide 8 · LIMA 的"少即多"

LIMA 论文 (Zhou 2023)：
- 1000 条精选示例
- 击败 InstructGPT (13k 数据)

→ 启示：base model 已经会，SFT 只是"激活"。

---

## Slide 9 · 三轨实现

```
sft_minimal.py    手写 mask + cross_entropy
sft_trl.py        SFTTrainer (packed sequence, lora 支持)
sft_axolotl       生产 (axolotl yaml 配置)
```

---

## Slide 10 · SFT 与 LoRA 组合

工程标配：
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base, config)
# 之后正常 SFT，只更新 LoRA
```

显存 -75%，效果接近 full fine-tune。

---

## Slide 11 · 评估 SFT 模型

- **MT-Bench** (80 题，GPT-4 judge)
- **AlpacaEval** (805 题，对照 GPT-4 win rate)
- **MMLU** (基础知识保持率)

SFT 后 MMLU 下降 1-3pp 是正常代价。

---

## Slide 12 · 与 RM 阶段衔接

SFT model 既是：
- PPO 的 actor 初始化
- PPO 的 ref model (KL 约束的"原点")
- RM 训练的 backbone（共享，加 reward head）

→ 一个 SFT ckpt 支撑后续全部三段。

---

## Slide 13 · 常见坑

| 问题 | 修 |
|------|---|
| 输出循环重复 | 数据质量+repetition_penalty |
| 拒答能力下降 | safety 数据混入 |
| 多轮 chat 不对 | dialog template 错 |
| EOS 不学 | 数据要带 `</s>` |

---

## Slide 14 · 一句话总结

> SFT = base model 的"礼仪课"。教它"接到 prompt 就回答"，为 RM/PPO 提供起跳点。

下一讲 L03 — Reward Model 训练。
