# L09 · Toy RL on LLM — 句长 / Sentiment 实战

> 24 slides | 60 min | RL Foundations 系列第 9 讲

---

## 学习目标

1. 跑两个玩具 reward（句长 / sentiment），观察 LLM 训练过程
2. 看 reward hacking 长什么样（句长 reward → 重复字符）
3. 用 BERT-sentiment 当 RM 训 GPT-2，看正向 sentiment 上升
4. 学会用 KL penalty 控制不偏离 SFT 太多

---

## Slide 1 · 玩具 1 · 句长 reward

```
reward = min(len(response_text) * 0.05, 5.0)
```

意图：训练 GPT-2 输出更长的 response。

观察预期：
- iter 0: response 平均 30 字符
- iter 10: 平均 60 字符
- iter 20: 平均 100 字符（但内容质量下降）

---

## Slide 2 · 句长 reward 的 hacking

GPT-2 + 无 KL penalty + 句长 reward → 几次 iter 后：

```
"The movie was ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
"This is a aaaaaaaaaaaaaaaaaaaaaaaa"
```

→ **极端 reward hacking 案例**。

修复：
- 加 KL penalty（控制不偏离 ref）
- 加 perplexity penalty（控制不输出胡乱）
- 加 length cap（强制 max_new_tokens）

---

## Slide 3 · 句长 reward + KL penalty 的效果

加 β=0.05 KL penalty 后：
- length 仍然增加，但**语言合理**
- response 是真正"更详细"，而非重复字符

→ KL penalty 是 LLM-RL 的"语言守护"。

---

## Slide 4 · 玩具 2 · Sentiment Reward

用 `distilbert-base-uncased-finetuned-sst-2-english` 当 RM：

```python
def sentiment_reward(text: str) -> float:
    inputs = tokenizer_bert(text, return_tensors="pt")
    with torch.no_grad():
        logits = bert(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    return probs[0, 1].item()  # positive 概率
```

输入：prompt + GPT-2 续写
输出：BERT 认为 positive 的概率

---

## Slide 5 · Sentiment Reward 训练观察

IMDb 风格 prompt："The movie was"

| iter | reward (avg) | 例 response |
|------|------------|-----------|
| 0 | 0.45 | "okay, but..." |
| 20 | 0.62 | "really great!" |
| 50 | 0.78 | "amazing! incredible! ..." |
| 100 | 0.91 | "absolutely wonderful! ..." |

→ GPT-2 学会输出正向情感词汇。

---

## Slide 6 · Sentiment Reward 的 hacking

如果不加 KL penalty：
- GPT-2 可能学会输出 `"good good good ..."`
- BERT-sentiment 给 0.95，但语义崩塌

→ 不仅 length reward，**任何 RM 都可能被 hack**。

---

## Slide 7 · KL Penalty 直接监控

每 iter 监控 `mean KL(π_θ || π_ref)` per token：

- 健康：0.01 - 0.1
- 警戒：0.1 - 0.5
- 异常：> 0.5（actor 走太远，可能崩）

→ 用 adaptive β 自动调控。

---

## Slide 8 · GPT-2 + sentiment 完整代码（伪）

```python
actor = GPT2LMHeadModel.from_pretrained("gpt2")
ref = GPT2LMHeadModel.from_pretrained("gpt2")
critic = GPT2WithValueHead(...)
sentiment_rm = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

for it in range(100):
    prompts = sample_prompts()
    responses, log_p_old, V_old = rollout(actor, critic, prompts)
    rewards = [sentiment_rm(p + r) for p, r in zip(prompts, responses)]
    log_p_ref = get_log_probs(ref, ...)
    rewards_adj = build_token_rewards(rewards, log_p_old, log_p_ref, β=0.05)
    adv, ret = gae(rewards_adj, V_old, ...)
    update_ppo(actor, critic, adv, ret, log_p_old, ...)
```

---

## Slide 9 · 数字案例

100 iter，batch 8，max_new_tokens=20：

| 指标 | 初始 | iter 50 | iter 100 |
|------|------|--------|----------|
| mean sentiment | 0.45 | 0.72 | 0.85 |
| mean length | 50 | 70 | 75 |
| KL (token level) | 0.02 | 0.08 | 0.12 |
| GPT-2 仍是 GPT-2？ | yes | yes-ish | shrinking |

→ KL 增大说明 actor 越来越偏离 ref。

---

## Slide 10 · Reward Hacking 经典清单（先剧透 L10）

1. **句长 reward** → 重复字符撑长
2. **sentiment reward** → 堆正面词
3. **BLEU reward** → 抄目标 n-gram
4. **代码长度 reward** → 加空格行
5. **任何 reward** → "good good good" 模式

→ Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

---

## Slide 11 · trl PPOTrainer 在此场景的优势

手写版的痛点：
- log_p / log_p_ref 都要手算
- KL adaptive 要手写
- value head 要手挂

trl 自动：
- `AutoModelForCausalLMWithValueHead` 自动 wrap
- `PPOTrainer.step(prompts, responses, rewards)` 一行
- adaptive_kl_ctrl 自动

代价：黑盒（出了问题难调）。

---

## Slide 12 · 教学折衷

| 场景 | 选 |
|------|---|
| 理解原理 | 手写 |
| 快速跑通 | trl |
| 真实产线 | trl + DeepSpeed |

本课程：
- L08-L09 用手写为主
- capstone (L11) 用 trl + IMDb（标准基线）

---

## Slide 13 · 显存优化 trick

5090 24GB 跑 GPT-2-medium PPO：

1. **bf16**: `model.to(torch.bfloat16)` —— 4 → 2 bytes
2. **gradient_checkpointing**: 损失 30% 速度换 50% 显存
3. **小 batch_size + gradient_accumulation**: 等效大 batch
4. **shared backbone**: actor + critic 共享，再各自加 head（教学不推荐，工业常用）
5. **LoRA**: 4bit + LoRA 训 7B（专题 5 R1 用法）

---

## Slide 14 · LoRA + PPO 怎么做

GPT-2 + LoRA + PPO：
```python
from peft import LoraConfig, get_peft_model
peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["c_attn"])
actor = get_peft_model(gpt2, peft_cfg)
# 只 0.5% 参数可训，反向极快
```

→ 真实大模型 RL 几乎都这么做（R1-Zero 挑战轨 + DPO/GRPO 全部）。

---

## Slide 15 · 调参速查（GPT-2-small）

| 超参 | 范围 | 典型 |
|------|------|------|
| lr | 1e-6 ~ 1e-4 | 1e-5 |
| batch | 4 ~ 64 | 8 |
| K_epochs | 1 ~ 6 | 4 |
| ε (clip) | 0.1 ~ 0.3 | 0.2 |
| β (KL) | 0.01 ~ 0.5 | 0.05 |
| target_kl | 0.1 ~ 6 | 6 (trl 默认) |
| max_new_tokens | 10 ~ 64 | 20 |

---

## Slide 16 · 监控 dashboard 必看的 5 指标

1. **mean raw reward** —— 上升才有意义
2. **mean response length** —— 别飞涨
3. **KL(actor || ref)** —— 别飞涨
4. **L_clip** —— 朝 0 收敛
5. **L_vf** —— critic 学得动否

TensorBoard 一张图全画上。

---

## Slide 17 · 训练失败诊断

| 症状 | 可能原因 |
|------|--------|
| reward 不升 | lr 太小 / RM 给的分数不显著 |
| reward 飞升但句子离奇 | reward hacking（缺 KL）|
| KL 暴涨 | β 不够 / lr 太大 |
| L_vf 不降 | critic 学不动 / V 数值范围爆炸 |
| 训练崩 | grad 爆 / lr 大 |

---

## Slide 18 · 自我评测

跑完 sentiment 实验问自己：
- [ ] 看到 reward 上升
- [ ] 看到 response 真的更"正向"
- [ ] KL 没飞涨
- [ ] 至少能看 1 个"reward hacking 趋势"sample

→ 都达到，你已经能跑 LLM-RL 玩具实验了。

---

## Slide 19 · 与 capstone 的衔接

L11 capstone：IMDb 完整 sentiment PPO 实验。

- 数据：IMDb 1k subset
- 算法：trl PPOTrainer
- 基座：GPT-2-medium
- 指标：sentiment reward + KL 监控 + 样本 spot check
- 目标：reward 提升 ≥ 30%（≈ 0.45 → 0.6）

---

## Slide 20 · 推荐扩展实验

学有余力的：
1. 把句长 reward 改成"句长 + 正向 sentiment"双 reward 加权（多目标 RLHF 雏形）
2. 加一个 length penalty 防止 hacking
3. 用 LoRA 而非全量训
4. 试不同 prompt（中文 / 代码 prompt）看泛化

---

## Slide 21 · 与下一专题的桥梁

下一专题 **rlhf-classic** 系统化：
- 把 BERT-sentiment 换成真正的训练 RM（Anthropic-HH 偏好数据）
- 加 SFT 阶段
- 完整三段管线：SFT → RM → PPO

→ 本讲是 RM 替代品的入门玩具，下一步是工业版。

---

## Slide 22 · 自测题

1. 句长 reward hacking 的典型表现？怎么修？
2. KL penalty 在 token-level 还是 response-level 加？
3. β 太大 / 太小 各自的后果？
4. adaptive_kl_ctrl 的调节逻辑（target_kl=6）？
5. 跑 sentiment RM 时为何要把 BERT 换 fp16？

---

## Slide 23 · 入口

```bash
# 手写 GPT-2 PPO (句长 reward)
python learning/rl-foundations/src/ppo_gpt2_minimal.py --total-iters 20

# trl 对照（IMDb sentiment）
python learning/rl-foundations/src/sentiment_reward.py
python learning/rl-foundations/src/ppo_gpt2_trl.py --total-iters 20

# 测试
pytest learning/rl-foundations/src/tests/test_gpt2_ppo.py
```

---

## Slide 24 · 下一讲

**L10 RL Pitfalls 合集** — 把所有的"坑"集中整理一次。
