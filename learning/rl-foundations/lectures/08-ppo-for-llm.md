# L08 · PPO for LLM — 第一次接触 LLM-RL

> 26 slides | 65 min | RL Foundations 系列第 8 讲

---

## 学习目标

1. 看清 token-level PPO 与 CartPole PPO 的相同与不同
2. 理解 LLM-PPO 的 **4 模型协同**（actor / critic / ref / RM）
3. 显存账：为何 GPT-2-small + 24GB 才勉强够
4. KL penalty 的作用与 adaptive 控制
5. Value head 怎么加在 LM 上

---

## Slide 1 · 同与不同

| | CartPole | LLM |
|---|----------|-----|
| state | 4 dim | token ids（变长）|
| action | 2 | vocab_size（50257 for GPT-2）|
| reward | 每步 +1 | 通常仅末端给（RM 打分）|
| ep len | ≤ 500 | response 长度 ≤ 256~2k |
| 算法 | PPO clip | PPO clip + KL ref penalty |
| 模型 | 2 head MLP | 4 个 LLM（actor/critic/ref/RM）|

---

## Slide 2 · LLM 是 stochastic policy

```
π(a_t | s_t; θ_LM) = softmax(LM_θ(s_t))_at
```

- `s_t` = `[prompt, response[:t]]`（已生成 token 前缀）
- `a_t` = response 第 t 个 token
- `π(a|s)` = LM 在该 token 位的概率

→ 与 categorical policy 完全一致，只是 vocab 大得多。

---

## Slide 3 · Reward 的来源：RM

RLHF 三段管线（专题 2 完整讲）：
1. SFT：用 instruction-response pair 微调 LM
2. RM：用人类偏好 pair 训 reward model（输出 scalar）
3. PPO：用 RM 当 reward 训 actor

本讲不训 RM，**直接用 BERT-sentiment 当 reward**：
- prompt = IMDb 影评开头
- response = LLM 续写
- reward = `BERT_sent(prompt + response)` 的正向情感概率

---

## Slide 4 · Reward 何时给

最简单：**末端唯一 reward**
```
r_t = 0 for t < T
r_T = RM(prompt + response)
```

GAE 把这个末端 reward 反向"广播"到每个 token，得到 token-level advantage。

变体：每个 token 都有 reward（如长度 reward `+1/len`），下一讲。

---

## Slide 5 · 4 模型协同

```
   prompt → ACTOR (LM_θ)   →  response tokens
                                |
            CRITIC (LM_φ + V head) → V(s_t) for each step
                                |
              REF (LM_ref, frozen) → π_ref(a_t|s_t) → KL ref penalty
                                |
                       RM (frozen) → reward at step T
```

actor + critic 通常**独立**（避免 RM 与 actor 相互渗透）；ref + RM 都 frozen。

---

## Slide 6 · Value Head 怎么加

```python
class GPT2WithValueHead(nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        self.gpt2 = gpt2_model
        self.v_head = nn.Linear(gpt2.config.hidden_size, 1)

    def forward(self, input_ids):
        out = self.gpt2(input_ids, output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, T, hidden)
        V = self.v_head(h).squeeze(-1)  # (B, T)
        return out.logits, V
```

每个 token 位都有一个 V，配合 GAE 算 token-level advantage。

---

## Slide 7 · KL Penalty：为什么需要

无 KL：PPO 可能把 LM 推到"reward 高但语言塌缩"的位置：
- 反复输出 "good good good"（IMDb sentiment 很高，但毫无语义）
- 这是 **reward hacking** 经典案例

加 KL penalty：限制 actor 与 ref（SFT 完成的 LM）不太远。

```
r_t' = r_t - β · log(π_θ(a_t|s_t) / π_ref(a_t|s_t))
```

token-level KL，加在 reward 上。

---

## Slide 8 · KL Penalty 数学性质

`r_t' = r_t - β · log(π_θ / π_ref)` 等价于在 reward 上加 KL 惩罚。

最终 expected reward：
```
E[Σ r_t'] = E[Σ r_t] - β · KL(π_θ || π_ref)
```

→ PPO 优化等价于最大化 `RM 分数 - β · KL(π_θ || π_ref)`。

---

## Slide 9 · β 怎么选 · Adaptive KL

固定 β 难调：
- 太大 → actor 学不出去
- 太小 → reward hacking

**InstructGPT 的做法**：
```python
if KL > 1.5 * target_KL:    β ←  β · 1.5
if KL < 0.5 * target_KL:    β ←  β / 1.5
```

target_KL 经验值 = 6 (RLHF 论文)，CartPole 不需要。

---

## Slide 10 · token-level vs response-level loss

两种聚合 PPO loss 的方式：

**token-level**（DAPO 推荐）：
```
L = mean over (B, T) tokens [ -min(r·A, clip(r,1±ε)·A) ]
```

**response-level**（PPO 原始）：
```
L = mean over (B) [ Σ_t -min(...) / T ]
```

差异：response-level 让长 response 每 token 权重小；token-level 一视同仁。

→ R1 时代发现 token-level 显著更好（DAPO Token-PG Loss）。

---

## Slide 11 · 显存账（GPT-2-small, 124M）

| 模型 | bytes/param | size |
|------|------------|------|
| actor | 4 (fp32) | 500 MB |
| actor opt state (Adam: m + v) | 8 | 1 GB |
| critic | 4 + 8 | 1.5 GB |
| ref (frozen) | 4 (可 fp16) | 250 MB |
| RM (frozen, BERT) | 4 (可 fp16) | 250 MB |
| activations / KV cache | 变 | ~5 GB |
| **总** | | **~8 GB** |

→ 24GB 5090 跑 GPT-2-small 完全可以；GPT-2-medium (355M) 也勉强。

---

## Slide 12 · GPT-2-medium 的紧迫

| 模型 | bytes/param | size |
|------|------------|------|
| actor + opt | 12 | 4.2 GB |
| critic + opt | 12 | 4.2 GB |
| ref + RM | 8 | 1 GB |
| activations | | ~10 GB |
| **总** | | **~20 GB** |

→ 24GB 5090 跑 medium 必须 fp16/bf16 + 小 batch。本讲后续 capstone 用 medium。

---

## Slide 13 · trl PPOTrainer 速览

```python
from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLMWithValueHead

actor = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = PPOConfig(model_name="gpt2", learning_rate=1.41e-5, batch_size=32)
trainer = PPOTrainer(config, actor, ref, tokenizer)

for prompts in dataloader:
    responses = trainer.generate(prompts, max_new_tokens=20)
    rewards = compute_reward(prompts, responses)  # 用 BERT-sentiment
    trainer.step(prompts, responses, rewards)
```

注：trl 0.13 后接口稍变，详见 `src/ppo_gpt2_trl.py`。

---

## Slide 14 · 手写 PPO for LLM 的差异

`src/ppo_gpt2_minimal.py` 与 CartPole PPO 的差异：

1. obs = token ids (变长) → 需要 pad + mask
2. action = vocab_size → log_prob 是 categorical at vocab 大小
3. reward 仅末端 → 用 GAE 反向广播
4. KL penalty in reward
5. 4 model 协同（vs 2 head）

---

## Slide 15 · rollout for LLM

```python
def rollout(actor, tokenizer, prompts: list[str], max_new_tokens=20):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = actor.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, top_p=0.9, temperature=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return outputs.sequences, outputs.scores
```

`scores` 是每步 token 的 logits → 算 log_prob_old 用。

---

## Slide 16 · 算 log_prob 与 KL

```python
def get_log_probs(model, input_ids, attention_mask):
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]                   # 预测下一 token
    targets = input_ids[:, 1:]                       # 真实下一 token
    log_pi = F.log_softmax(logits, dim=-1)
    return log_pi.gather(2, targets.unsqueeze(-1)).squeeze(-1)
# log_probs shape: (B, T-1)
```

KL ref penalty:
```python
log_p_act = get_log_probs(actor, ...)
log_p_ref = get_log_probs(ref, ...)
kl = (log_p_act - log_p_ref)   # per-token KL
reward_adjusted = reward - beta * kl
```

---

## Slide 17 · token-level GAE

```python
# rewards: (B, T)，多数为 0，仅末端非零（外加 KL penalty 项）
# values: (B, T) from critic
# dones: (B, T)，仅末端为 1
adv, ret = compute_gae(rewards, values, dones, ...)
```

→ GAE 自动把末端 reward 反向广播到每个 token。

---

## Slide 18 · 训练 loop 骨架

```python
for it in range(N_iters):
    # 1. Rollout
    prompts = sample_batch_prompts()
    responses, log_probs_old, values_old = rollout(actor, prompts)

    # 2. Reward
    raw_rewards = RM(prompts, responses)
    log_p_ref = get_log_probs(ref, ...)
    rewards_per_token = build_token_reward(raw_rewards, log_probs_old, log_p_ref, beta)

    # 3. GAE
    adv, ret = compute_gae(rewards_per_token, values_old, dones)

    # 4. K epoch × M minibatch
    for epoch in range(K):
        for mb in minibatches:
            update_actor_critic(mb)

    # 5. KL adaptive
    actual_kl = compute_kl(log_probs_old, log_probs_new)
    beta = update_beta(beta, actual_kl, target_kl)
```

---

## Slide 19 · 与 CartPole 对照（一图）

```
                    CartPole          LLM-RL
state               4 dim             token seq
action              2                 vocab (50k)
reward              every step        only end + KL
ref model           ✗                 ✓ frozen LM
critic backbone     small MLP         LM + V head
KL adaptive         ✗                 ✓ adaptive β
modes               2                 50,000 of which only "few good"
```

---

## Slide 20 · 训练时长（GPT-2-medium，5090）

| 阶段 | 时长 |
|------|------|
| 1 iter rollout + reward | 30 s |
| 1 iter update | 60 s |
| 100 iter（典型 toy） | ~2.5 h |
| 1k iter（capstone） | ~25 h |

→ LLM-RL 一夜起步。CartPole 几分钟。

---

## Slide 21 · 早期问题清单

| 问题 | 现象 | 修复 |
|------|------|------|
| KL 飞涨 | β 不够强 | adaptive 调 β / target_kl 6 |
| Response 长度暴增 | reward 偏长 | 加 length penalty |
| reward 上升但生成胡言乱语 | reward hacking | 加 KL penalty / 看样本 |
| 训练崩 | lr 大 / clip 不当 | lr=1e-5 / max_grad_norm=1.0 |

---

## Slide 22 · 与 RLHF / R1 的桥梁

本讲教 PPO for LLM。下一专题（**rlhf-classic**）讲：
- SFT → RM 训练 → PPO 完整三段管线
- InstructGPT 论文复刻
- LLaMA-2 / Sparrow / Constitutional AI

R1 时代（**reasoning-r1**）讲：
- GRPO = 去 critic 的 PPO + group baseline
- R1-Zero 的 rule-based reward（无 RM 也行）
- DAPO 的四件套优化

→ 你现在学的 PPO 是基础，后面变体都从这里来。

---

## Slide 23 · 自测题

1. LLM-PPO 与 CartPole-PPO 的本质区别？
2. 4 模型协同各自负责什么？
3. 为什么需要 KL ref penalty？不加会怎样？
4. GPT-2-small 需要多少显存？为何 medium 紧？
5. token-level 与 response-level PPO loss 的差异？

---

## Slide 24 · 阅读

- **必读**：Ziegler 2019（LLM-RLHF 起源）
- **必读**：Stiennon 2020（TL;DR 摘要 RLHF）
- 推荐：trl `PPOTrainer` 源码
- 跳过：Schulman 原 PPO 论文的 KL penalty 实验

---

## Slide 25 · 实战代码入口

```bash
# 手写 PPO for GPT-2
python learning/rl-foundations/src/ppo_gpt2_minimal.py --total-iters 20

# trl 对照
python learning/rl-foundations/src/ppo_gpt2_trl.py --total-iters 20

# 测试
pytest learning/rl-foundations/src/tests/test_gpt2_ppo.py
```

---

## Slide 26 · 下一讲

**L09 toy RL-LLM** — 把 PPO 跑到具体的"句长奖励 / sentiment 奖励"上，亲眼看 reward hacking 长什么样。
