# L11 · Capstone — GPT-2-medium + IMDb 情感 PPO

> 24 slides | 60 min | RL Foundations 系列 Capstone

---

## Capstone 目标

**任务**：用 PPO 训 GPT-2-medium，让它对 IMDb 评论开头续写出**显著更正向**的文本。

**指标**：sentiment reward 提升 ≥ 30%（典型 0.45 → 0.6+）。

**预算**：5090 24GB，约 4-6 小时。

**输出**：notebook + tensorboard 日志 + 5 个 spot-check 样本。

---

## Slide 1 · 任务设置

| 项 | 值 |
|----|---|
| 基座 | GPT-2-medium (355M) |
| RM | distilbert-SST-2 |
| Dataset | IMDb 1k subset（仅取前 30 token 当 prompt） |
| Algorithm | trl PPOTrainer |
| Batch size | 16 |
| Max new tokens | 30 |
| KL target | 6 (adap) |
| 训练 step | 256 step batch 16 |
| Wall time | ~5 h |

---

## Slide 2 · 数据准备

```python
from datasets import load_dataset
ds = load_dataset("imdb", split="train[:1000]")
prompts = [t[:120] for t in ds["text"]]  # 取前 120 字符
# 再 tokenize 截断到 30 token
```

注意：IMDb 文本既有正面也有负面 → prompt 可能引导 GPT-2 续写负面 → PPO 任务是"把 GPT-2 续写偏向正面"，这种"调头"很考验 RL。

---

## Slide 3 · trl PPO 实现

```python
config = PPOConfig(
    model_name="gpt2-medium",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    cliprange=0.2,
    vf_coef=0.1,
    init_kl_coef=0.05,
    target_kl=6.0,
    adap_kl_ctrl=True,
)
trainer = PPOTrainer(config, model, ref_model, tokenizer)
```

详见 `src/capstone_imdb_ppo.py`。

---

## Slide 4 · Training Loop

```python
for batch in dataloader:
    prompt_tensors = [tokenize(p) for p in batch["text"]]
    response_tensors = trainer.generate(prompt_tensors, max_new_tokens=30)
    responses_text = [tokenizer.decode(r) for r in response_tensors]
    rewards = sentiment_rm.score([p + r for p, r in zip(batch, responses_text)])
    stats = trainer.step(prompt_tensors, response_tensors, rewards.tolist())
    log_to_tensorboard(stats, mean_reward=rewards.mean())
```

---

## Slide 5 · 显存预估

GPT-2-medium 355M：
- actor + opt: 4.2 GB
- ref (frozen, fp16): 0.7 GB
- value head: 几 MB
- BERT RM (frozen, fp16): 0.13 GB
- activations + KV: 6 GB
- **总**: 11 GB

→ 24GB 5090 完全够。

---

## Slide 6 · 监控仪表盘

每 iter 记录到 tensorboard：

| 指标 | 期望趋势 |
|------|--------|
| mean_reward | ↑ 单调 |
| mean_kl | 0.05 - 0.15 稳定 |
| mean_response_length | 25-30 稳定 |
| L_clip | 收敛到 0 |
| L_vf | 持续下降 |
| entropy | 缓慢下降，> 0.3 |

任何一项异常 → 立刻停 → 看样本。

---

## Slide 7 · Spot Check 样本

训完每 50 iter 随机抽 5 个样本看：

```
prompt: "I just watched this movie and"
iter 0: " it was okay, nothing special."
iter 100: " it was incredibly entertaining and the actors were great!"
iter 200: " it was absolutely the best movie I've ever seen, fantastic!"
```

观察：句子是否仍**通顺**、**多样**？还是已开始 hack（"amazing amazing amazing"）？

---

## Slide 8 · 失败模式 1 · Reward Hacking

迹象：
- mean_reward 单调上升 0.7+
- 但 spot check 显示重复"good wonderful amazing"

修复：
- 加大 KL β
- 加 length penalty
- 重训 RM（更鲁棒）

---

## Slide 9 · 失败模式 2 · 训练崩塌

迹象：
- 某 iter reward 突跌
- L_clip / L_vf 突然爆炸
- KL > 1.0

修复：
- lr ÷ 2 重训
- max_grad_norm 加严
- 检查上一 iter 的样本

---

## Slide 10 · 失败模式 3 · KL 飞涨

迹象：
- KL 单调上升不收敛

修复：
- target_kl 6 → 4
- init_kl_coef 0.05 → 0.1
- adaptive 调 β

---

## Slide 11 · 评估方法

1. **Mean reward** before vs after
2. **Distribution of reward** (histogram)
3. **Mean response length** before vs after
4. **Mean KL(actor || ref)** at the end
5. **Spot check 10 samples** — 人工评：是更正向？是否通顺？

---

## Slide 12 · 完整 pipeline 流程图

```
IMDb 1k --(取前 120 字符 → tokenize)→ prompts
                  |
                  v
              Rollout
                  |  GPT-2-medium 生成 30 token
                  v
        prompts + responses
                  |
                  v
       BERT-SST2 RM 打分
                  |
                  v
      PPOTrainer.step()
       (内部：KL adj + GAE + clip + K epoch)
                  |
                  v
        wandb / tensorboard
```

---

## Slide 13 · 完整脚本入口

```bash
# 训完整 capstone（5090 24GB，约 4-6 h）
python learning/rl-foundations/src/capstone_imdb_ppo.py \
    --total-iters 256 \
    --batch-size 16 \
    --max-new-tokens 30 \
    --tb-log-dir runs/capstone_imdb

# 监控
tensorboard --logdir runs/capstone_imdb

# 评估
python learning/rl-foundations/src/capstone_imdb_ppo.py --eval-only \
    --ckpt runs/capstone_imdb/final
```

---

## Slide 14 · 期望结果

跑完 ≥ 30 iter，应看到：

| 指标 | 期望值 |
|------|------|
| mean reward (start) | 0.45 ± 0.05 |
| mean reward (end) | ≥ 0.6 |
| KL @ end | < 0.2 |
| mean length | 25-32 |
| spot check 5/5 都是正向 | ✓ |

→ 提升 ≥ 30% 算 capstone PASS。

---

## Slide 15 · 与下一专题的桥梁

本 capstone 用 BERT-SST2 当 RM。下一专题 `rlhf-classic` 升级为：

| 阶段 | 本 capstone | 专题 2 capstone |
|------|------------|---------------|
| SFT | 跳过（用 GPT-2 base）| 用 SFT-trainer 微调 |
| RM | 用 BERT-SST2 | 训自己的 RM（BT loss） |
| PPO | trl PPOTrainer | trl PPOTrainer |
| 数据 | IMDb | Anthropic-HH 1k |
| 任务 | sentiment 调头 | helpful + harmless |

---

## Slide 16 · 系列 takeaway 自评

跑完 capstone 你应该能回答：

- [ ] PPO clip 的几何意义
- [ ] GAE λ=0.95 的 trade-off
- [ ] 为什么 LLM-RL 要 4 model
- [ ] KL ref penalty 的两种实现方式（reward 内 vs loss 外）
- [ ] adaptive_kl_ctrl 的调节逻辑
- [ ] reward hacking 的至少 3 类 + 防御策略
- [ ] PPO 7 件套清单 + 各自影响

→ 都能答上你已掌握 80% RL 基础。

---

## Slide 17 · capstone 提交清单

- [ ] tensorboard 截图：mean_reward / KL / L_clip 三条
- [ ] 5 个 spot check 样本（iter 0 / 100 / 200 各 5 个）
- [ ] 一段总结：observed reward 提升 + 是否 hacking
- [ ] 提交：commit ckpt 到 `runs/capstone_imdb/`

---

## Slide 18 · 排错小贴士

| 问题 | 解决 |
|------|------|
| trl 0.13 接口变化 | 看官方 issue tracker |
| OOM | batch_size ÷ 2 / max_new_tokens ÷ 2 / bf16 |
| 训不动 | lr ↑ / KL β ↓ |
| 训崩 | lr ↓ / KL β ↑ |
| BERT 加载慢 | 缓存到本地 |

---

## Slide 19 · 自测题

1. 为什么用 IMDb 而非随机 prompt？
2. trl PPOTrainer 的 4 model 怎么管理显存？
3. capstone PASS 标准是什么？
4. 如果 mean_reward 上升但 spot check 都是 garbage，是否 PASS？
5. 训完如何决定模型是否值得保留？

---

## Slide 20 · 与 R1 时代的对照

R1 时代（专题 5 capstone）任务：
- Countdown-3 / GSM8K
- 算法：GRPO (无 critic, group baseline)
- Reward：format + accuracy（rule-based, 无 RM）
- 期待：aha moment

**对照本 capstone 的差异**：
- 用 rule-based reward 而非 RM
- 无 critic → 更省显存但更高方差
- 期待"涌现"现象

→ R1 capstone 是本 capstone 的逻辑延续。

---

## Slide 21 · 系列里程碑

完成本 capstone 后：

```
tag: rl-foundations ✓
methods: 12 (REINFORCE / A2C / TRPO / PPO / GAE / 7 tricks / KL pen)
hours: 14
lectures: 12
notebooks: 12
```

下一程：专题 2 RLHF Classic（InstructGPT 三段管线）。

---

## Slide 22 · 总结一句话

> 学完 PPO 基础你就有了 RL 大半工具箱。后续每一个新算法（DPO / GRPO / DAPO / VAPO / ...）都是 PPO 的变体或简化。

---

## Slide 23 · 入口（最后）

```bash
# Capstone
python learning/rl-foundations/src/capstone_imdb_ppo.py

# Notebook
jupyter notebook learning/rl-foundations/notebooks/11-capstone-ppo-llm.ipynb

# 测试
pytest learning/rl-foundations/src/tests/ -v
```

下一讲：**L12 总结 + 引出 RLHF**。

---

## Slide 24 · 检查表

- [ ] 完整 capstone 至少跑过 1 次（30 iter 起）
- [ ] tensorboard 看 5 条曲线
- [ ] 自评 spot check 5 个样本
- [ ] mean reward 提升 ≥ 30%（≥ 0.58）
- [ ] 写下 3 句话总结
