# L03 · Reward Model — Bradley-Terry 实战

> 22 slides | 55 min | RLHF Classic 第 3 讲

---

## 学习目标

1. 推导 Bradley-Terry pairwise ranking loss
2. 实现 RM = LM + scalar head + BT loss
3. 在 Anthropic-HH 1k 上训练 + 评估准确率
4. 看清 RM 训练的 3 个常见陷阱（length bias / sycophancy / position）

---

## Slide 1 · 为什么不直接 regress 分数

不直接学 `r(x, y) = scalar_in_[0,1]`，因为：
- 人没有"绝对分数"的共识
- 不同标注者尺度不一（A 给 0.7，B 给 0.9 是同样满意）
- 比较是更稳定的信号

→ 改用 pairwise：**A 比 B 好** 这种判断标注者间一致性高。

---

## Slide 2 · BT 模型回顾

每个 response y 有 "skill" `r(y)`。

P(y_w 战胜 y_l) = sigmoid(r(y_w) - r(y_l))

数据 (y_w, y_l)：人类排序。

最大化 likelihood = 最小化 BT loss：

```
L_BT = -E_(y_w, y_l) [ log sigmoid(r(y_w) - r(y_l)) ]
```

---

## Slide 3 · RM 网络

```python
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm
        hidden = base_lm.config.hidden_size
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.lm(input_ids, attention_mask=attention_mask,
                      output_hidden_states=True)
        h = out.hidden_states[-1]  # (B, T, hidden)
        # 取最后一个非 pad token 的 hidden 作为 reward
        last_idx = attention_mask.sum(-1) - 1
        r = self.v_head(h[range(len(h)), last_idx]).squeeze(-1)
        return r
```

→ 关键：**取最后一个非 pad token** 当 reward，不是 mean pool。

---

## Slide 4 · 训练 loop（手写）

```python
for batch in dataloader:
    chosen_ids, chosen_mask = batch["chosen"]
    rejected_ids, rejected_mask = batch["rejected"]

    r_chosen = model(chosen_ids, chosen_mask)
    r_rejected = model(rejected_ids, rejected_mask)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    opt.zero_grad(); loss.backward(); opt.step()
```

注意：chosen / rejected 必须分别 forward（不能 concat batch，因 attention pattern 不同）。

---

## Slide 5 · trl RewardTrainer

```python
from trl import RewardTrainer, RewardConfig

trainer = RewardTrainer(
    model=model,
    args=RewardConfig(output_dir="rm", num_train_epochs=1,
                      per_device_train_batch_size=8,
                      learning_rate=1e-5),
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

trl 自动处理 chosen/rejected forward + BT loss。

---

## Slide 6 · 数据准备 · Anthropic-HH

```python
ds = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
# 每条 {"chosen": str, "rejected": str}

def tokenize(ex):
    return {
        "chosen_ids": tokenizer(ex["chosen"], truncation=True, max_length=512)["input_ids"],
        "rejected_ids": tokenizer(ex["rejected"], ...)["input_ids"],
    }
ds = ds.map(tokenize)
```

注意 padding：trl 自动；手写需 `pad_and_mask`。

---

## Slide 7 · 评估指标

```python
def evaluate(model, eval_ds):
    correct = 0
    for ex in eval_ds:
        r_w = model(ex["chosen_ids"])
        r_l = model(ex["rejected_ids"])
        if r_w > r_l: correct += 1
    return correct / len(eval_ds)
```

**Accuracy = % of pairs where r_chosen > r_rejected.**

| 数据 | 期望 RM acc |
|------|---------|
| Anthropic-HH (test) | 0.65 - 0.75 |
| 单一任务 | 0.80+ |
| 自然标注噪声 | 上限 0.73-0.78 |

---

## Slide 8 · 陷阱 1 · Length Bias

**症状**：RM 偏好长 response（无论质量）。

**原因**：训练数据中 chosen 平均比 rejected 长 → RM 学到"长 = 好"。

**修复**：
- 数据集 length 中性化（chosen/rejected 长度匹配）
- ODIN：训练时随机交换 chosen/rejected 的 prefix 长度
- 加 length-normalized 项

---

## Slide 9 · 陷阱 2 · Sycophancy

**症状**：用户说"我同意 X"，RM 偏好回答"X 对"，无论是否真对。

**原因**：训练数据中 labeler 倾向于赞同自己已有观点的回答。

**修复**：
- Constitutional 训练（Anthropic 思路）
- adversarial 数据
- 多视角偏好

---

## Slide 10 · 陷阱 3 · Position Bias

**症状**：labeler 标注时偏好第一个看到的 response。

**修复**：
- 随机化呈现顺序
- 多 labeler 多 round

---

## Slide 11 · RM 训练超参（推荐）

| 超参 | 值 |
|------|---|
| base | GPT-2-medium / Llama-3-8B |
| lr | 1e-5 |
| batch | 8 |
| epoch | 1 |
| max_length | 512 |
| weight decay | 0.01 |

→ 1 epoch 通常够，过拟合风险高。

---

## Slide 12 · RM 大小的影响

InstructGPT 用 6B RM 训 175B actor —— 看似 mismatch。

经验：
- RM ≥ 1B 显著优于小 RM
- RM 与 actor 同尺寸最稳
- 175B + 6B 是为了"省成本 + 6B 够强"

实际：现代 RLHF 多用 7B-13B RM。

---

## Slide 13 · RM 的反向工程：看 model 怎么"打分"

```python
samples = ["Response A...", "Response B...", "Response C..."]
scores = [model(tokenize(s)) for s in samples]
# 排序 + 看是否符合人类直觉
```

→ 训完后 spot check，看 RM 是不是真在区分质量。

---

## Slide 14 · 进阶：用 LoRA 训 RM

LoRA 适合 RM（全量训太费）：

```python
from peft import LoraConfig, get_peft_model
peft_cfg = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_cfg)
# 显存 ↓ 80%，准确率几乎不变
```

→ 工业 RM 几乎都是 LoRA 训。

---

## Slide 15 · Capstone preview · 训 RM 阶段

L11 capstone 中：
- 数据：summarize_from_feedback 1k pair
- 基座：GPT-2-medium + LoRA
- 训 1 epoch（约 30 min）
- 期望 RM accuracy > 60%

---

## Slide 16 · 与 DPO 的对比（剧透专题 3）

DPO 不训 RM，直接用偏好数据训 LLM：

```
L_DPO = -log sigmoid( β · log(π_θ(y_w)/π_ref(y_w)) - β · log(π_θ(y_l)/π_ref(y_l)) )
```

→ 把 "r = β log(π_θ / π_ref)" 当作隐式 RM，再用 BT loss。

**完全跳过显式 RM 训练**。

---

## Slide 17 · 多 RM 加权（多目标）

实际场景需要多个 RM：
- helpful RM
- harmless RM
- truthful RM

加权：
```
total_reward = w_h · R_help + w_s · R_safe + w_t · R_truth
```

经验：归一化每个 R 到 [0, 1] 后加权；w 调比例。

---

## Slide 18 · RM 的 calibration

**问题**：RM 输出的 logits 可能不稳定（数量级飘）。

**修复**：
- 训完做 z-score normalization
- 用 sigmoid(r) ∈ [0, 1]
- 或用 quantile 校准

---

## Slide 19 · RM 后处理：clip / threshold

PPO 训练时，extreme RM scores 容易 dominate advantage：
- clip `r ∈ [-5, 5]`
- 或者 z-score per batch

→ 防止 reward outlier 破坏训练。

---

## Slide 20 · 自测题

1. 为什么用 pairwise 而非 absolute 评分？
2. RM 取"最后一个非 pad token"hidden 的原因？
3. Length bias 怎么诊断？怎么修？
4. 175B actor 配 6B RM 合理吗？
5. DPO 跟显式 RM 的关系？

---

## Slide 21 · 入口

```bash
# 训 RM toy
python learning/rlhf-classic/src/rm_minimal.py
python learning/rlhf-classic/src/rm_trl.py

# 测试
pytest learning/rlhf-classic/src/tests/test_rm_consistency.py
```

---

## Slide 22 · 下一讲

**L04 PPO for LLM 深化** —— 把 RM 当 reward 跑 PPO，完整三段管线。
