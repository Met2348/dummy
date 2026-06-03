# L14 · Capstone Track B — Qwen-1.5B + GSM8K 挑战轨

> 14 slides | 45 min | 真训练 + aha emergence ⭐⭐⭐⭐⭐

---

## Slide 1 · Track B 目标

**真训**：看 aha moment 涌现.
- base: **Qwen2.5-1.5B-Base**
- quant: **4bit nf4 (bitsandbytes)**
- LoRA: r=16 on q/k/v/o
- task: **GSM8K-tiny** (500 train + 100 test)
- algo: **GRPO**
- 显存: 5090 24GB OK
- 时长: ~4h

---

## Slide 2 · 与 Track A 的区别

| 维度 | Track A | Track B |
|------|---------|---------|
| base | GPT-2-M | Qwen-1.5B |
| task | Countdown | GSM8K |
| algo | 4 对照 | GRPO 单 |
| 显存 | 24GB 余裕 | 24GB 紧 |
| 时长 | 6h 总 | 4h 单 |
| 看点 | pipeline | **aha** |

---

## Slide 3 · 显存配方 (5090 24GB)

```
4bit base    ~3GB
+ LoRA      ~0.3GB
+ ref (frozen) ~3GB
+ rollout KV  ~10GB
+ grad + adam ~6GB
= ~22GB (紧)
```

→ 需要 max_response_len ≤ 256, k=4。

---

## Slide 4 · LoRA + 4bit setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32,
                  target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(model, lora)
```

---

## Slide 5 · GSM8K-tiny

500 训 + 100 测，从 openai/gsm8k 取子集：
```python
ds = load_dataset("openai/gsm8k", "main", split="train")
ds_tiny = ds.shuffle(seed=42).select(range(500))
```

---

## Slide 6 · 训练超参

```yaml
algo: GRPO
k: 4
clip_eps: 0.2
beta_kl: 0.04
lr: 5e-6
max_response_len: 256
rollout_batch: 16
ppo_epochs: 2
total_steps: 1000
temperature: 0.7
```

---

## Slide 7 · aha 词频监控

每 100 step 评估：
```python
def aha_ratio(responses):
    aha_words = ["wait", "let me reconsider", "actually", "rethink"]
    has_aha = sum(1 for r in responses
                  if any(w in r.lower() for w in aha_words))
    return has_aha / len(responses)
```

→ 目标 ≥ 5%。

---

## Slide 8 · 预期曲线

```
step 0:    acc 5%   len 100  aha 0%
step 200:  acc 12%  len 150  aha 1%
step 500:  acc 20%  len 200  aha 4%
step 1000: acc 25%  len 250  aha 7%  ⭐
```

→ aha 词频升至 ≥ 5% = 涌现成功。

---

## Slide 9 · 实战入口

```bash
python learning/reasoning-r1/src/r1_zero_track_b.py \
  --base Qwen/Qwen2.5-1.5B --steps 1000
```

writes `runs/track_b/{step}/{aha_words.json, len_dist.csv}`.

---

## Slide 10 · 失败排查

| 现象 | 修 |
|------|---|
| OOM | k 减半 / max_resp 减半 |
| 不 aha | 加 step / 加 max_resp_len |
| reward 不动 | beta 调小 → 0.02 |
| KL 飞 | beta 调大 → 0.08 |
| LoRA 不学 | r 加大 → 32 |

---

## Slide 11 · 评估 reasoning quality

不仅看 accuracy。还要 spot check：
- 推理是否合理？
- 是否中英混（R1-Zero 现象）？
- 长度是否合理（不灌水）？

→ 每 100 step 看 10 个样本。

---

## Slide 12 · 与 Spurious Rewards 警示对照

L12 讲：Qwen + 随机 reward 也涨 21pp。
→ Track B 必须：
- held-out test (100 题不见过)
- 报告 vs base 而非 vs random
- spot check 实际推理

---

## Slide 13 · Track B 退出条件

- [ ] 1000 step 完整跑完
- [ ] GSM8K test accuracy ≥ 20% (base ~5%)
- [ ] response length ≥ 200 (vs base 100)
- [ ] aha 词频 ≥ 5%
- [ ] 10 个 spot check 样本：推理质量合格

→ 任一不达成可写为 "教学完成 + R1-Zero 现象局部观察"。

---

## Slide 14 · 一句话总结

> Track B = 个人 dev 在 24GB 上看 aha emergence 的可行路径。复现 TinyZero 的精神，2025 标配 capstone。

🎓 **Topic 5 reasoning-r1 系列高峰 完结。**
下一讲 L15 — R1 时代 takeaway 总结。
