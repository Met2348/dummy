# L12 · Capstone — DAPO 4 件套加在 R1-Zero baseline

> 14 slides | 35 min | RL SOTA 2026 实操收官

---

## Slide 1 · Capstone 目标

承接 Topic 5 capstone-A R1-Zero baseline ckpt，逐 trick 加 DAPO 4 件套，看每个贡献:
- config 0: 原 GRPO
- config 1: + Clip-Higher
- config 2: + + Dynamic Sampling
- config 3: + + + Token-level PG
- config 4: + + + + Overlong (= DAPO 全开)

---

## Slide 2 · 设置

```yaml
base: 专题 5 R1-Zero baseline ckpt (Qwen-1.5B + LoRA)
task: GSM8K-tiny (100 测)
incremental_steps: 200
per_config_time: ~1h
total: 5 config × 1h = 5h
```

---

## Slide 3 · DAPO 4 件套（回顾 L01）

```
1. Clip-Higher: ε_high=0.28 (vs sym 0.2)
2. Dynamic Sampling: rollout 直到有对有错
3. Token-level PG: mean over (B*k*T) 而非 (B*k)
4. Overlong Shaping: sigmoid 软惩罚
```

---

## Slide 4 · 监控指标

每 50 step:
- accuracy on held-out
- mean response length
- KL(actor || ref)
- entropy
- aha 词频

5 个 config × 4 个指标 = 20 条曲线。

---

## Slide 5 · 预期对照表

| config | acc | Δ | len | aha% |
|--------|-----|---|-----|------|
| GRPO baseline | 20% | — | 200 | 3% |
| + Clip-Higher | 22.5% | +2.5 | 220 | 4% |
| + + Dynamic Sampling | 24% | +1.5 | 230 | 5% |
| + + + Token-level | 29% | +5.0 | 270 | 7% |
| + + + + Overlong | 31.5% | +2.5 | 285 | 8% |

Token-level 最大贡献。

---

## Slide 6 · Trick 1 单独贡献

Clip-Higher：
- 早 1000 step 几乎无差异
- 后期（KL 高时）+3pp
- 副作用：entropy 升 25%（多探索）

→ 必加。

---

## Slide 7 · Trick 2 单独贡献

Dynamic Sampling：
- 节省 30% 无效 rollout
- accuracy +1-2pp
- 实现复杂（rollout loop 重写）

→ 看时间 budget。

---

## Slide 8 · Trick 3 单独贡献（最大）

Token-level PG：
- accuracy +5pp（最大单 trick 收益）
- 长 response 中关键 token 的更新生效
- length +20%
- 实现简单（1 行）

→ **必加**。

---

## Slide 9 · Trick 4 单独贡献

Overlong Shaping：
- 在 max_response_len=4096 时 +3pp
- 在 max=256 时几乎无效（很少 truncate）

→ 与 max_response_len 配合。

---

## Slide 10 · 资源约束下的选择

```
budget 紧 (单卡 6h):
    必加: Token-level PG (+5pp)
    推荐: Clip-Higher (+2.5pp, 零成本)
    skip: Dynamic Sampling (复杂)
    可选: Overlong (看 max_response_len)
```

→ Token-level + Clip-Higher 是 minimum 80% 收益。

---

## Slide 11 · 输出 jsonl

每 config 训完 dump:
```json
{
  "config": "clip_higher",
  "step": 200,
  "metrics": {
    "accuracy": 0.225,
    "mean_length": 220,
    "kl": 4.5,
    "aha_freq": 0.04
  }
}
```

后续 jupyter 画 5 condition × 4 metric 矩阵图。

---

## Slide 12 · 与 verl 对照

verl 0.4+ 的 DAPO recipe：
```yaml
algorithm:
  adv_estimator: grpo
  clip_low: 0.2
  clip_high: 0.28
  use_token_level_loss: true
data:
  rollout:
    dynamic_sampling: true
reward:
  shaping:
    type: overlong_soft
    target_len: 4096
```

→ 一键全开。

---

## Slide 13 · 退出条件

- [ ] 5 个 config 全跑完
- [ ] DAPO 总收益 ≥ +8pp vs baseline
- [ ] Token-level 单 trick 收益 ≥ +3pp
- [ ] 4 个 trick 曲线对照图完成
- [ ] aha 词频 ≥ 5% (任一 config)

---

## Slide 14 · 一句话总结

> Capstone 验证 DAPO 4 件套累加性。Token-level 最大，Clip-Higher 零成本，Dynamic 与 Overlong 看资源。

🎓 **Topic 6 RL SOTA 2026 完结。**
下一专题 7 — 多模态 + Agent + 五线综合毕业。
