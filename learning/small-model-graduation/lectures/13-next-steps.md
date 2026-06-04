# L13 · 下一程 — Module 4 「改大模型」

> 12 slides | 35 min ⭐⭐⭐⭐⭐

## Slide 1 · Module 3 结束 → Module 4 开始

```
Module 3: 造大模型 (本系列)
  ↓ ckpt E 已有
Module 4: 改大模型 (SFT/PEFT/RLHF/RL reasoning)
  ↓ 在 ckpt E 上 SFT/RL
Module 5/6/7: 多模态 / Agent / 安全
```

## Slide 2 · Module 4 = "改" 五线汇总

```
五线综合 (已在 plan 文件):
1. Prompt tuning (input)
2. LoRA (weight)
3. Adapter (structure)
4. RLHF / DPO (distribution)
5. R1 / GRPO (trajectory)
```

## Slide 3 · 用 ckpt E 做 SFT 第一步

```python
# Module 4 Topic 2 RLHF-classic / Topic 3 DPO 起点
model = load("ckpt_E.pt")
sft_data = load_dataset("alpaca")
train(model, sft_data, lr=2e-5, max_step=1000)
```

## Slide 4 · 7 个 Module 4 专题

```
专题 1: rl-foundations    (RL 基础)
专题 2: rlhf-classic       (InstructGPT)
专题 3: dpo-family         (DPO/IPO/KTO/...)
专题 4: process-reward     (PRM)
专题 5: reasoning-r1       (R1 复现)
专题 6: rl-sota-2026       (DAPO/PRIME)
专题 7: multimodal-agent   (VLM-R1)
```

## Slide 5 · 时间预期

```
Module 4: 88 方法 / 90 lectures / ~ 101h
Module 5/6/7: 待规划
Module 3 (本) + 4: ~ 220h
```

## Slide 6 · 推荐顺序

```
完成 Module 3 capstone E
→ 立刻进 Module 4 Topic 1 (RL 基础)
→ 顺次至 Topic 7 (毕业)
```

## Slide 7 · 复用本系列工件

```
- src/phi_tiny_model.py: 基础模型
- src/training_loop.py: train loop 模板
- ckpt_E.pt: base ckpt
- 数据 pipeline (Topic 1)
- 评测脚本 (Topic 7)
```

## Slide 8 · 不同 module 的工程难度

```
Module 3 (本):
  数据 + 训练 + infra, 单 5090 24G 可玩
Module 4 (SFT/RL):
  显存翻倍 (4-model PPO), 推荐 24-80 GB
Module 5+: 多模态 + 多卡
```

## Slide 9 · "改" vs "造" 的本质区别

```
造:  从无 → 有 (pretraining loss)
改:  从有 → 更对齐/能力 (后训练 loss)
  - SFT: 学指令格式
  - RLHF: 学人偏好
  - R1: 学推理过程
```

## Slide 10 · 仍是 next-token prediction

```
SFT: CE on instruction tuning data
DPO: implicit RM (BT loss)
GRPO: explicit reward + KL penalty
本质都是改 token-level distribution
```

## Slide 11 · 个人项目示例

```
1. ckpt E (本系列) → SFT 1k step (Alpaca-tiny)
2. SFT → DPO 500 step (Anthropic-HH-tiny)
3. DPO → GRPO 200 step (GSM8K-tiny)
4. GRPO → 部署 vLLM
5. 同 prompt 测 5 ckpt 对照
```

## Slide 12 · 总结

```
Module 3 结束 = 造小模型完成
ckpt E 是下一程基石
"改大模型" 是更大冒险
但已学会"造", 难度可控
```

## 参考
- Module 4 plan (docs/plan/造改 plan)
- 本系列 README
