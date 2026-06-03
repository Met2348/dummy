# Multimodal + Agent + 毕业 ⭐⭐⭐⭐⭐

> 专题 7 / 7 — **系列收官**。VLM-R1 / WebRL / SWE-Gym + 五线综合理论高峰 + 毕业作品。
>
> Design: [link](../../docs/superpowers/specs/2026-06-03-multimodal-agent-design.md)
> Plan: [link](../../docs/superpowers/plans/2026-06-03-multimodal-agent.md)

## 已完成（本轮）

- L13 五线综合统一 lecture (32 slides) ⭐⭐⭐⭐⭐ 系列理论高峰
- src/unified_view.py: 等价关系数值验证（LoRA ≡ Parallel Adapter / DPO ≡ BT-implicit-RM / GRPO advantage）
- environment / papers / README

## 待续

L01 Vision-R1 / L02 VLM-R1 / L03 Kimi k1.5 vision
L04 WebRL / L05 SWE-Gym / L06 ComputerRL / L07 ToolRL
L08 Long Context RL / L09 s1 + Don't Overthink / L10 Claude 4 + Gemini 2.5
L11 Safety + Constitutional Classifiers
L12 Capstone-1: VLM-R1 玩具 (Qwen2-VL-2B + CLEVR counting)
**L14 ⭐⭐⭐⭐⭐ 毕业 Capstone: 5 个 ckpt 同题对照（Vanilla / LoRA / Adapter / DPO / R1-Zero）**

## 毕业 Capstone (L14)

```
同一道 GSM8K 题（Janet 鸡蛋）：

1. Vanilla GPT-2-base       → 朴素续写
2. LoRA 版本 (lora-family L01 ckpt)
3. Pfeiffer Adapter (adapter L01 ckpt)
4. DPO 版本 (dpo-family L01 ckpt)
5. R1-Zero 版本 (reasoning-r1 capstone-A ckpt)

输出：5 个 response 对比 + 推理过程可视化 + 性能表
```

🎓 **毕业证书**：完成 7 专题 = 128 方法 / 90 lecture / ~130h

## 入口

```bash
python learning/multimodal-agent/src/unified_view.py   # 数值验证
```

