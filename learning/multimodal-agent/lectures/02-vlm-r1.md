# L02 · VLM-R1 — GRPO 直接训 VLM

> 16 slides | 45 min | 多模态 R1 范式

---

## Slide 1 · 故事

2025.02，OM-AI Lab 发布 VLM-R1。
> 跳过 SFT，直接 GRPO 训 Qwen2-VL，counting/grounding 任务 +15-25pp.

→ R1 范式从 text 跨到 vision。

---

## Slide 2 · 设置

```
base: Qwen2-VL-2B-Instruct (或 7B)
algo: GRPO (k=4)
task:
    - counting (CLEVR)
    - grounding (RefCOCO)
    - visual reasoning (MathVista)
reward: format + accuracy (rule-based)
```

无 SFT cold-start。

---

## Slide 3 · Counting 任务 reward

```python
def counting_reward(response, gt_count):
    # 提 <answer>N</answer> 或 末尾数字
    n = extract_number(response)
    return 1.0 if n == gt_count else 0.0
```

简单规则即可。

---

## Slide 4 · Grounding 任务 reward

```python
def grounding_reward(pred_bbox, gt_bbox, iou_th=0.5):
    iou = compute_iou(pred_bbox, gt_bbox)
    return 1.0 if iou > iou_th else iou
```

IoU 当 reward，可 soft (linear) 或 hard (threshold)。

---

## Slide 5 · 训练 pipeline

```
1. 图像 + 问题 → Qwen2-VL → 4 个候选
2. parse 每个 answer
3. 规则打分
4. group advantage (z-score)
5. GRPO update (LoRA-only)
6. 显存 ~22GB (5090 OK)
```

---

## Slide 6 · 与 Vision-R1 的不同

| 维度 | Vision-R1 | VLM-R1 |
|------|-----------|--------|
| cold-start | 有 (50k SFT) | **无** |
| algo | PPO | GRPO |
| 复杂度 | 高 | 低 |
| 适用 | 复杂 reasoning | counting/grounding |

VLM-R1 更直接复刻 R1-Zero 的"纯 RL"精神。

---

## Slide 7 · 实测结果

CLEVR counting:
| 方法 | accuracy |
|------|---------|
| Qwen2-VL-2B base | 35% |
| + SFT | 52% |
| + Vision-R1 (cold + RL) | 68% |
| **+ VLM-R1 (纯 RL)** | **62%** |

→ 略低于 Vision-R1 但工程简单 10x。

---

## Slide 8 · CoT 涌现

训练后 VLM 输出：
```
<think>I see 3 red cubes, 2 blue spheres, and 1 yellow cylinder.
Let me count red ones: 1, 2, 3. Three red cubes.</think>
<answer>3</answer>
```

→ visual CoT 涌现，与 text R1-Zero 类似。

---

## Slide 9 · 显存 + 速度

Qwen2-VL-2B + 4bit + LoRA + GRPO k=4:
```
Model: ~6GB (4bit)
LoRA: ~0.5GB
KV cache for k=4 rollout: ~10GB
gradients: ~5GB
Total: ~22GB on 5090
```

速度: ~1 epoch / 30min on 1k CLEVR。

---

## Slide 10 · 与 Kimi k1.5 vision 对比

Kimi: long context + 多模态 + RL 联合训。
VLM-R1: 简化 (只多模态 + R1)。

Kimi 强但难复现；VLM-R1 是社区可复现版。

---

## Slide 11 · 工程坑

| 坑 | 修 |
|----|---|
| 图像 token 太多 OOM | image resolution ↓ |
| 数字 parse 漏 | 多 regex pattern |
| reward 噪声大 | k ↑ + 平均 |
| 格式不收敛 | format reward 权重 ↑ |

---

## Slide 12 · 应用拓展

VLM-R1 范式可用于:
- 文档理解 (Doc-VQA)
- 表格推理 (TableQA)
- chart/plot 分析
- OCR + reasoning
- robotic perception

→ 任何 visual + symbolic reasoning 任务。

---

## Slide 13 · 与 Vision Transformer Encoder 关系

GRPO 只调 LoRA on LLM head，**vision encoder 完全冻结**。
- 优势：vision 知识保留
- 劣势：不能学新视觉 pattern

→ 简单/快但天花板低。

---

## Slide 14 · 三轨实现

```
vlm_r1_minimal.py   reward 函数 + setup 伪代码
vlm_r1_grpo.py      完整训练 (verl 风格)
vlm_r1_axolotl     yaml 配置
```

---

## Slide 15 · 与 Capstone-1 衔接

下一讲 Capstone-1: 在 Qwen2-VL-2B + CLEVR 上跑 GRPO，看 counting accuracy 上升。

---

## Slide 16 · 一句话总结

> VLM-R1 = Qwen2-VL + GRPO + 规则 reward。多模态 R1 范式起点，开源社区可复现。

下一讲 L03 — Kimi k1.5 vision 部分。
