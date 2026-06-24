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

## 运行验证（Runbook）

> 本段命令即 [`runbook.yaml`](runbook.yaml) 登记的"文档入口命令"，已在 ERIC-3080Ti（RTX 3080 Ti 16GB）上 V0+V1 验证通过（6/6 PASS）。
> 一键复验本模块：
> ```powershell
> python scripts/eric_3080ti_env_audit.py --runbook --modules multimodal-agent
> ```

本模块 6 个 demo **全部无 argparse、无可调预算** → 直跑到完成即验证（runbook 里 `v0: false`，跳过 `--help` 探针）：

```powershell
# VLM-R1：规则 reward（counting / grounding-IoU / format）+ GRPO 训练伪代码
python learning/multimodal-agent/src/vlm_r1_minimal.py
# Vision-R1：模态桥接冷启动 + HFRRF 硬格式奖励 + GRPO z-score advantage + PTST 阶段
python learning/multimodal-agent/src/vision_r1_original_minimal.py
# s1 budget forcing：think<min 注入 "Wait" 续推 / think>max 截断（双侧预算）
python learning/multimodal-agent/src/s1_budget_forcing.py
# Safe-RLHF：拉格朗日对偶 max R_helpful s.t. R_harmless≥阈值，λ 违约上升
python learning/multimodal-agent/src/safe_rlhf_minimal.py
# 五线综合等价：LoRA≡Parallel-Adapter / DPO≡BT-implicit-RM / GRPO advantage mean=0
python learning/multimodal-agent/src/unified_view.py
# 毕业 Capstone：同一道 GSM8K 题 × 5 路径响应对照
python learning/multimodal-agent/src/capstone_graduation.py
```

> ℹ️ **本地无需重型依赖即可全跑**：`src/` 全是手写 toy，真实 import 仅 `re` / `torch`。
> 讲义里的 `Qwen2-VL` / `peft` / `PIL` / `verl` 只出现在 `vlm_r1_minimal.py` 的 `vlm_r1_setup_code()`
> **返回的伪代码字符串**里（不会执行）。VLM 用符号化 image_facts / 数字答案模拟，**不加载真视觉模型**。
> 因此这些 demo 全在 CPU 小张量上跑（`gpu: false`），秒级完成。
>
> ℹ️ **mock 诚实标注**：`capstone_graduation.py` 的 5 路径 response 是 **mock 字面量**（非真加载 5 个 ckpt，
> 代码与输出均标注"真实需加载 5 个 ckpt"）；`vlm_r1_setup_code()` 是伪代码——二者均不冒充"已跑通真实训练"。
> 真实 VLM-R1 GRPO 训练需 Qwen2-VL-2B + 4bit + LoRA（讲义 L02 Slide 9 估 ~22GB，5090 量级）。

**测试（V2）**：

```powershell
python -m pytest learning/multimodal-agent/src/tests/ -v
# 或经审计 harness：python scripts/eric_3080ti_env_audit.py --modules multimodal-agent --tests
```

